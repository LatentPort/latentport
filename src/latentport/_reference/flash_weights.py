"""Load the pinned BF16 text weights with memory-mapped experts and n-gram embeddings."""

import json
from pathlib import Path

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding
from torch import nn
from transformers import AutoConfig, Qwen4ExpForCausalLM


class Store:
    def __init__(self, snapshot):
        self.root = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]

    def __getitem__(self, name):
        from safetensors import safe_open

        filename = self.index[name]
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Invalid shard filename for {name}")
        with safe_open(str(self.root / filename), framework="pt", device="cpu") as shard:
            return shard.get_tensor(name)


class ShardedEmbedding(nn.Module):
    def __init__(self, ref, shards):
        super().__init__()
        self.weight = ref.weight
        self.shards = shards
        self.ends = torch.tensor([len(s) for s in shards]).cumsum(0)
        if (self.ends[-1], shards[0].size(1)) != ref.weight.shape:
            raise ValueError("Ngram table shape mismatch")

    def forward(self, ids):
        flat = ids.cpu().reshape(-1)
        which = torch.bucketize(flat, self.ends, right=True)
        output = torch.empty((len(flat), self.weight.shape[1]), dtype=self.shards[0].dtype)
        for idx in which.unique().tolist():
            pos = torch.where(which == idx)[0]
            rows = flat[pos] - (self.ends[idx - 1] if idx else 0)
            output[pos] = self.shards[idx][rows]
        return output.view(*ids.shape, -1).to(ids.device)


class StreamExperts(nn.Module):
    def __init__(self, ref, gate, down):
        super().__init__()
        if gate.shape != ref.gate_up_proj.shape or down.shape != ref.down_proj.shape:
            raise ValueError("Expert shape mismatch")
        self.gate = gate
        self.down = down
        self.act = ref.act_fn

    def forward(self, x, indices, weights):
        result = torch.zeros_like(x)
        # Sort in expert order and obtain dispatch offsets with one host sync.
        order = indices.flatten().argsort(stable=True)
        counts = indices.flatten().bincount(minlength=len(self.gate)).tolist()
        start = 0
        for expert, count in enumerate(counts):
            if not count:
                continue
            flat = order[start : start + count]
            start += count
            tokens = flat // indices.size(1)
            gates, up = nn.functional.linear(x[tokens], self.gate[expert].to(x.device)).chunk(2, -1)
            out = nn.functional.linear(self.act(gates) * up, self.down[expert].to(x.device))
            result.index_add_(0, tokens, (out * weights.flatten()[flat, None]).to(x.dtype))
        return result


def load_weights(snapshot):
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    resident_layers = max(0, min(48, int((free_gib - 40) / 4.7)))
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config)
    store = Store(snapshot)

    def source(key):
        return key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key

    for path, param in list(model.named_parameters()):
        parent, field = path.rsplit(".", 1)
        mod = model.get_submodule(parent)
        key = source(path)
        if ".ngram_embedding." in path:
            prefix = key.removesuffix(".weight") + ".shard_"
            names = sorted(
                (k for k in store.index if k.startswith(prefix)),
                key=lambda k: int(k[len(prefix) :].split(".")[0]),
            )
            model.set_submodule(parent, ShardedEmbedding(mod, [store[n] for n in names]))
        elif ".mlp.experts." in path and int(path.split(".")[2]) >= resident_layers:
            if field == "gate_up_proj":
                model.set_submodule(
                    parent, StreamExperts(mod, store[key], store[source(parent) + ".down_proj"])
                )
        else:
            data = store[key]
            if data.shape != param.shape:
                raise ValueError((key, data.shape, param.shape))
            setattr(mod, field, nn.Parameter(data.cuda(), requires_grad=False))
    for path, buf in list(model.named_buffers()):
        parent, field = path.rsplit(".", 1)
        key = source(path)
        data = store[key] if key in store.index else buf
        setattr(model.get_submodule(parent), field, data.cuda())
    return model.eval().requires_grad_(False)
