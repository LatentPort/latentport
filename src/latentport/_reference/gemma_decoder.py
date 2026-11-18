"""Pinned Gemma 4 text decoder with layer-major BF16 execution on one L40S."""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4 import modeling_gemma4 as gemma

from .rotary import configure_global_yarn


class Decoder:
    options = None

    def __init__(self, snapshot, device="cuda", *, options=None):
        self.options = options
        self.device = torch.device(device)
        path = Path(snapshot)
        self.cfg = AutoConfig.from_pretrained(path, local_files_only=True).text_config
        self.config = self.cfg
        self.cfg._attn_implementation = "sdpa"
        with torch.device("meta"):
            self.text = gemma.Gemma4TextModel(self.cfg).eval().requires_grad_(False)
        weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        self.host = {}
        # Reserve room for the 128K full cache, branch workspaces, RoPE and activations.
        keep = max(0, int((torch.cuda.mem_get_info(self.device)[0] / 2**30 - 37) / 1.05))
        for name, param in list(self.text.named_parameters()) + list(self.text.named_buffers()):
            key = "model.language_model." + name
            if key not in weight_map:
                if name.startswith("rotary_emb.") or name == "embed_tokens.embed_scale":
                    continue
                raise KeyError(key)
            parent, field = name.rsplit(".", 1)
            with safe_open(str(path / weight_map[key]), framework="pt", device="cpu") as shard:
                value = shard.get_tensor(key)
            if value.shape != param.shape:
                raise ValueError(name)
            if name.startswith("layers.") and int(name.split(".")[1]) >= keep:
                idx = int(name.split(".")[1])
                self.host.setdefault(idx, {})[name.split(".", 2)[2]] = value
            else:
                value = value.to(self.device)
            module = self.text.get_submodule(parent)
            if field in module._parameters:
                setattr(module, field, torch.nn.Parameter(value, requires_grad=False))
            else:
                setattr(module, field, value)
        self.text.rotary_emb = gemma.Gemma4TextRotaryEmbedding(self.cfg).to(self.device)
        configure_global_yarn(self.text.rotary_emb, options, layer_type="full_attention")
        emb = self.text.embed_tokens
        emb.embed_scale = torch.tensor(emb.scalar_embed_scale, device=self.device, dtype=torch.bfloat16)

    @contextmanager
    def layer(self, idx):
        module = self.text.layers[idx]
        host = self.host.get(idx)
        if host:
            module.load_state_dict(
                {k: v.to(self.device) for k, v in host.items()}, assign=True, strict=True
            )
        try:
            yield module
        finally:
            if host:
                module.load_state_dict(host, assign=True, strict=True)

    @torch.inference_mode()
    def step(self, ids, cache, clock, *, keep_last=1):
        ids = torch.as_tensor(ids, device=self.device).reshape(1, -1)
        pos = torch.arange(clock, clock + ids.numel(), device=self.device)[None]
        hidden = self.text.embed_tokens(ids)
        rotations = {kind: self.text.rotary_emb(hidden, pos, kind) for kind in set(self.cfg.layer_types)}
        for i, kind in enumerate(self.cfg.layer_types):
            with self.layer(i) as layer:
                slot = cache.layers[i]
                for start in range(0, ids.numel(), 512):
                    end = min(ids.numel(), start + 512)
                    count = end - start
                    prior = slot.keys.shape[-2] if slot.is_initialized else 0
                    kv = torch.arange(prior + count, device=self.device)[None, :]
                    q = (prior + torch.arange(count, device=self.device))[:, None]
                    allowed = kv <= q
                    if kind == "sliding_attention":
                        allowed &= kv > q - self.cfg.sliding_window
                    rope = tuple(v[:, start:end] for v in rotations[kind])
                    hidden[:, start:end] = layer(
                        hidden[:, start:end],
                        position_embeddings=rope,
                        attention_mask=allowed[None, None],
                        position_ids=pos[:, start:end],
                        shared_kv_states={},
                        past_key_values=cache,
                    )
        return self.text.norm(hidden[:, -keep_last:])

    def logits(self, hidden):
        scores = torch.nn.functional.linear(hidden, self.text.embed_tokens.weight)
        cap = self.cfg.final_logit_softcapping
        return (scores / cap).tanh() * cap if cap else scores
