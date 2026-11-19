"""BF16 and NF4 weights for the Qwen3.6 hybrid mixture-of-experts text model."""
import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

from .bf16 import BF16Backend, fast_kernels
from .flash_weights import Store
from ..context import configure_text
from .._models import model_geometry


class PackedExperts(nn.Module):
    """Keep both fused expert banks in NF4 and materialize one layer per call."""

    def __init__(self, experts, store, prefix, device):
        super().__init__()
        import bitsandbytes.functional as bnb

        self.experts = experts
        self.states = {}
        for name, parameter in experts.named_parameters():
            weight = store[f"{prefix}.{name}"]
            if weight.shape != parameter.shape:
                raise ValueError(f"Invalid expert shape for {prefix}.{name}")
            packed, state = bnb.quantize_4bit(weight.to(device), quant_type="nf4", compress_statistics=True)
            self.register_buffer("packed_" + name, packed)
            self.states[name] = state

    def forward(self, *args):
        from bitsandbytes.functional import dequantize_4bit

        weights = {name: dequantize_4bit(getattr(self, "packed_" + name), state)
                   for name, state in self.states.items()}
        return torch.func.functional_call(self.experts, weights, args, strict=True)


class Qwen36Backend(BF16Backend):
    def kernels(self):
        return fast_kernels(moe)

    def _model(self, identity):
        side = self._side(identity)
        if side in self._models:
            return self._models[side]
        path = self._path(side)
        cfg = configure_text(AutoConfig.from_pretrained(path, local_files_only=True).text_config,
                             self.options, model_geometry(identity.repo_id))
        cfg._attn_implementation = "sdpa"
        cfg._experts_implementation = "grouped_mm"
        with torch.device("meta"):
            model = moe.Qwen3_5MoeForCausalLM(cfg)
        model.model.rotary_emb = moe.Qwen3_5MoeTextRotaryEmbedding(cfg).to(self.device)
        store = Store(path)
        for name, parameter in list(model.named_parameters()):
            parent, field = name.rsplit(".", 1)
            key = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
            if self.name == "nf4" and parent.endswith(".mlp.experts"):
                if field == "gate_up_proj":
                    model.set_submodule(parent, PackedExperts(model.get_submodule(parent), store,
                                                               key.rsplit(".", 1)[0], self.device))
                continue
            module = model.get_submodule(parent)
            weight = store[key]
            if weight.shape != parameter.shape:
                raise ValueError(f"Invalid checkpoint shape for {key}")
            if self.name == "nf4" and isinstance(module, nn.Linear) and parent != "lm_head":
                from .nf4 import quantized_linear
                model.set_submodule(parent, quantized_linear(module, weight, self.device))
            else:
                setattr(module, field, nn.Parameter(weight.to(self.device), requires_grad=False))
        model.generation_config.eos_token_id = [248044, 248046]
        self._models[side] = model.eval().requires_grad_(False)
        return self._models[side]
