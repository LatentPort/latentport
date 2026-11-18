"""Standalone NF4 weights with BF16 attention/conv and FP32 recurrent state."""
from .bf16 import BF16Backend
from ..context import configure_text
from .._models import MODEL_REVISIONS, model_geometry
from ..errors import CapabilityError, CompatibilityError

NF4_CONFIG = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
              "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16",
              "bnb_4bit_quant_storage": "uint8", "llm_int8_enable_fp32_cpu_offload": False}


def transformers_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(**{**NF4_CONFIG, "bnb_4bit_compute_dtype": torch.bfloat16,
                                 "bnb_4bit_quant_storage": torch.uint8})


def quantized_linear(module, weight, device):
    import bitsandbytes as bnb
    import torch

    with torch.device("meta"):
        linear = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None,
                                  compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4")
    packed = bnb.nn.Params4bit(weight, requires_grad=False, compress_statistics=True,
                              quant_type="nf4", module=linear)
    linear.weight = packed.to(device)
    return linear.eval()


class NF4Backend(BF16Backend):
    name = "nf4"

    def _model(self, identity):
        import torch
        side = self._side(identity)
        if side != "target" or identity.revision != MODEL_REVISIONS.get(identity.repo_id):
            raise CompatibilityError("Standalone NF4 requires the pinned base revision")
        if side in self._models:
            return self._models[side]
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise CapabilityError("Standalone NF4 requires exactly one visible CUDA GPU")
        from transformers import AutoConfig, Qwen3_5ForCausalLM
        path = self._path(side)
        config = configure_text(AutoConfig.from_pretrained(path, local_files_only=True).text_config,
                                self.options, model_geometry(identity.repo_id))
        model, info = Qwen3_5ForCausalLM.from_pretrained(
            path, config=config, local_files_only=True, dtype=torch.bfloat16,
            quantization_config=transformers_config(), device_map={"": 0}, attn_implementation="sdpa",
            low_cpu_mem_usage=True, key_mapping={r"^model\.language_model\.": "model."},
            output_loading_info=True)
        extra = [k for k in info["unexpected_keys"] if not k.startswith(("model.visual.", "mtp.", "model.mtp."))]
        if info["missing_keys"] or extra or info.get("mismatched_keys") or info.get("error_msgs"):
            raise CompatibilityError("NF4 text checkpoint tensor mismatch")
        tensors = [*model.parameters(), *model.buffers()]
        if any(p.device.type != "cuda" or p.device.index != 0 for p in tensors):
            raise CompatibilityError("NF4 parameters and buffers must remain on cuda:0")
        if any(str(d) not in ("0", "cuda", "cuda:0") for d in getattr(model, "hf_device_map", {}).values()):
            raise CompatibilityError("NF4 CPU/disk weight offload is forbidden")
        if not getattr(model, "is_loaded_in_4bit", False):
            raise CompatibilityError("NF4 quantized loading was not applied")
        model.generation_config.eos_token_id = [248044, 248046]
        self._models[side] = model.eval().requires_grad_(False)
        return model
