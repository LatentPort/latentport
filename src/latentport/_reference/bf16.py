"""Single-model BF16 execution using the common complete-state interface."""
from contextlib import contextmanager

from .native import ReferenceBackend, validate_components
from . import cache_abi
from .._positions import continuation_inputs, extend_positions
from ..errors import CapabilityError, CompatibilityError, StateError


@contextmanager
def fast_kernels():
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from .native import KERNEL_LOCK
    accepted = frozenset(inspect.signature(causal_conv1d_fn).parameters)
    def conv(*args, **kwargs):
        return causal_conv1d_fn(*args, **{k: v for k, v in kwargs.items() if k in accepted})
    replacements = {"torch_chunk_gated_delta_rule": chunk_gated_delta_rule,
                    "torch_recurrent_gated_delta_rule": fused_recurrent_gated_delta_rule,
                    "causal_conv1d_fn": conv, "causal_conv1d_update": causal_conv1d_update}
    with KERNEL_LOCK:
        old = {name: getattr(qwen, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(qwen, name, value)
            yield
        finally:
            for name, value in old.items():
                setattr(qwen, name, value)


class BF16Backend(ReferenceBackend):
    name = "bf16"
    kernels = staticmethod(fast_kernels)

    def _model(self, identity):
        import torch
        side = self._side(identity)
        if side in self._models:
            return self._models[side]
        if not torch.cuda.is_available():
            raise CapabilityError("BF16 execution requires CUDA")
        from transformers import AutoConfig, Qwen3_5ForCausalLM
        path = self._path(side)
        config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
        model, info = Qwen3_5ForCausalLM.from_pretrained(
            path, config=config, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True,
            key_mapping={r"^model\.language_model\.": "model."}, output_loading_info=True)
        # The text runtime excludes the vision branch and auxiliary MTP heads.
        unexpected = [key for key in info["unexpected_keys"]
                      if not key.startswith(("model.visual.", "mtp.", "model.mtp."))]
        if info["missing_keys"] or unexpected:
            raise CompatibilityError("BF16 text checkpoint tensor mismatch")
        from .architecture import geometry
        from .native import _geometry
        if geometry(config) != _geometry("target"):
            raise CompatibilityError("LPCompact model geometry mismatch")
        model.generation_config.eos_token_id = [248044, 248046]
        self._models[side] = model.eval().requires_grad_(False)
        return model

    def _run(self, identity, values, clock, ids):
        import torch
        if not ids or clock + len(ids) > 262144:
            raise StateError("Invalid continuation length or logical position limit")
        model = self._model(identity)
        cache = self._cache(model, values, clock) if values is not None else cache_abi.new_cache(model.config)
        terminal = []
        hook = model.model.norm.register_forward_pre_hook(lambda _, args: terminal.append(args[0][:, -1:].detach().clone()))
        try:
            with torch.inference_mode(), self.kernels():
                for offset in range(0, len(ids), 1024):
                    batch = ids[offset:offset + 1024]
                    # Explicit positions also protect chunked native prefill.
                    physical = int(cache.get_seq_length())
                    layout = {"P": torch.arange(physical, device=self.device)} if physical else {}
                    extra = continuation_inputs(layout, clock + offset, len(batch), device=self.device, dtype=torch.bfloat16)
                    model(input_ids=torch.tensor([batch], device=self.device), past_key_values=cache,
                          use_cache=True, logits_to_keep=1, **extra)
                    terminal[:] = terminal[-1:]
        finally:
            hook.remove()
        out = self._extract(model, cache, terminal[-1])
        if values is not None and "P" in values:
            out["P"] = extend_positions(values, clock, len(ids))
        validate_components(out, clock + len(ids), "target", finite=True)
        return out

    def capture(self, identity, token_ids):
        return self._run(identity, None, 0, list(token_ids))

    def advance(self, identity, components, clock, token_ids):
        validate_components(components, clock, "target")
        return self._run(identity, components, clock, list(token_ids))

    def close(self):
        self._models.clear()
