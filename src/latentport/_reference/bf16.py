"""Single-model BF16 execution using the common complete-state interface."""
from contextlib import contextmanager

from .native import ReferenceBackend, validate_components
from ..errors import CapabilityError, CompatibilityError, StateError
from ..context import ContextOptions, configure_text
from .._models import model_geometry


@contextmanager
def fast_kernels(qwen=None):
    import inspect
    if qwen is None:
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

    def __init__(self, *args, options=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options or ContextOptions()

    def check_capacity(self, clock, physical, count=0):
        self.options.check(clock, physical, count)

    def _model(self, identity):
        import torch
        side = self._side(identity)
        if side in self._models:
            return self._models[side]
        if not torch.cuda.is_available():
            raise CapabilityError("BF16 execution requires CUDA")
        from transformers import AutoConfig, Qwen3_5ForCausalLM
        path = self._path(side)
        config = configure_text(AutoConfig.from_pretrained(path, local_files_only=True).text_config,
                                self.options, model_geometry(identity.repo_id))
        model, info = Qwen3_5ForCausalLM.from_pretrained(
            path, config=config, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True,
            key_mapping={r"^model\.language_model\.": "model."}, output_loading_info=True)
        # The text runtime excludes the vision branch and auxiliary MTP heads.
        unexpected = [key for key in info["unexpected_keys"]
                      if not key.startswith(("model.visual.", "mtp.", "model.mtp."))]
        if info["missing_keys"] or unexpected:
            raise CompatibilityError("BF16 text checkpoint tensor mismatch")
        if any(p.device.type != "cuda" or p.device.index != 0 for p in model.parameters()):
            raise CompatibilityError("BF16 weights must remain on cuda:0")
        model.generation_config.eos_token_id = [248044, 248046]
        self._models[side] = model.eval().requires_grad_(False)
        return model

    def _run(self, identity, values, clock, ids):
        from .._positions import physical_length
        from ..state import State
        from .live import LiveCache
        if not ids:
            raise StateError("append requires one or more tokens")
        physical = physical_length(values, clock) if values is not None else 0
        self.options.check(clock, physical, len(ids))
        parent = State(identity, clock, "context_ready", "standalone_install", _components=values) if values is not None else None
        with LiveCache(self, identity, parent, _owned=True) as live:
            for start in range(0, len(ids), 1024):
                live.append(ids[start:start + 1024])
            return live.snapshot(release=True)._read_tensors()

    def live(self, identity, state=None):
        from .live import LiveCache
        return LiveCache(self, identity, state)

    def begin_decode(self, identity, components, clock, *, mtp=False):
        from ..state import State
        from .live import LiveCache, LiveDecode
        if mtp:
            raise CapabilityError("Standalone Transformers execution disables MTP")
        parent = State(identity, clock, "answer_ready", "standalone_decode", _components=components)
        return LiveDecode(LiveCache(self, identity, parent, _owned=True))

    def capture(self, identity, token_ids):
        return self._run(identity, None, 0, list(token_ids))

    def advance(self, identity, components, clock, token_ids):
        validate_components(components, clock, "target", geometry=model_geometry(identity.repo_id))
        return self._run(identity, components, clock, list(token_ids))

    def close(self):
        self._models.clear()
