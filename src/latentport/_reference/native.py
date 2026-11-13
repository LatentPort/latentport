"""Lazy Transformers/NF4 backend for the selected Qwen 4B and 27B pair."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from threading import RLock

from ..errors import CapabilityError, CompatibilityError, StateError
from . import cache_abi
from .._positions import physical_length, continuation_inputs, extend_positions


STATE_ABI = "qwen3.5.complete-hybrid.v1"
KERNEL_LOCK = RLock()


def _geometry(side: str):
    return json.loads((Path(__file__).resolve().parent.parent / "configs/geometry.json").read_text())[side]


def validate_components(values: Mapping[str, Any], clock: int, side: str, *, finite: bool = False, native_precision: bool = False) -> None:
    import torch
    geo = _geometry(side)
    expected = {"H"}
    physical = physical_length(values, clock)
    if "P" in values:
        expected.add("P")
    for index, kind in enumerate(geo["types"]):
        expected.update(f"{field}.{index}" for field in (("K", "V") if kind == "full_attention" else ("C", "R")))
    if set(values) != expected or clock < 1:
        raise StateError(f"{side} state needs exactly {len(expected)} complete fields and a positive clock")
    shapes = {"H": (1, 1, geo["hidden"]),
              "K": (1, geo["kv_heads"], physical, geo["head_dim"]),
              "V": (1, geo["kv_heads"], physical, geo["head_dim"]),
              "C": (1, geo["conv_channels"], geo["conv_width"]),
              "R": (1, geo["recurrent_heads"], geo["recurrent_k"], geo["recurrent_v"])}
    finite_checks = []
    for name, value in values.items():
        if name == "P":
            continue
        field = name.split(".", 1)[0]
        if tuple(value.shape) != shapes[field]:
            raise StateError(f"Wrong {name} shape: {tuple(value.shape)} != {shapes[field]}")
        allowed = {torch.float32 if field == "R" else torch.bfloat16}
        if native_precision and field in ("H", "C"):
            allowed.add(torch.float32)
        if value.dtype not in allowed:
            raise StateError(f"Wrong {name} dtype: {value.dtype}")
        if finite:
            finite_checks.append(torch.isfinite(value).all())
    if finite_checks and not bool(torch.stack(finite_checks).all()):
        raise StateError("Nonfinite state component")


def _original(module, name):
    stack, seen = [getattr(module, name)], set()
    while stack:
        fn = stack.pop()
        if not callable(fn) or id(fn) in seen:
            continue
        seen.add(id(fn))
        code = getattr(fn, "__code__", None)
        if code and code.co_name == name and str(code.co_filename).replace("\\", "/").endswith("modeling_qwen3_5.py"):
            return fn
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value):
                stack.append(value)
    raise CapabilityError(f"Pinned native Qwen kernel unavailable: {name}")


@contextmanager
def reference_kernels():
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
             "causal_conv1d_fn", "causal_conv1d_update")
    KERNEL_LOCK.acquire()
    old = {name: getattr(native, name) for name in names}
    try:
        for name in names:
            setattr(native, name, _original(native, name))
        yield
    finally:
        for name, fn in old.items():
            setattr(native, name, fn)
        KERNEL_LOCK.release()


class ReferenceBackend:
    name = "reference"
    state_abi = STATE_ABI
    supports_mtp = False
    translation_device = "cuda"
    kernels = staticmethod(reference_kernels)

    def __init__(self, locations: Mapping[str, str | Path | None], specs: Mapping[str, Mapping[str, Any]],
                 *, device: str = "cuda:0", resolver=None):
        self.locations = {side: Path(path) if path is not None else None for side, path in locations.items()}
        self.specs = specs
        if device != "cuda:0":
            raise CapabilityError("The selected translator is pinned to cuda:0")
        self.device = device
        self.translation_device = device
        self.resolver = resolver
        self._models = {}
        self._codecs = {}
        self._verified = set()

    def _path(self, side, *, metadata_only=False):
        path = self.locations[side]
        if path is None:
            if self.resolver is None:
                raise CapabilityError(f"No local {side} checkpoint or Hub resolver")
            path = Path(self.resolver(side, metadata_only=metadata_only))
        if side not in self._verified:
            for name, metadata in self.specs[side]["metadata_files"].items():
                file = path / name
                if not file.is_file():
                    raise CompatibilityError(f"Missing pinned {side} metadata: {name}")
                sha = hashlib.sha256(file.read_bytes()).hexdigest()
                if sha != metadata["sha256"]:
                    raise CompatibilityError(f"Pinned {side} metadata hash mismatch: {name}")
            self._verified.add(side)
        return path

    def _side(self, identity):
        for side, spec in self.specs.items():
            if (identity.repo_id, identity.revision) == (spec["repo_id"], spec["revision"]):
                return side
        raise CompatibilityError("Model is not a member of this bridge bundle")

    def codec(self, identity):
        side = self._side(identity)
        if side not in self._codecs:
            from transformers import AutoTokenizer
            from ..codec import Codec
            tokenizer = AutoTokenizer.from_pretrained(self._path(side, metadata_only=True), local_files_only=True)
            self._codecs[side] = Codec(tokenizer, self.specs[side].get("template_kwargs", {}))
        return self._codecs[side]

    def _model(self, identity):
        import torch
        side = self._side(identity)
        if side in self._models:
            return self._models[side]
        if not torch.cuda.is_available():
            raise CapabilityError("Reference execution requires a CUDA GPU")
        from transformers import AutoConfig, BitsAndBytesConfig, Qwen3_5ForCausalLM
        path = self._path(side)
        config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
        config.use_cache = True
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_quant_storage=torch.uint8,
                                  llm_int8_enable_fp32_cpu_offload=False)
        model, info = Qwen3_5ForCausalLM.from_pretrained(
            path, config=config, local_files_only=True, dtype=torch.bfloat16,
            quantization_config=quant, device_map={"": 0},
            attn_implementation="sdpa", low_cpu_mem_usage=True,
            key_mapping={r"^model\.language_model\.": "model."}, output_loading_info=True)
        if info["missing_keys"] or info["unexpected_keys"]:
            raise CompatibilityError(f"{side} checkpoint key mismatch")
        from .architecture import geometry
        if geometry(config) != _geometry(side):
            raise CompatibilityError(f"{side} model geometry changed")
        # The source checkpoint has no generation_config.json. The selected
        # bridge's native policy terminates at either end-of-text or chat EOS.
        model.generation_config.eos_token_id = [248044, 248046]
        model.eval().requires_grad_(False)
        self._models[side] = model
        return model

    def _extract(self, model, cache, terminal):
        result = {"H": terminal.detach()}
        for i, kind in enumerate(model.config.layer_types):
            a, b = cache_abi.extract(cache, i, kind)
            x, y = ("K", "V") if kind == "full_attention" else ("C", "R")
            result[f"{x}.{i}"], result[f"{y}.{i}"] = a.detach(), b.detach()
        return result

    def _cache(self, model, values, clock):
        cache = cache_abi.new_cache(model.config)
        for i, kind in enumerate(model.config.layer_types):
            x, y = ("K", "V") if kind == "full_attention" else ("C", "R")
            cache_abi.assign(cache, i, kind, values[f"{x}.{i}"], values[f"{y}.{i}"])
        if int(cache.get_seq_length()) != physical_length(values, clock):
            raise StateError("Installed cache clock differs from state clock")
        return cache

    def capture(self, identity, token_ids: Sequence[int]):
        import torch
        if not token_ids:
            raise ValueError("Cannot capture an empty stream")
        model = self._model(identity)
        terminal = []
        hook = model.model.norm.register_forward_pre_hook(lambda _, args: terminal.append(args[0][:, -1:].detach().clone()))
        try:
            with torch.inference_mode(), self.kernels():
                tensor = torch.tensor([list(token_ids)], device=self.device, dtype=torch.long)
                output = model(input_ids=tensor, use_cache=True, logits_to_keep=1)
        finally:
            hook.remove()
        if len(terminal) != 1:
            raise StateError("Ambiguous terminal H capture")
        values = self._extract(model, output.past_key_values, terminal[0])
        validate_components(values, len(token_ids), self._side(identity), finite=True)
        return values

    def advance(self, identity, components, clock: int, token_ids: Sequence[int]):
        import torch
        if not token_ids:
            raise ValueError("Cannot advance by zero tokens")
        side = self._side(identity)
        validate_components(components, clock, side)
        model = self._model(identity)
        cache = self._cache(model, components, clock)
        terminal = []
        hook = model.model.norm.register_forward_pre_hook(lambda _, args: terminal.append(args[0][:, -1:].detach().clone()))
        try:
            with torch.inference_mode(), self.kernels():
                tensor = torch.tensor([list(token_ids)], device=self.device, dtype=torch.long)
                extra = (continuation_inputs(components, clock, len(token_ids), device=self.device,
                                             dtype=components["H"].dtype) if "P" in components else {})
                output = model(input_ids=tensor, past_key_values=cache, use_cache=True, logits_to_keep=1, **extra)
        finally:
            hook.remove()
        if len(terminal) != 1:
            raise StateError("Ambiguous terminal H after append")
        values = self._extract(model, output.past_key_values, terminal[0])
        if "P" in components:
            values["P"] = extend_positions(components, clock, len(token_ids))
        validate_components(values, clock + len(token_ids), side, finite=True)
        return values

    def first_logits(self, identity, components, clock):
        import torch
        validate_components(components, clock, self._side(identity))
        model = self._model(identity)
        with torch.inference_mode():
            return model.lm_head(model.model.norm(components["H"]))[:, -1]

    def install(self, identity, components, clock):
        validate_components(components, clock, self._side(identity))
        self._cache(self._model(identity), components, clock)

    def eos_ids(self, identity):
        config = self._model(identity).generation_config.eos_token_id
        return frozenset(int(x) for x in (config if isinstance(config, (list, tuple)) else [config]))

    def peak_vram_bytes(self):
        import torch
        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    def begin_decode(self, identity, components, clock, *, mtp=False):
        if mtp:
            raise CapabilityError("Reference backend has no MTP draft runner")
        validate_components(components, clock, self._side(identity))
        model = self._model(identity)
        cache = self._cache(model, components, clock)
        return _ReferenceDecodeHandle(model, cache, components["H"], clock, self.device,
                                      positions=components.get("P"), kernels=self.kernels)


class _ReferenceDecodeHandle:
    def __init__(self, model, cache, terminal, clock, device, *, positions=None, kernels=reference_kernels):
        import torch
        self.model, self.cache, self.terminal = model, cache, terminal
        self.clock, self.device = clock, device
        self.positions, self.kernels = positions, kernels
        with torch.inference_mode():
            self.logits = model.lm_head(model.model.norm(terminal))[:, -1]

    def step(self, token_id):
        import torch
        terminal = []
        hook = self.model.model.norm.register_forward_pre_hook(
            lambda _, args: terminal.append(args[0][:, -1:].detach().clone()))
        try:
            with torch.inference_mode(), self.kernels():
                extra = (continuation_inputs({"P": self.positions}, self.clock, 1,
                         device=self.device, dtype=self.terminal.dtype) if self.positions is not None else {})
                result = self.model(input_ids=torch.tensor([[token_id]], device=self.device),
                                    past_key_values=self.cache, use_cache=True, logits_to_keep=1, **extra)
        finally:
            hook.remove()
        if len(terminal) != 1:
            raise StateError("Ambiguous terminal H during decoding")
        self.cache, self.terminal = result.past_key_values, terminal[0]
        if self.positions is not None:
            self.positions = extend_positions({"P": self.positions}, self.clock, 1)
        self.clock += 1
        physical = len(self.positions) if self.positions is not None else self.clock
        if int(self.cache.get_seq_length()) != physical:
            raise StateError("Decode cache clock drift")
        self.logits = result.logits[:, -1]

    def snapshot(self):
        result = {"H": self.terminal.detach()}
        if self.positions is not None:
            result["P"] = self.positions
        for i, kind in enumerate(self.model.config.layer_types):
            a, b = cache_abi.extract(self.cache, i, kind)
            x, y = ("K", "V") if kind == "full_attention" else ("C", "R")
            result[f"{x}.{i}"], result[f"{y}.{i}"] = a.detach(), b.detach()
        return result
