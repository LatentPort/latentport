"""LPCompact: an immutable state-to-state attention-memory operation."""
from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from ..errors import CapabilityError
from .._models import MODEL_REVISIONS, FLASH_NEXT, FLASH_ABI, GEMMA, GEMMA_ABI, QWEN3, QWEN3_ABI, model_geometry
from ..identity import CompactionInfo
from .._positions import physical_length
from .policy import budget, choose, MIN_CONTEXT


def _device(values):
    return next(value.device for name, value in values.items() if name.startswith("V."))


def compact(state, *, ratio=8, force=False, allow_unqualified=False):
    # Validate before loading the selector or materializing a disk snapshot.
    budget(0, ratio)
    if type(force) is not bool:
        raise TypeError("force must be a bool")
    repo = state.identity.repo_id
    if repo not in MODEL_REVISIONS or state.identity.backend not in ("bf16", "gguf", "nf4", "reference"):
        raise CapabilityError("LPCompact requires a supported model and state backend")
    if state.identity.revision != MODEL_REVISIONS[repo]:
        raise CapabilityError("LPCompact requires the trained base revision")
    from .compatibility import require_weights
    require_weights(state.identity, allow_unqualified=allow_unqualified)
    is_flash = repo == FLASH_NEXT
    if is_flash and (state.identity.state_abi != FLASH_ABI or state.identity.backend != "bf16"):
        raise CapabilityError("Flash-Next requires its complete BF16 indexed state")
    if repo == QWEN3 and (state.identity.state_abi != QWEN3_ABI or state.identity.backend != "bf16"):
        raise CapabilityError("Qwen3 requires its complete BF16 attention state")
    if repo == GEMMA and (state.identity.state_abi != GEMMA_ABI or state.identity.backend != "bf16"):
        raise CapabilityError("Gemma 4 requires its complete BF16 sliding and global attention state")
    if repo not in (QWEN3, GEMMA, FLASH_NEXT) and state.identity.state_abi not in ("qwen3.5.complete-hybrid.v1",
                                        "llama.cpp.b19cbe9.qwen3.5.complete-hybrid.bf16.v2"):
        raise CapabilityError("LPCompact requires a supported complete hybrid-state layout")
    from .._reference.native import validate_components
    import torch
    values = state._read_tensors()
    geometry = model_geometry(repo)
    validate_components(values, state.clock, "target", geometry=geometry, native_precision=state.identity.backend == "gguf")
    physical = physical_length(values, state.clock)
    target = budget(physical, ratio)
    if (not force and physical < geometry.get("minimum_context", MIN_CONTEXT)) or target == physical:
        return state
    device = _device(values)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    sync()
    started = perf_counter()
    from .weights import score
    with torch.inference_mode():
        scores = score(values, repo)
        if is_flash:
            from ..flash_next import _blocks
            indices = _blocks(torch.tensor(scores, device=device), max(target, 1024 + physical % 4))
            target = indices.numel()
        else:
            indices = torch.tensor(choose(scores, target), device=device, dtype=torch.long)
        # Unchanged tensors share immutable storage, just like State.fork().
        # All backend writes already materialize a private working copy.
        out = {name: value.index_select(2, indices.to(value.device)) if name.startswith(("K.", "V.")) else value
               for name, value in values.items()}
        if is_flash:
            for name, value in values.items():
                if name.startswith("I."):
                    out[name] = value.index_select(1, indices.to(value.device))
        positions = values["P"] if "P" in values else torch.arange(state.clock, device=device)
        out["P"] = positions.index_select(0, indices.to(positions.device))
    validate_components(out, state.clock, "target", geometry=geometry, native_precision=state.identity.backend == "gguf")
    sync()
    elapsed = perf_counter() - started
    kv_before = sum(v.numel() * v.element_size() for k, v in values.items() if k.startswith(("K.", "V.", "SK.", "SV.")))
    kv_after = sum(v.numel() * v.element_size() for k, v in out.items() if k.startswith(("K.", "V.", "SK.", "SV.")))
    protected = 1024 + physical % 4 if is_flash else 1024
    info = CompactionInfo("LPCompact", "0.1", ratio, physical, target, state.clock,
                          kv_before, kv_after, elapsed, max(0, target - protected), target <= protected)
    return replace(state, _components=out, _snapshot=None, _disk_index=None,
                   state_id=uuid4().hex, parent_id=state.state_id, shares_storage=True,
                   origin="compaction", compaction=info)
