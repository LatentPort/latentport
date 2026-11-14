"""LPCompact: an immutable state-to-state attention-memory operation."""
from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from ..errors import CapabilityError
from ..identity import CompactionInfo
from .._positions import physical_length
from .policy import budget, choose, MIN_CONTEXT


def _device(values):
    return values["V.3"].device


def compact(state, *, ratio=8, force=False):
    # Validate before loading the selector or materializing a disk snapshot.
    budget(0, ratio)
    if type(force) is not bool:
        raise TypeError("force must be a bool")
    if state.identity.repo_id != "Qwen/Qwen3.8-27B" or state.identity.backend not in ("bf16", "gguf", "reference"):
        raise CapabilityError("LPCompact supports Qwen3.8-27B states")
    if state.identity.state_abi not in ("qwen3.5.complete-hybrid.v1",
                                        "llama.cpp.b19cbe9.qwen3.5.complete-hybrid.bf16.v2"):
        raise CapabilityError("LPCompact requires a supported complete hybrid-state layout")
    from .._reference.native import validate_components
    import torch
    values = state._read_tensors()
    validate_components(values, state.clock, "target", native_precision=state.identity.backend == "gguf")
    physical = physical_length(values, state.clock)
    target = budget(physical, ratio)
    if (not force and physical < MIN_CONTEXT) or target == physical:
        return state
    device = _device(values)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    sync()
    started = perf_counter()
    from .weights import score
    with torch.inference_mode():
        indices = torch.tensor(choose(score(values), target), device=device, dtype=torch.long)
        # Unchanged tensors share immutable storage, just like State.fork().
        # All backend writes already materialize a private working copy.
        out = {name: value.index_select(2, indices.to(value.device)) if name.startswith(("K.", "V.")) else value
               for name, value in values.items()}
        positions = values.get("P", torch.arange(state.clock, device=device))
        out["P"] = positions.index_select(0, indices.to(positions.device))
    validate_components(out, state.clock, "target", native_precision=state.identity.backend == "gguf")
    sync()
    elapsed = perf_counter() - started
    kv_before = sum(v.numel() * v.element_size() for k, v in values.items() if k.startswith(("K.", "V.")))
    kv_after = sum(v.numel() * v.element_size() for k, v in out.items() if k.startswith(("K.", "V.")))
    info = CompactionInfo("LPCompact", "0.1", ratio, physical, target, state.clock,
                          kv_before, kv_after, elapsed, max(0, target - 1024), target == 1024)
    return replace(state, _components=out, _snapshot=None, _disk_index=None,
                   state_id=uuid4().hex, parent_id=state.state_id, shares_storage=True,
                   origin="compaction", compaction=info)
