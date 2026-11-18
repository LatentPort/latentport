"""LPCombine joins two same-model contexts at native token boundaries."""
from dataclasses import replace

from ._models import DEFAULT_MODEL
from .errors import CapabilityError, StateError
from .identity import CombinationInfo

PROFILE_REPO = "latentport/LPCombine-Qwen3.8-27b"


def combine(model, left, right, *, method="auto"):
    model._check_open()
    left.require_model(model.identity)
    right.require_model(model.identity)
    if model.identity.repo_id != DEFAULT_MODEL or model.identity.backend not in ("bf16", "nf4", "gguf"):
        raise CapabilityError("LPCombine requires the standalone Qwen3.8-27B runtime")
    if left.phase != "context_ready" or right.phase != "context_ready":
        raise StateError("Combine requires two context-ready states")
    if method not in ("auto", "tokens", "cache"):
        raise ValueError("method must be 'auto', 'tokens', or 'cache'")
    check = getattr(model.backend, "check_capacity", None)
    if check is not None:
        check(left.clock, left.physical_kv_length, right.clock)
    ids = right.rendered_token_ids
    complete = ids is not None and len(ids) == right.clock
    if method == "auto":
        method = "tokens" if complete else "cache"
    if method == "tokens":
        if not complete:
            raise StateError("The right state has no complete token history. Capture with prefill() or use live(retain_replay=True).")
    else:
        if model.identity.backend != "nf4":
            raise CapabilityError("Cache-only combine requires NF4. BF16 and GGUF combine use retained token IDs.")
        from ._combine_readout import recover_tokens
        ids = recover_tokens(model, right)
        if len(ids) != right.clock:
            raise StateError("Recovered token count differs from the right state clock")
    child = model.append(left, ids, phase="context_ready")
    return replace(child, origin="combination", messages=(), messages_complete=False,
                   generated_token_ids=(),
                   combination=CombinationInfo(method, left.state_id, right.state_id, len(ids)))
