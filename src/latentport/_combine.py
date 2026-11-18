"""LPCombine joins two same-model contexts at native token boundaries."""
from dataclasses import replace

from ._models import DEFAULT_MODEL, FLASH_NEXT, GEMMA, QWEN3, MODEL_REVISIONS
from .errors import CapabilityError, StateError
from .identity import CombinationInfo

PROFILE_REPOS = {
    DEFAULT_MODEL: "latentport/LPCombine-Qwen3.8-27b",
    "Qwen/Qwen3.5-4B": "latentport/LPCombine-Qwen3.5-4B",
    "Qwen/Qwen3.5-9B": "latentport/LPCombine-Qwen3.5-9B",
    GEMMA: "latentport/LPCombine-Gemma-4-31B-it",
    QWEN3: "latentport/LPCombine-Qwen3-14B",
    FLASH_NEXT: "latentport/LPCombine-Qwen3.8-Flash-Next",
}
CACHE_BACKENDS = {
    DEFAULT_MODEL: ("nf4",),
    "Qwen/Qwen3.5-4B": ("bf16", "nf4"),
    "Qwen/Qwen3.5-9B": ("bf16", "nf4"),
    GEMMA: ("bf16",),
    QWEN3: ("bf16",),
    FLASH_NEXT: ("bf16",),
}
ALIASES = {alias.casefold(): base for base, repo in PROFILE_REPOS.items()
           for alias in (repo, repo.rsplit("/", 1)[1])}


def combine(model, left, right, *, method="auto"):
    model._check_open()
    left.require_model(model.identity)
    right.require_model(model.identity)
    base, backend = model.identity.repo_id, model.identity.backend
    if base not in PROFILE_REPOS or backend not in ("bf16", "nf4", "gguf"):
        raise CapabilityError("LPCombine requires a supported standalone model and backend")
    if model.identity.revision != MODEL_REVISIONS[base]:
        raise CapabilityError("LPCombine requires the profile's pinned base revision")
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
        if backend not in CACHE_BACKENDS[base]:
            raise CapabilityError("This LPCombine model/backend requires retained token IDs. Use prefill() or live(retain_replay=True).")
        if base in (GEMMA, QWEN3):
            from ._combine_values import recover_tokens
        elif base == FLASH_NEXT:
            from ._combine_flash import recover_tokens
        else:
            from ._combine_readout import recover_tokens
        ids = recover_tokens(model, right)
        if len(ids) != right.clock:
            raise StateError("Recovered token count differs from the right state clock")
    child = model.append(left, ids, phase="context_ready")
    return replace(child, origin="combination", messages=(), messages_complete=False,
                   generated_token_ids=(),
                   combination=CombinationInfo(method, left.state_id, right.state_id, len(ids)))
