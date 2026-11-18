"""LPCombine joins two same-model contexts at native token boundaries."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

from ._models import DEFAULT_MODEL, FLASH_NEXT, GEMMA, QWEN3, MODEL_REVISIONS
from .errors import CapabilityError, CompatibilityError, StateError
from .identity import CombinationInfo

PROFILE_REPOS = {
    DEFAULT_MODEL: "latentport/LPCombine-Qwen3.8-27b",
    "Qwen/Qwen3.5-4B": "latentport/LPCombine-Qwen3.5-4B",
    "Qwen/Qwen3.5-9B": "latentport/LPCombine-Qwen3.5-9B",
    GEMMA: "latentport/LPCombine-Gemma-4-31B-it",
    QWEN3: "latentport/LPCombine-Qwen3-14B",
    FLASH_NEXT: "latentport/LPCombine-Qwen3.8-Flash-Next",
}
LEGACY_CACHE_BACKENDS = {
    DEFAULT_MODEL: ("nf4",),
    "Qwen/Qwen3.5-4B": ("bf16", "nf4"),
    "Qwen/Qwen3.5-9B": ("bf16", "nf4"),
    GEMMA: ("bf16",),
    QWEN3: ("bf16",),
    FLASH_NEXT: ("bf16",),
}
ALIASES = {alias.casefold(): base for base, repo in PROFILE_REPOS.items()
           for alias in (repo, repo.rsplit("/", 1)[1])}


@lru_cache(maxsize=len(PROFILE_REPOS))
def load_profile(base):
    """Resolve a Hub profile manifest while keeping packaged pins available offline."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled

    repo = PROFILE_REPOS[base]
    try:
        path = Path(hf_hub_download(repo, "config.json", token=False, etag_timeout=2))
    except (HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled):
        path = Path(__file__).parent / "configs" / (repo.rsplit("/", 1)[1].lower() + ".json")
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("repo_id"), config.get("base_model"), config.get("base_revision")) != (
            repo, base, MODEL_REVISIONS[base]):
        raise CompatibilityError("LPCombine profile does not match the runtime's pinned checkpoint")
    return config["base_model"], config["base_revision"]


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
    if method in ("auto", "tokens"):
        ids = right.rendered_token_ids
        complete = ids is not None and len(ids) == right.clock
    if method == "auto":
        method = "tokens" if complete else "cache"
    if method == "tokens":
        if not complete:
            raise StateError("The right state has no complete token history. Capture with prefill() or use live(retain_replay=True).")
    else:
        activations = right._input_embeddings()
        if activations is not None and model._embedding_cache is not None:
            ids = model._embedding_cache.recover(activations, right.clock)
        else:
            ids = _recover_legacy(model, right, base, backend)
        if len(ids) != right.clock:
            raise StateError("Recovered token count differs from the right state clock")
    child = model.append(left, ids, phase="context_ready")
    return replace(child, origin="combination", messages=(), messages_complete=False,
                   generated_token_ids=(),
                   combination=CombinationInfo(method, left.state_id, right.state_id, len(ids)))


def _recover_legacy(model, right, base, backend):
    if backend not in LEGACY_CACHE_BACKENDS[base]:
        raise CapabilityError("This older snapshot requires retained token IDs. Capture a new right state for cache-only combine.")
    if base in (GEMMA, QWEN3):
        from ._combine_values import recover_tokens
    elif base == FLASH_NEXT:
        from ._combine_flash import recover_tokens
    else:
        from ._combine_readout import recover_tokens
    return recover_tokens(model, right)
