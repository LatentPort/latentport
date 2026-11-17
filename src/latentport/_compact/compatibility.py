"""Offline base-weight qualification for learned compaction."""
from functools import lru_cache
import json
from pathlib import Path

from ..errors import CapabilityError


@lru_cache(maxsize=1)
def catalog():
    return json.loads((Path(__file__).parent.parent / "configs" /
                       "compaction-compatibility.json").read_text(encoding="utf-8"))["models"]


def require_weights(identity, *, allow_unqualified=False):
    if type(allow_unqualified) is not bool:
        raise TypeError("allow_unqualified must be a bool")
    spec = catalog().get(identity.repo_id)
    if spec is None or identity.revision != spec["revision"]:
        raise CapabilityError("LPCompact requires the trained base revision")
    base = identity.weights.partition(":execution:")[0]
    if identity.backend in ("bf16", "nf4", "reference"):
        # Pair loaders verify every shard against their pinned bridge manifest.
        qualified = base in (identity.backend + ":" + spec["revision"],
                            identity.backend + "-sha256:" + spec["safetensors_fingerprint"])
    elif identity.backend == "gguf":
        qualified = base.startswith("gguf-sha256:") and base[12:] in spec["gguf_sha256"]
    else:
        qualified = False
    if not qualified and not allow_unqualified:
        raise CapabilityError(
            "LPCompact has no qualification receipt for these exact base weights. "
            "Use a qualified checkpoint or explicitly pass allow_unqualified=True "
            "for experimental compaction; model revision and state ABI checks still apply."
        )
