"""Tensor-only LPCompact entry point for external inference engines."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._compact.policy import MIN_CONTEXT, budget, choose
from ._compact.weights import _model, score_features, selector_config
from ._models import FLASH_NEXT, MODEL_REVISIONS, feature_layers
from .errors import CompatibilityError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor


@dataclass(frozen=True)
class SelectorSpec:
    base_model: str
    base_revision: str
    artifact: str
    artifact_revision: str
    feature_layers: tuple[int, ...]
    minimum_rows: int
    granularity: int
    feature_width: int

    def target_rows(self, rows: int, ratio: int) -> int:
        target = budget(rows, ratio)
        if target < rows and self.granularity > 1:
            tail = rows % self.granularity
            target = max(1024 + tail, target)
            target -= (target - tail) % self.granularity
        return target


def get_selector_spec(
    base_model: str, base_revision: str, artifact: str
) -> SelectorSpec:
    """Resolve the checkpoint identity without importing PyTorch or downloading."""
    config = selector_config(base_model)
    if base_revision != MODEL_REVISIONS[base_model]:
        raise CompatibilityError("LPCompact requires the trained base revision")
    if artifact != config["repo_id"]:
        raise CompatibilityError(
            f"Expected selector {config['repo_id']} for {base_model}"
        )
    return SelectorSpec(
        base_model=base_model,
        base_revision=base_revision,
        artifact=artifact,
        artifact_revision=config["revision"],
        feature_layers=feature_layers(base_model),
        minimum_rows=config.get("minimum_physical_positions", MIN_CONTEXT),
        granularity=config.get("block_size", 4 if base_model == FLASH_NEXT else 1),
        feature_width=config["feature_dimension"],
    )


class KVSelector:
    """Score contiguous row ranges supplied by an engine-owned KV cache."""

    def __init__(self, spec: SelectorSpec, model):
        self.spec = spec
        self._model = model

    def select(
        self,
        read_values: "Callable[[int, int], Sequence[Tensor]]",
        rows: int,
        ratio: int,
    ) -> list[int]:
        """Return sorted physical row indices from value blocks shaped [rows, heads, dim]."""
        import torch

        target = self.spec.target_rows(rows, ratio)
        if target == rows:
            return list(range(rows))
        parts = []
        with torch.inference_mode():
            for start in range(0, rows, 8192):
                stop = min(rows, start + 8192)
                features = read_values(start, stop)
                if (
                    len(features) != len(self.spec.feature_layers)
                    or any(
                        value.ndim != 3 or value.shape[0] != stop - start
                        for value in features
                    )
                    or sum(value.shape[1] * value.shape[2] for value in features)
                    != self.spec.feature_width
                ):
                    raise CompatibilityError("Unexpected LPCompact feature layout")
                parts.append(score_features(self._model, features))
            scores = torch.cat(parts)
            if self.spec.granularity > 1:
                from .flash_next import _blocks

                return _blocks(scores, target).tolist()
            return choose(scores.cpu().tolist(), target)


def load_kv_selector(
    base_model: str,
    base_revision: str,
    artifact: str,
    *,
    device: str,
    weights_path: str | None = None,
) -> KVSelector:
    """Load the pinned selector for external engine KV, including checksum validation."""
    spec = get_selector_spec(base_model, base_revision, artifact)
    return KVSelector(spec, _model(str(device), spec.base_model, weights_path))
