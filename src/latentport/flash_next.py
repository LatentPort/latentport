"""LPCompact for native Transformers Qwen3.8-Flash-Next caches."""
import copy

from ._compact.policy import MIN_CONTEXT, budget
from .errors import CompatibilityError

from ._models import FLASH_NEXT as BASE_MODEL
FEATURE_LAYERS = (3, 15, 31)


def _blocks(scores, target):
    import torch

    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("Expected one finite selector score per physical position")
    n = scores.numel()
    full, tail = divmod(n, 4)
    positions = torch.arange(n, device=scores.device)
    if n <= target:
        return positions
    blocks = torch.arange(full, device=scores.device)
    fixed = (blocks * 4 < 512) | ((blocks + 1) * 4 > n - 512)
    slots = (target - tail) // 4 - int(fixed.sum())
    if slots < 0:
        raise ValueError("Budget cannot hold the protected QSA blocks")
    candidates = blocks[~fixed]
    means = scores[:4 * full].reshape(full, 4).float().mean(dim=1)
    ranked = torch.argsort(means[candidates], descending=True, stable=True)
    keep = fixed.clone()
    keep[candidates[ranked[:slots]]] = True
    return torch.cat((positions[:4 * full][keep.repeat_interleave(4)], positions[4 * full:]))


def compact_flash_next_cache(cache, *, ratio=8):
    """Fork a batch-one native Flash-Next text cache with 4× or 8× compaction.

    Compatibility entry point for callers with an existing Transformers cache.
    For LatentPort states use State.compact().

    The pinned selector supports Qwen/Qwen3.8-Flash-Next at revision
    812d96e10a0dd82b69d749e68693698cf6c36daa. Continue the child with
    explicit logical position_ids starting after the input history.
    """
    import torch
    from transformers.cache_utils import DynamicCache, DynamicIndexedLayer, LinearAttentionLayer
    from ._compact.weights import score

    positions = getattr(cache, "position_ids", None)
    if (not isinstance(cache, DynamicCache) or len(cache.layers) != 48 or
            positions is None or positions.ndim != 3 or positions.shape[:2] != (3, 1)):
        raise CompatibilityError("Expected a batch-one Flash-Next DynamicCache with logical positions")
    n = positions.shape[-1]
    target = budget(n, ratio)
    for i, layer in enumerate(cache.layers):
        if i % 4 == 3:
            if (not isinstance(layer, DynamicIndexedLayer) or layer.keys is None or
                    layer.keys.shape != (1, 2, n, 256) or layer.values is None or
                    layer.values.shape != layer.keys.shape or layer.indexer_keys is None or
                    layer.indexer_keys.shape != (1, n, 128)):
                raise CompatibilityError(f"Incomplete Flash-Next indexed cache at layer {i}")
        elif (not isinstance(layer, LinearAttentionLayer) or layer.recurrent_states.get(0) is None or
              layer.conv_states.get(0) is None or
              (i == 1 and any(layer.conv_states.get(j) is None for j in (1, 2)))):
            raise CompatibilityError(f"Incompatible Flash-Next recurrent layer {i}")
    if n < MIN_CONTEXT:
        return copy.deepcopy(cache)
    device = cache.layers[FEATURE_LAYERS[0]].values.device
    with torch.no_grad():
        values = {f"V.{i}": cache.layers[i].values for i in FEATURE_LAYERS}
        scores = torch.tensor(score(values, BASE_MODEL), device=device, dtype=torch.float32)
        keep = _blocks(scores, target)
        child = copy.copy(cache)
        child.layers = []
        for layer in cache.layers:
            if isinstance(layer, DynamicIndexedLayer):
                new = copy.copy(layer)
                new.keys = layer.keys.index_select(2, keep.to(layer.keys.device))
                new.values = layer.values.index_select(2, keep.to(layer.values.device))
                new.indexer_keys = layer.indexer_keys.index_select(1, keep.to(layer.indexer_keys.device))
            else:
                new = copy.deepcopy(layer)
            child.layers.append(new)
        child.position_ids = positions.index_select(-1, keep.to(positions.device))
        return child
