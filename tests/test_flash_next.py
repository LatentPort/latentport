import pytest
import torch

from latentport import CompatibilityError, compact_flash_next_cache
from latentport import flash_next
from latentport._compact import weights


def test_flash_next_cache_keeps_aligned_blocks_and_independent_state(monkeypatch):
    from transformers import DynamicCache, Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(num_hidden_layers=48, ple_layer_ids=[2], eos_token_id=248044)
    cache = DynamicCache(config=config)
    n = 8195
    positions = torch.arange(n)
    cache.position_ids = positions.view(1, 1, -1).expand(3, 1, -1)
    value = (positions % 251).to(torch.bfloat16).view(1, 1, n, 1).expand(1, 2, n, 256)
    for i, kind in enumerate(config.layer_types):
        if kind == "qwen_sparse_attention":
            cache.update(value, value, i)
            cache.update_indexer(value[:, 0, :, :128], i)
        else:
            cache.update_conv_state(torch.ones(1, 1, 4), i, conv_kernel_size=4)
            cache.update_recurrent_state(torch.ones(1, 1, 1, 1), i)
            if i == 1:
                for j in (1, 2):
                    cache.update_conv_state(torch.ones(1, 1, 2), i, state_idx=j, conv_kernel_size=2)
    # Exercise real compaction with a small footprint, including a three-token tail.
    monkeypatch.setattr(flash_next, "MIN_CONTEXT", 2048)
    monkeypatch.setattr(weights, "score", lambda values, repo: list(range(n)))
    child = compact_flash_next_cache(cache, ratio=4)
    keep = child.position_ids[0, 0]
    assert len(keep) <= n // 4
    assert torch.equal(keep[:512], positions[:512]) and torch.equal(keep[-512:], positions[-512:])
    assert (keep[:-3].reshape(-1, 4).diff() == 1).all()
    layer = child.layers[3]
    assert torch.equal(layer.keys[0, 0, :, 0], value[0, 0, keep, 0])
    assert torch.equal(layer.indexer_keys[0, :, 0], layer.keys[0, 0, :, 0])
    child.layers[1].conv_states[2].zero_()
    child.layers[0].recurrent_states[0].zero_()
    assert cache.layers[1].conv_states[2].all() and cache.layers[0].recurrent_states[0].all()
    cache.layers[1].conv_states[2] = None
    with pytest.raises(CompatibilityError, match="recurrent layer 1"):
        compact_flash_next_cache(cache)
