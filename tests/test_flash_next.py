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

def test_standard_model_registry_and_state_compaction(monkeypatch):
    from latentport import ModelIdentity, State
    from latentport._models import FLASH_ABI, FLASH_NEXT, MODEL_REVISIONS

    n = 8195
    geom = dict(types=["full_attention", "linear_attention"], hidden=4, kv_heads=1,
                head_dim=2, recurrent_heads=1, recurrent_k=2, recurrent_v=2,
                conv_channels=3, conv_width=4, index_width=1, ple_channels=2, ple_width=3, ngram_width=2)
    values = {"H": torch.zeros(1,1,4,dtype=torch.bfloat16),
              "K.0": torch.zeros(1,1,n,2,dtype=torch.bfloat16),
              "V.0": torch.zeros(1,1,n,2,dtype=torch.bfloat16),
              "I.0": (torch.arange(n)%127).to(torch.bfloat16).view(1,n,1),
              "C.1": torch.zeros(1,3,4,dtype=torch.bfloat16), "R.1": torch.zeros(1,1,2,2),
              "PLE": torch.ones(1,2,3,dtype=torch.bfloat16),"NGRAM": torch.tensor([[4,5]])}
    from latentport import _compact
    monkeypatch.setattr(_compact, "model_geometry", lambda repo: geom)
    monkeypatch.setattr(_compact, "_device", lambda data: data["I.0"].device)
    monkeypatch.setattr(weights, "score", lambda data, repo: list(range(n)))
    ident = ModelIdentity(FLASH_NEXT, MODEL_REVISIONS[FLASH_NEXT], "bf16", "bf16:" + MODEL_REVISIONS[FLASH_NEXT], "tok", "template", FLASH_ABI)
    parent = State(ident,n,"context_ready","test",_components=values)
    assert parent.compact() is parent
    child = parent.compact(ratio=4,force=True)
    kept = child._read_tensors()
    assert child.clock==n and child.physical_kv_length==2047
    assert torch.equal(kept["I.0"], values["I.0"].index_select(1,kept["P"]))
    assert kept["NGRAM"] is values["NGRAM"] and kept["PLE"] is values["PLE"]
    assert (kept["P"][:-3].reshape(-1,4).diff()==1).all()
