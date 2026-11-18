"""Gemma's global cache can shrink while its sliding attention keeps a dense tail."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.gemma4")
from latentport._reference.gemma import GemmaBackend
from latentport._reference.gemma_decoder import Decoder
from latentport._reference.native import validate_components
from transformers import DynamicCache, Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel


@torch.inference_mode()
def test_layer_major_matches_transformers_and_restores_sliding_window():
    cfg = Gemma4TextConfig(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size=32,
        intermediate_size=64,
        hidden_size_per_layer_input=0,
        head_dim=16,
        global_head_dim=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        sliding_window=8,
        attention_k_eq_v=True,
        vocab_size=64,
    )
    cfg._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        text = Gemma4TextModel(cfg).eval().to(torch.bfloat16)
    d = Decoder.__new__(Decoder)
    d.cfg, d.text, d.device, d.host = cfg, text, torch.device("cpu"), {}
    tokens = list(range(2, 21))
    full = DynamicCache(config=cfg)
    reference = text(torch.tensor([tokens]), past_key_values=full, use_cache=True)
    actual = DynamicCache(config=cfg)
    d.step(tokens[:12], actual, 0)
    hidden = d.step(tokens[12:], actual, 12)
    torch.testing.assert_close(hidden, reference.last_hidden_state[:, -1:], atol=0.04, rtol=0.04)
    backend = GemmaBackend.__new__(GemmaBackend)
    values = backend._extract(d, actual, hidden)
    geo = {
        "types": cfg.layer_types,
        "hidden": 32,
        "kv_heads": 1,
        "head_dim": 16,
        "sliding_heads": 1,
        "sliding_dim": 16,
        "sliding_window": 8,
    }
    validate_components(values, len(tokens), "target", geometry=geo)
    keep = torch.tensor([0, 1, 14, 15, 16, 17, 18])
    sparse = {
        **values,
        "P": keep,
        "K.1": values["K.1"].index_select(2, keep),
        "V.1": values["V.1"].index_select(2, keep),
    }
    validate_components(sparse, len(tokens), "target", geometry=geo)
    restored = backend._cache(d, sparse, len(tokens))
    assert restored.get_seq_length(0) == len(tokens)
    assert restored.get_seq_length(1) == keep.numel()
    d.step([21], restored, len(tokens))
    assert restored.layers[0].keys.shape[2] == 7
    assert restored.layers[1].keys.shape[2] == keep.numel() + 1

    # Compare chunked and single-token execution where native positions have ended.
    from latentport.context import ContextOptions
    d.options = ContextOptions("yarn", 8192, 1048576, cfg.max_position_embeddings)
    from latentport._reference.rotary import configure_global_yarn
    configure_global_yarn(d.text.rotary_emb, d.options, layer_type="full_attention")
    batched = DynamicCache(config=cfg)
    split = DynamicCache(config=cfg)
    expected = d.step([21, 22], batched, 999999)
    d.step([21], split, 999999)
    actual = d.step([22], split, 1000000)
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)


def test_state_compact_gathers_only_global_positions(monkeypatch):
    from latentport import ModelIdentity, State, _compact
    from latentport._compact import weights
    from latentport._models import GEMMA, GEMMA_ABI, MODEL_REVISIONS

    geo = {
        "types": ["sliding_attention", "full_attention"],
        "hidden": 2,
        "kv_heads": 1,
        "head_dim": 2,
        "sliding_heads": 1,
        "sliding_dim": 2,
        "sliding_window": 8,
    }
    n = 2048
    dense = torch.arange(n * 2, dtype=torch.bfloat16).reshape(1, 1, n, 2)
    tail = torch.arange(14, dtype=torch.bfloat16).reshape(1, 1, 7, 2)
    values = {
        "H": torch.ones(1, 1, 2, dtype=torch.bfloat16),
        "K.1": dense,
        "V.1": dense,
        "SK.0": tail,
        "SV.0": tail,
    }
    revision = MODEL_REVISIONS[GEMMA]
    identity = ModelIdentity(GEMMA, revision, "bf16", "bf16:" + revision, revision, "template", GEMMA_ABI)
    parent = State(identity, n, "context_ready", "test", _components=values)
    monkeypatch.setattr(_compact, "model_geometry", lambda repo: geo)
    monkeypatch.setattr(weights, "score", lambda values, repo: list(range(n)))
    child = parent.compact(force=True)
    assert child.physical_kv_length == 1024 and parent.physical_kv_length == n
    assert child._read_tensors()["SK.0"].data_ptr() == tail.data_ptr()
    assert child.compaction.kv_bytes_after == 2 * (1024 + 7) * 2 * 2
    assert child.clock == parent.clock


def test_standalone_shard_fingerprint_is_qualified(tmp_path, monkeypatch):
    import hashlib

    from latentport import ModelIdentity
    from latentport._compact import compatibility
    from latentport.errors import CapabilityError
    from latentport.single import _weight_fingerprint

    name = "model.safetensors"
    path = tmp_path / name
    path.write_bytes(b"frozen Gemma checkpoint fixture")
    shard = hashlib.sha256(path.read_bytes()).hexdigest()
    catalog = {
        "test/model": {
            "revision": "pin",
            "weight_files": {name: {"sha256": shard}},
            "safetensors_fingerprint": hashlib.sha256(name.encode() + b"\0" + shard.encode()).hexdigest(),
        }
    }
    monkeypatch.setattr(compatibility, "catalog", lambda: catalog)

    def identity():
        return ModelIdentity(
            "test/model",
            "pin",
            "bf16",
            "bf16-sha256:" + _weight_fingerprint(tmp_path),
            "pin",
            "template",
            "abi",
        )

    compatibility.require_weights(identity())
    path.write_bytes(b"altered checkpoint")
    with pytest.raises(CapabilityError):
        compatibility.require_weights(identity())
