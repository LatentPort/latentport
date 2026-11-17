import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3")
from latentport._reference.native import validate_components
from latentport._reference.qwen3 import Qwen3Backend, Qwen3ContextOptions
from latentport._reference.qwen3_decoder import Decoder
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM


@torch.inference_mode()
def test_qwen3_chunked_decoder_matches_native_and_sparse_restore():
    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=64,
        max_position_embeddings=40960,
    )
    cfg._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    decoder = Decoder.__new__(Decoder)
    decoder.model, decoder.text, decoder.cfg = model, model.model, cfg
    decoder.device = torch.device("cpu")
    ids = list(range(2, 20))
    full = DynamicCache(config=cfg)
    expected = model(torch.tensor([ids]), past_key_values=full).logits[:, -1:]
    cache = DynamicCache(config=cfg)
    decoder.step(ids[:10], cache, 0)
    result = decoder.logits(decoder.step(ids[10:], cache, 10))
    torch.testing.assert_close(result, expected, atol=0.025, rtol=0.025)
    backend = Qwen3Backend.__new__(Qwen3Backend)
    values = backend._extract(decoder, cache, decoder.step([20], cache, 18))
    geo = dict(types=["full_attention"] * 2, hidden=32, kv_heads=1, head_dim=16)
    validate_components(values, 19, "target", geometry=geo)
    keep = torch.tensor([0, 5, 16, 17, 18])
    values = {
        key: value.index_select(2, keep) if key.startswith(("K.", "V.")) else value
        for key, value in values.items()
    }
    values["P"] = keep
    validate_components(values, 19, "target", geometry=geo)
    restored = backend._cache(decoder, values, 19)
    decoder.step([21], restored, 19)
    assert restored.get_seq_length() == 6
    assert Qwen3ContextOptions().logical_limit == 40960
    with pytest.raises(Exception, match="Logical position"):
        Qwen3ContextOptions().check(40960, 1024, 1)


def test_native_qwen3_compaction_activation_and_positions(monkeypatch):
    from latentport import ModelIdentity, State, _compact
    from latentport._compact import weights
    from latentport._models import MODEL_REVISIONS, QWEN3, QWEN3_ABI

    geo = dict(types=["full_attention"], hidden=2, kv_heads=1, head_dim=2, minimum_context=32768)
    revision = MODEL_REVISIONS[QWEN3]
    identity = ModelIdentity(QWEN3, revision, "bf16", "bf16:" + revision, revision, "template", QWEN3_ABI)
    monkeypatch.setattr(_compact, "model_geometry", lambda repo: geo)
    monkeypatch.setattr(weights, "score", lambda values, repo: list(range(values["V.0"].shape[2])))

    def state(n):
        values = {
            "H": torch.zeros(1, 1, 2, dtype=torch.bfloat16),
            "K.0": torch.zeros(1, 1, n, 2, dtype=torch.bfloat16),
            "V.0": torch.zeros(1, 1, n, 2, dtype=torch.bfloat16),
        }
        return State(identity, n, "context_ready", "test", _components=values)

    short = state(32767)
    assert short.compact() is short
    parent = state(32768)
    child = parent.compact()
    assert child.clock == 32768 and child.physical_kv_length == 4096
    assert parent.physical_kv_length == 32768
    assert torch.equal(child._read_tensors()["P"][:512], torch.arange(512))
    assert torch.equal(child._read_tensors()["P"][-512:], torch.arange(32256, 32768))
