"""CPU invariants for compaction's integration with ordinary immutable States."""
from dataclasses import replace
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from latentport import ModelIdentity, State
from latentport.errors import CapabilityError, StateError
from latentport._compact.policy import budget, choose
from latentport._positions import continuation_inputs, physical_length, extend_positions
from latentport._reference.native import validate_components

GEO = {"types": ["linear_attention", "full_attention"], "hidden": 8,
       "kv_heads": 2, "head_dim": 4, "conv_channels": 3, "conv_width": 4,
       "recurrent_heads": 2, "recurrent_k": 3, "recurrent_v": 3}
IDENTITY = ModelIdentity("Qwen/Qwen3.8-27B", "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
                          "bf16", "bf16", "pinned", "template", "qwen3.5.complete-hybrid.v1")


@pytest.fixture
def geometry():
    with patch("latentport._reference.native._geometry", return_value=GEO):
        yield


def state(n=65536):
    # Small feature width exercises real tensor gather, ownership and serialization.
    values = {"H": torch.ones(1, 1, 8, dtype=torch.bfloat16),
              "K.1": torch.arange(n, dtype=torch.float32).to(torch.bfloat16)[None,None,:,None].expand(1,2,n,4).clone(),
              "V.1": torch.zeros(1,2,n,4,dtype=torch.bfloat16),
              "C.0": torch.ones(1,3,4,dtype=torch.bfloat16), "R.0": torch.ones(1,2,3,3)}
    return State(IDENTITY, n, "answer_ready", "native_capture", rendered_token_ids=tuple(range(n)),
                 messages=(("user", "preserve me"),), _components=values)


def compact(s, **kw):
    n = physical_length(s._read_tensors(), s.clock)
    with patch("latentport._compact.weights.score", return_value=[-1.] * n), \
         patch("latentport._compact._device", return_value=torch.device("cpu")):
        return s.compact(**kw)


def test_compact_is_a_state_and_preserves_parent_forks_and_full_hybrid_state(geometry):
    parent = state()
    fork = parent.fork()
    child = compact(parent)
    assert type(child) is State and child.clock == parent.clock
    assert child.identity == parent.identity and child.phase == parent.phase
    assert child.rendered_token_ids is parent.rendered_token_ids
    assert child.messages == parent.messages and child.parent_id == parent.state_id
    assert child.state_id != parent.state_id
    a, b = parent._read_tensors(), child._read_tensors()
    assert b["P"].shape == (8192,) and b["K.1"].shape[2] == 8192
    assert a["K.1"].shape[2] == 65536 and "P" not in fork._read_tensors()
    for field in ("H", "C.0", "R.0"):
        assert b[field] is a[field] and torch.equal(a[field], b[field])
    assert torch.equal(b["K.1"], a["K.1"].index_select(2, b["P"]))
    assert child.compaction.kv_bytes_before == 8 * child.compaction.kv_bytes_after
    private = child._tensors()
    private["R.0"].zero_()
    assert parent._read_tensors()["R.0"].sum() > 0


def test_short_context_is_identical_noop_without_loading_weights(geometry):
    parent = state(8192)
    with patch("latentport._compact.weights.score", side_effect=AssertionError("loaded selector")):
        assert parent.compact() is parent


def test_forced_floor_is_recorded_and_retains_prefix_suffix(geometry):
    child = compact(state(8192), force=True)
    assert child.compaction.floor_bound and child.compaction.learned_positions == 0
    assert child._read_tensors()["P"].tolist() == list(range(512)) + list(range(7680,8192))


def test_snapshot_roundtrip_and_repeat_compaction_keep_absolute_positions(geometry):
    child = compact(state())
    with tempfile.TemporaryDirectory() as temp:
        root = child.save(Path(temp) / "snapshot")
        loaded = State.load(root)
        assert loaded.compaction == child.compaction
        assert torch.equal(loaded._read_tensors()["P"], child._read_tensors()["P"])
        second = compact(loaded, force=True)
        assert second.clock == 65536 and second.compaction.effective_ratio == 64
        assert second._read_tensors()["P"][-1] == 65535
        assert torch.equal(second._read_tensors()["R.0"], child._read_tensors()["R.0"])


def test_legacy_snapshot_without_optional_receipt_remains_readable(geometry):
    import json
    parent = state(32)
    with tempfile.TemporaryDirectory() as temp:
        root = parent.save(Path(temp) / "legacy")
        manifest = json.loads((root / "manifest.json").read_text())
        manifest.pop("compaction")
        (root / "manifest.json").write_text(json.dumps(manifest))
        loaded = State.load(root)
        assert loaded.compaction is None and loaded.clock == parent.clock
        assert "P" not in loaded._read_tensors()


def test_selector_failure_leaves_original_snapshot_intact(geometry):
    original = state()
    before = original._read_tensors()["K.1"].clone()
    with patch("latentport._compact.weights.score", side_effect=RuntimeError("allocation")), \
         patch("latentport._compact._device", return_value=torch.device("cpu")):
        with pytest.raises(RuntimeError, match="allocation"):
            original.compact()
    assert torch.equal(before, original._read_tensors()["K.1"])


def test_physical_mask_and_rope_clock_are_separate():
    values = {"P": torch.tensor([0, 17, 65535])}
    args = continuation_inputs(values, 65536, 2, device="cpu", dtype=torch.bfloat16)
    assert args["position_ids"].tolist() == [[65536,65537]]
    mask = args["attention_mask"]["full_attention"]
    assert mask.shape == (1,1,2,5)
    assert mask[0,0,0,:4].eq(0).all() and mask[0,0,0,4].isneginf()
    assert mask[0,0,1].eq(0).all()
    assert extend_positions(values,65536,2).tolist() == [0,17,65535,65536,65537]


@pytest.mark.parametrize("positions", [[0,0], [2,1], [-1,1], [0,10]])
def test_invalid_sparse_metadata_fails_before_backend_mutation(positions):
    with pytest.raises(StateError):
        physical_length({"P": torch.tensor(positions)}, 10)


def test_dense_shape_cannot_be_mistaken_for_sparse_shape(geometry):
    values = state(2048)._read_tensors()
    values["P"] = torch.arange(1024)
    with pytest.raises(StateError, match="shape"):
        validate_components(values, 2048, "target")


def test_unsupported_model_and_invalid_policy_are_explicit(geometry):
    original = state(32)
    with pytest.raises(CapabilityError):
        replace(original, identity=replace(IDENTITY, repo_id="Qwen/Qwen3.5-4B")).compact()
    for ratio in (0, 2, 8.0, True):
        with pytest.raises(ValueError):
            original.compact(ratio=ratio)
    assert budget(131072,8)==16384
    with pytest.raises(ValueError, match="Nonfinite"):
        choose([float("nan")]*2048,1024)


def test_dense_mtp_keeps_existing_partial_capacity_behavior():
    from types import SimpleNamespace
    from latentport._native.engine import Engine, NativeOptions
    engine=object.__new__(Engine)
    engine.options=NativeOptions(context_size=64)
    engine.binding=SimpleNamespace(sparse_state=False,check=lambda value:value)
    engine.lib=SimpleNamespace(lp_clock=lambda _:62)
    engine._model=lambda _: (None,SimpleNamespace(context_size=64))
    engine._ensure_room('target',9,partial=True)
    with pytest.raises(StateError,match='configured context'):
        engine._ensure_room('target',9)


def test_local_weight_identity_depends_on_contents_not_directory():
    from latentport.single import _weight_fingerprint
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for name in ('a','b'):
            (root/name).mkdir()
            (root/name/'model.safetensors').write_bytes(b'original-tensor-bytes')
        first=_weight_fingerprint(root/'a')
        assert first==_weight_fingerprint(root/'b')
        (root/'b/model.safetensors').write_bytes(b'different-tensor-bytes')
        assert first!=_weight_fingerprint(root/'b')
