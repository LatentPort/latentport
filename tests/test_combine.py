"""LPCombine supports immutable, resident, restored and cache-only inputs."""
from dataclasses import replace

import pytest
import torch

from latentport import CapabilityError, CompatibilityError, Model, ModelIdentity, State, StateError
from latentport._models import DEFAULT_MODEL, PINNED_REVISION
from latentport._reference.bf16 import BF16Backend
from latentport._reference.nf4 import NF4Backend
from test_extended_context import tiny_backend
from test_runtime import FakeBackend, FakeCodec


@pytest.fixture(params=["bf16", "nf4", "gguf"])
def model(request):
    backend = FakeBackend()
    backend.name = request.param
    identity = ModelIdentity(DEFAULT_MODEL, PINNED_REVISION, backend.name, "fixture",
                             PINNED_REVISION, "template", backend.state_abi)
    return Model(identity, backend, role="target")


def test_combine_replays_right_in_order_and_saves_receipt(model, tmp_path):
    left = model.prefill("first\n")
    right = State.load(model.prefill("second").save(tmp_path / "right"))
    left_tensor = left._read_tensors()["H"].clone()
    output = left.combine(right, model=model)
    assert output.rendered_token_ids == left.rendered_token_ids + right.rendered_token_ids
    assert output.clock == left.clock + right.clock
    assert output.combination.method == "tokens"
    assert output.combination.left_id == left.state_id and output.combination.right_id == right.state_id
    assert output.parent_id == left.state_id and output.phase == "context_ready"
    assert model.backend.calls[-1] == (DEFAULT_MODEL, "advance", right.rendered_token_ids)
    assert torch.equal(left_tensor, left._read_tensors()["H"])
    assert State.load(output.save(tmp_path / "output")).combination == output.combination


def test_bad_input_preserves_current_session(model):
    left = model.prefill("first")
    right = model.prefill("second")
    session = model.session(left)
    wrong = replace(right, identity=replace(right.identity, weights="wrong"))
    with pytest.raises(CompatibilityError):
        session.combine(wrong)
    with pytest.raises(StateError, match="context-ready"):
        session.combine(replace(right, phase="in_answer"))
    with pytest.raises(StateError, match="token history"):
        session.combine(replace(right, rendered_token_ids=None), method="tokens")
    assert session.state is left


def test_explicit_cache_path_does_not_read_recorded_tokens(model, monkeypatch):
    left = model.prefill("one")
    right = model.prefill("two")
    if model.identity.backend != "nf4":
        with pytest.raises(CapabilityError, match="requires retained token IDs"):
            model.combine(left, right, method="cache")
        return
    monkeypatch.setattr("latentport._combine_readout.recover_tokens",
                        lambda owner, state: (65, 66, 67))
    child = model.combine(left, right, method="cache")
    assert child.rendered_token_ids[-3:] == (65, 66, 67)
    assert child.combination.method == "cache"


@pytest.mark.parametrize("kind", [BF16Backend, NF4Backend])
def test_resident_replay_opt_in_preserves_forks_and_continues(kind, tmp_path):
    with tiny_backend(kind) as (backend, identity, _):
        backend.codec = lambda _: FakeCodec()
        backend.eos_ids = lambda _: frozenset({127})
        model = Model(identity, backend, role="target")
        with model.live(retain_replay=True) as left, model.live(retain_replay=True) as right:
            left.append([2, 3, 5])
            right.append([7, 11])
            with right.fork() as fork:
                fork.append([13])
                restored = State.load(fork.save(tmp_path / "branch"))
            assert right.snapshot().rendered_token_ids == (7, 11)
            output = left.combine(restored)
            assert output.rendered_token_ids == (2, 3, 5, 7, 11, 13)
            left.append([17])
            assert left.snapshot().rendered_token_ids == (2, 3, 5, 7, 11, 13, 17)
        with model.live() as default:
            default.append([2, 3])
            assert default.snapshot().rendered_token_ids is None


def test_cli_combines_saved_states(model, tmp_path, monkeypatch):
    from latentport.__main__ import main
    left = model.prefill("left").save(tmp_path / "left")
    right = model.prefill("right").save(tmp_path / "right")
    output = tmp_path / "merged"
    monkeypatch.setattr("latentport.load_model", lambda *args, **kwargs: model)
    assert main(["combine", str(left), str(right), str(output)]) == 0
    state = State.load(output)
    assert state.rendered_text == "leftright" and state.combination.method == "tokens"
    assert main(["combine", str(left), str(right), str(output)]) == 1


@pytest.mark.parametrize("alias", [
    "LPCombine-Qwen3.5-4B", "LPCombine-Qwen3.5-9B", "LPCombine-Gemma-4-31B-it",
    "LPCombine-Qwen3-14B", "LPCombine-Qwen3.8-Flash-Next",
])
def test_family_profiles_join_ordered_histories(alias):
    from latentport._combine import ALIASES
    from latentport._models import MODEL_REVISIONS

    repo = ALIASES[alias.casefold()]
    backend = FakeBackend()
    backend.name = "bf16"
    identity = ModelIdentity(repo, MODEL_REVISIONS[repo], backend.name, "fixture",
                             "tokenizer", "template", backend.state_abi)
    with Model(identity, backend, role="target") as model:
        left, right = model.prefill("first\n"), model.prefill("second")
        merged = model.combine(left, right)
        assert merged.rendered_text == "first\nsecond"
        assert model.combine(right, left).rendered_text == "secondfirst\n"
        assert merged.identity == left.identity
        assert merged.combination.method == "tokens"
