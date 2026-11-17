"""Execution composition and transactional failure tests, without GPU models."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from latentport import Session, State, CompatibilityError
from latentport.model import Model
from latentport._reference.bf16 import BF16Backend
from test_extended_context import tiny_backend
from test_runtime import FakeCodec, pair


def test_resident_session_append_ask_fork_save_and_resume(tmp_path):
    with tiny_backend(BF16Backend) as (backend, identity, net):
        backend.codec = lambda _: FakeCodec()
        backend.eos_ids = lambda _: frozenset({127})
        model = Model(identity, backend, role="target")
        with model.live() as session:
            assert isinstance(session, Session) and session.resident
            session.append([2, 3, 5, 7])
            resident = session._resident
            session.append([11])
            assert session._resident is resident and session._state is None
            parent = session.snapshot()
            with session.fork() as branch:
                assert branch.resident
                branch.append([13])
                assert session.clock == 5 and branch.clock == 6
                assert parent.clock == 5
                branch.save(tmp_path / "branch")
                restored = State.load(tmp_path / "branch")
                assert restored.clock == 6 and restored.rendered_text is None
            answer = session.ask("Q", max_new_tokens=1)
            assert answer.state.clock > 5 and answer.state.rendered_text is None
            old_clock = session.clock
            session.append([17])
            assert session.clock == old_clock + 1
            frozen = session.snapshot(release=True)
            assert session.closed and frozen.clock == old_clock + 1


@pytest.mark.parametrize("failure", ["operation", "install", "identity"])
def test_resident_transformation_failure_retains_exact_source_and_can_resume(failure):
    with tiny_backend(BF16Backend) as (backend, identity, net):
        model = Model(identity, backend, role="target")
        with model.session(resident=True) as session:
            session.append([2, 3, 5])
            before = session.snapshot()
            def operation(state, **kwargs):
                if failure == "operation":
                    raise RuntimeError("operation failed")
                if failure == "identity":
                    return replace(state, identity=replace(identity, weights="wrong"))
                return state
            with patch.object(State, "compact", side_effect=operation, autospec=True):
                if failure == "install":
                    with patch("latentport.stream.ResidentCache", side_effect=RuntimeError("install failed")):
                        with pytest.raises(RuntimeError, match="install failed"):
                            session.compact()
                else:
                    with pytest.raises((RuntimeError, CompatibilityError)):
                        session.compact()
            after = session.snapshot()
            assert session.model is model and after.clock == before.clock
            assert all(torch.equal(v, after._read_tensors()[k]) for k, v in before._read_tensors().items())
            session.append([7])
            assert session.clock == 4


def test_shared_transaction_switches_model_only_after_successful_install(tmp_path):
    source, target, bridge, backend = pair(tmp_path)
    session = source.session(source.prefill("hello"))
    before = session.state
    backend.fail_target = True
    with pytest.raises(RuntimeError, match="allocation failed"):
        session.escalate(target=target, bridge=bridge)
    assert session.model is source and session.state is before
    backend.fail_target = False
    session.escalate(target=target, bridge=bridge)
    assert session.model is target and session.state.identity == target.identity
    session.close()


def test_bridge_rejects_suffix_only_and_sparse_before_translation(tmp_path):
    source, target, bridge, backend = pair(tmp_path)
    state = source.prefill("hello")
    suffix = replace(state, rendered_text=None, rendered_token_ids=None, stream_tail_ids=(1, 2))
    sparse = replace(state, _components={**state._read_tensors(),
        "K.0": torch.zeros(1, 1, 2, 1), "V.0": torch.zeros(1, 1, 2, 1), "P": torch.tensor([0, 4])})
    with patch.object(bridge._translator, "translate") as translate:
        for candidate, message in ((suffix, "alignment metadata"), (sparse, "sparse")):
            with pytest.raises(CompatibilityError, match=message):
                bridge.translate(candidate)
        translate.assert_not_called()


def test_resident_transaction_can_install_a_different_model():
    with tiny_backend(BF16Backend) as (backend, identity, net):
        source = Model(identity, backend, role="target")
        destination = Model(replace(identity, weights="other-fixture"), backend, role="target")
        with source.session(resident=True) as session:
            session.append([2, 3])
            before = session.snapshot()
            child = session._transform(lambda state: replace(state, identity=destination.identity), target=destination)
            assert session.model is destination and child.identity == destination.identity
            session.append([5])
            assert session.clock == 3 and before.clock == 2


def test_checkpoint_assignment_replaces_live_cache_and_validates_before_release():
    with tiny_backend(BF16Backend) as (backend, identity, net):
        model = Model(identity, backend, role="target")
        with model.live() as session:
            session.append([2, 3])
            checkpoint = session.snapshot()
            session.append([5, 7])
            handle = session._resident
            wrong = replace(checkpoint, identity=replace(identity, weights="wrong"))
            with pytest.raises(CompatibilityError):
                session.state = wrong
            assert session._resident is handle and session.clock == 4
            session.state = checkpoint
            assert handle.closed and session.clock == 2
            session.append([11])
            assert session.clock == 3
            session.state = None
            assert session.clock == 0
            session.append([13])
            assert session.clock == 1
