"""Fault injection at public request and ownership boundaries; no GPU required."""
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import runpy
from unittest.mock import Mock, patch

import pytest
import torch

from latentport import GenerationInterrupted, Port, State, StateError
from latentport.composite import Composite
from latentport.context import ContextOptions
from test_runtime import FakeHandle, pair


@pytest.mark.parametrize("policy", ["strict", "cached_handoff"])
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("failure", ["translate", "allocation", "callback"])
def test_composite_failure_restores_request_boundary_and_retries(tmp_path, policy, followup, failure):
    source, target, bridge, backend = pair(tmp_path)
    chat = Composite(source, target, bridge).session(policy=policy)
    if followup:
        chat.ask("Previous")
    if followup and policy == "cached_handoff" and failure == "translate":
        return  # Cached target turns do not translate.
    parent, reader, target_session, synced = chat.state, chat.reader.state, chat.target_session, chat._synced
    emitted = []
    def callback(token):
        emitted.append(token)
        raise RuntimeError("callback failure")
    guard = patch.object(bridge, "translate", side_effect=RuntimeError("translate failure")) if failure == "translate" else nullcontext()
    backend.fail_target = failure == "allocation"
    with guard, pytest.raises(RuntimeError) as error:
        chat.ask("Retry me", on_token=callback if failure == "callback" else None)
    assert chat.state is parent and chat.reader.state is reader
    assert chat.target_session is target_session and chat._synced == synced
    if emitted:
        assert isinstance(error.value, GenerationInterrupted)
        assert error.value.token_ids == tuple(emitted)
        assert error.value.recovery_state is parent
    backend.fail_target = False
    answer = chat.ask("Retry me")
    assert answer.finished
    assert answer.state.messages.count(("user", "Retry me")) == 1


@pytest.mark.parametrize("existing", [False, True])
def test_ordinary_ask_retries_allocation_and_invalid_request(tmp_path, existing):
    _, target, _, backend = pair(tmp_path)
    session = target.session()
    if existing:
        session.ask("Old")
    parent = session.state
    backend.fail_target = True
    with pytest.raises(RuntimeError, match="allocation"):
        session.ask("New")
    assert session.state is parent
    backend.fail_target = False
    with pytest.raises(ValueError):
        session.ask("New", max_new_tokens=0)
    assert session.state is parent
    assert session.ask("New").state.messages.count(("user", "New")) == 1


@pytest.mark.parametrize("failure", ["first_step", "later_step", "snapshot", "stats", "close", "publish", "cleanup", "cancel"])
def test_stream_failure_reports_exposed_tokens_and_rolls_back(tmp_path, failure):
    _, target, _, backend = pair(tmp_path)
    session = target.session()
    parent = session.prepare("Question")
    emitted, handles = [], []
    class Handle(FakeHandle):
        def step(self, token):
            if failure == "first_step" or (failure == "later_step" and self.clock > parent.clock):
                raise RuntimeError("step failure")
            super().step(token)
        def snapshot(self):
            if failure == "snapshot":
                raise RuntimeError("snapshot failure")
            return super().snapshot()
        def stats(self):
            if failure == "stats":
                raise RuntimeError("stats failure")
            return {}
        def close(self):
            self.closed = True
            if failure == "close":
                raise RuntimeError("close failure")
    def begin(identity, values, clock, **kwargs):
        handle = Handle(clock)
        handles.append(handle)
        return handle
    def emit(token):
        assert handles[-1].clock == parent.clock + len(emitted) + 1
        emitted.append(token)
        if failure == "cancel":
            raise KeyboardInterrupt()
    backend.finish_request = Mock(side_effect=RuntimeError("cleanup failure") if failure == "cleanup" else None)
    backend.peak_vram_bytes = Mock(side_effect=RuntimeError("publish failure") if failure == "publish" else None,
                                   return_value=None)
    with patch.object(backend, "begin_decode", side_effect=begin), pytest.raises(RuntimeError) as error:
        session.generate(on_token=emit)
    assert handles[0].closed and session.state is parent
    backend.finish_request.assert_called_once()
    if failure == "first_step":
        assert emitted == [] and not isinstance(error.value, GenerationInterrupted)
    else:
        assert isinstance(error.value, GenerationInterrupted)
        assert error.value.token_ids == tuple(emitted)
        assert error.value.recovery_state is parent
    backend.finish_request.side_effect = backend.peak_vram_bytes.side_effect = None
    assert session.generate().finished


def test_mtp_callback_failure_invalidates_entire_request_stream(tmp_path):
    _, target, _, backend = pair(tmp_path)
    backend.supports_mtp = True
    session = target.session()
    parent = session.prepare("Q")
    class Handle(FakeHandle):
        mtp = True
        def greedy_batch(self, limit, eos):
            self.clock += 2
            return [66, 33]
        def close(self):
            self.closed = True
    handle = Handle(parent.clock)
    emitted = []
    def callback(token):
        emitted.append(token)
        if token == 66:
            assert handle.clock == parent.clock + 3
            raise RuntimeError("callback")
    with patch.object(backend, "begin_decode", return_value=handle), pytest.raises(GenerationInterrupted) as error:
        session.generate(mtp=True, on_token=callback)
    assert error.value.token_ids == (65, 66) and session.state is parent and handle.closed


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("throws", [False, True])
def test_translator_cannot_mutate_parent_or_forks(tmp_path, inference, disk, throws):
    source, _, bridge, _ = pair(tmp_path)
    with torch.inference_mode(inference):
        parent = source.prefill("Hello")
    if disk:
        parent = State.load(parent.save(tmp_path / "parent"))
    branch = parent.fork()
    before = parent._tensors()["H"]
    def translate(values, clock, metadata):
        values["H"].add_(1)
        if throws:
            raise RuntimeError("translator failure")
        return values
    with patch.object(bridge._translator, "translate", side_effect=translate) as translator:
        try:
            with torch.inference_mode(inference):
                child = bridge.translate(branch)
        except RuntimeError as exc:
            assert str(exc) in ("translator failure", "Translator mutated an immutable source state")
        else:
            with torch.inference_mode():
                child._read_tensors()["H"].zero_()
        translator.assert_called_once()
    assert torch.equal(parent._read_tensors()["H"], before)
    assert torch.equal(branch._read_tensors()["H"], before)


def test_port_owns_backend_and_model_views_close_independently(tmp_path):
    source, target, bridge, backend = pair(tmp_path)
    backend.close = Mock()
    port = Port(source, target, bridge)
    parent = source.prefill("Keep")
    source_session = source.session(parent)
    with source:
        pass
    backend.close.assert_not_called()
    assert target.session().ask("Still alive").finished
    with pytest.raises(StateError, match="closed"):
        source_session.append("x")
    port.close()
    port.close()
    target.close()
    backend.close.assert_called_once()
    with pytest.raises(StateError, match="closed"):
        target.session()
    assert State.load(parent.save(tmp_path / "closed-owner")).clock == parent.clock
    source_session.close()


def test_standalone_model_owns_backend(tmp_path):
    _, target, _, backend = pair(tmp_path)
    backend.close = Mock()
    with target:
        pass
    target.close()
    backend.close.assert_called_once()


def test_composite_commit_failure_invalidates_stream_and_is_retryable(tmp_path):
    from latentport import Session
    source, target, bridge, _ = pair(tmp_path)
    chat = Composite(source, target, bridge).session()
    descriptor = Session.state
    emitted = []
    def setter(session, value):
        if session is chat.reader and value is None:
            raise RuntimeError("commit failure")
        descriptor.fset(session, value)
    with patch.object(Session, "state", property(descriptor.fget, setter)):
        with pytest.raises(GenerationInterrupted) as error:
            chat.ask("Q", on_token=emitted.append)
    assert error.value.token_ids == tuple(emitted) == (65, 33)
    assert chat.state is None and chat.target_session is None
    assert chat.ask("Q").finished


def test_cached_target_remains_usable_after_source_view_closes(tmp_path):
    source, target, bridge, backend = pair(tmp_path)
    backend.close = Mock()
    port = Port(source, target, bridge)
    chat = port.composite.session()
    chat.ask("First")
    source.close()
    assert chat.ask("Second").finished
    backend.close.assert_not_called()
    port.close()


def test_unstructured_append_marks_existing_messages_incomplete(tmp_path):
    source, _, _, _ = pair(tmp_path)
    parent = source.capture((("user", "Old"),))
    child = source.append(parent, "New raw data")
    assert parent.messages_complete and not child.messages_complete


def test_resident_noop_preserves_live_handle():
    from latentport import Model
    from latentport._reference.bf16 import BF16Backend
    from test_extended_context import tiny_backend
    with tiny_backend(BF16Backend) as (backend, identity, _):
        model = Model(identity, backend, role="target")
        with model.live() as session:
            session.append([2, 3])
            handle = session._resident
            with patch.object(State, "compact", autospec=True, side_effect=lambda state, **kw: state):
                with patch.object(handle, "close", wraps=handle.close) as close:
                    assert session.compact().clock == 2
                    close.assert_not_called()
            assert session._resident is handle
            session.append([5])
            assert session.clock == 3


@pytest.mark.parametrize("ids", [[66.9], [True], [-1], ["66"], []])
def test_exact_tokens_rejected_before_backend_work(tmp_path, ids):
    from latentport.stream import ResidentCache
    source, target, bridge, backend = pair(tmp_path)
    parent = source.prefill("hello")
    backend.live = Mock()
    resident = ResidentCache(source)
    calls = list(backend.calls)
    with pytest.raises(ValueError):
        source.append(parent, ids)
    with pytest.raises(ValueError):
        resident.append(ids)
    with pytest.raises(ValueError):
        Port(source, target, bridge).handoff(parent, target_suffix_ids=ids)
    assert backend.calls == calls
    resident._live.append.assert_not_called()


def test_handoff_marks_arbitrary_suffix_transcript_incomplete(tmp_path):
    source, target, bridge, _ = pair(tmp_path)
    port = Port(source, target, bridge)
    parent = source.capture((("user", "Old"),))
    state = port.handoff(parent, target_suffix_ids=target.codec.encode("<user>New!<assistant>"))
    assert not state.messages_complete and state.messages == ()
    answer = target.session(state).generate()
    assert "New" in answer.state.rendered_text
    assert not answer.state.messages_complete and answer.state.messages == ()
    restored = State.load(answer.state.save(tmp_path / "handoff"))
    assert not restored.messages_complete


@pytest.mark.parametrize("field,value", [("messages", [["user", "red"]]), ("clock", 999),
                                         ("identity", {}), ("rendered_token_ids", [42])])
def test_manifest_edits_fail_integrity_check(tmp_path, field, value):
    source, _, _, _ = pair(tmp_path)
    path = source.prefill("blue").save(tmp_path / "snapshot")
    manifest = path / "manifest.json"
    data = json.loads(manifest.read_text())
    data[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(StateError, match="manifest checksum"):
        State.load(path)


def test_snapshot_receipt_required_and_legacy_requires_explicit_opt_in(tmp_path):
    source, _, _, _ = pair(tmp_path)
    path = source.prefill("blue").save(tmp_path / "snapshot")
    (path / "manifest.sha256").unlink()
    with pytest.raises(StateError, match="missing"):
        State.load(path, allow_legacy=True)
    manifest = path / "manifest.json"
    data = json.loads(manifest.read_text())
    data["format"] = "latentport.state.v1"
    manifest.write_text(json.dumps(data))
    with pytest.raises(StateError, match="allow_legacy"):
        State.load(path)
    assert State.load(path, allow_legacy=True).clock == 4


def test_noop_compaction_does_not_install(tmp_path):
    source, _, _, backend = pair(tmp_path)
    parent = source.prefill("hello")
    session = source.session(parent)
    with patch.object(State, "compact", autospec=True, side_effect=lambda state, **kw: state):
        assert session.compact() is parent
    assert not any(call[1] == "install" for call in backend.calls)


def test_capacity_aware_compaction_reserves_budget_before_install(tmp_path):
    _, target, _, backend = pair(tmp_path)
    backend.options = ContextOptions(physical_limit=32768)
    backend.check_capacity = backend.options.check
    parent = target.prefill("x" * 32768)
    session = target.session(parent)
    def compact(state, **kwargs):
        assert kwargs["force"] is True
        return replace(state, _components={"K.0": torch.zeros(1, 1, 4096, 1),
                                           "V.0": torch.zeros(1, 1, 4096, 1), "P": torch.arange(4096)})
    with patch.object(State, "compact", autospec=True, side_effect=compact):
        child = session.compact(reserve_tokens=1024)
    assert child.physical_kv_length == 4096 and child.clock == 32768
    session.state = parent
    with patch.object(State, "compact") as operation, pytest.raises(StateError, match="Physical"):
        session.compact(reserve_tokens=32768)
    operation.assert_not_called()
    assert session.state is parent


def test_default_model_compaction_rejects_untrained_revision_before_loading(tmp_path):
    from test_compact import state
    parent = state(32)
    parent = replace(parent, identity=replace(parent.identity, revision="different"))
    with patch.object(State, "_read_tensors") as read, pytest.raises(RuntimeError, match="trained base revision"):
        parent.compact(force=True)
    read.assert_not_called()


@pytest.mark.parametrize("backend", [None, "bf16", "nf4", "gguf"])
def test_compact_example_loader_configuration(tmp_path, backend):
    history = tmp_path / "history.txt"
    history.write_text("fixture")
    argv = ["compact.py", str(history), "Question"] + (["--backend", backend] if backend else [])
    with patch("sys.argv", argv), patch("latentport.load_model") as loader:
        runpy.run_path(str(Path(__file__).parents[1] / "examples/compact.py"), run_name="__main__")
    options = loader.call_args.kwargs
    assert options["backend"] == (backend or "nf4")
    assert (options["native_options"] is not None) == (backend == "gguf")


@pytest.mark.parametrize("mtp", [False, True])
def test_chat_example_requests_mtp_allocation(mtp):
    argv = ["chat.py"] + (["--mtp"] if mtp else [])
    with patch("sys.argv", argv), patch("latentport.load") as loader:
        answer = loader.return_value.__enter__.return_value.composite.session.return_value.ask.return_value
        answer.stats.request_ttft_s = 0.01
        answer.text, answer.finished = "fixture", True
        runpy.run_path(str(Path(__file__).parents[1] / "examples/chat.py"), run_name="__main__")
    assert loader.call_args.kwargs["native_options"].mtp_drafts == (4 if mtp else 0)
    loader.return_value.__enter__.return_value.warm.assert_called_once_with(mtp=mtp)
