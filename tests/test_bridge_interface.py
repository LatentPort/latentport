"""The public transfer and conversation calls do not depend on model pair."""
from pathlib import Path
import runpy
from unittest.mock import patch

import pytest
import torch

from latentport import CompatibilityError, GenerationInterrupted, Port
from latentport.hub import BRIDGE_REVISIONS, LAYER_ALIGNED
from test_runtime import pair


@pytest.fixture(params=["qwen-hybrid-mixed-v1", LAYER_ALIGNED])
def port(tmp_path, request):
    source, target, bridge, backend = pair(tmp_path)
    bridge.manifest["architecture"] = request.param
    bridge._translator.translate = lambda values, clock, metadata: {
        "H": torch.tensor([metadata["target_length"]], dtype=torch.bfloat16)}
    return Port(source, target, bridge)


def test_same_capture_translate_ask_workflow(port):
    history = [("user", "The ticket is red.")]
    state = port.source.capture(history)
    transferred = port.bridge.translate(state)
    assert transferred.target_prompt_tokens == 0
    assert transferred.messages == tuple(history) and transferred.messages_complete
    answer = port.target.session(transferred).ask("What color?")
    assert answer.finished
    assert answer.state.messages == tuple(history) + (("user", "What color?"), ("assistant", "A"))
    target_calls = [row for row in port.target.backend.calls if row[0] == "target"]
    assert not any(row[1] == "capture" for row in target_calls)
    assert [row[2] for row in target_calls if row[1] == "advance"] == [port.target.codec.encode("<user>What color?!<assistant>")]
    assert state.phase == "context_ready" and state.messages == tuple(history)


@pytest.mark.parametrize("policy", ["cached_handoff", "strict"])
@pytest.mark.parametrize("context", [None, [("user", "Evidence")]])
def test_same_conversation_interface_handles_both_transfer_boundaries(port, policy, context):
    chat = port.composite.session(context, policy=policy)
    with patch.object(port.bridge, "translate", wraps=port.bridge.translate) as translate:
        first = chat.ask("First")
        second = chat.ask("Second")
    expected_transfers = 1 if policy == "cached_handoff" else 2
    if context is None and port.bridge.requires_request_suffix:
        expected_transfers -= 1
    assert translate.call_count == expected_transfers
    assert first.finished and second.finished
    assert second.state.messages.count(("user", "First")) == 1
    assert second.state.messages.count(("user", "Second")) == 1
    assert second.state.messages.count(("assistant", "A")) == 2
    if port.bridge.requires_request_suffix:
        for call in translate.call_args_list:
            assert call.args[0].phase in ("context_ready", "turn_ready")
        assert first.stats.target_prefill_s > 0 and first.stats.target_prompt_tokens > 0
        assert first.stats.request_ttft_s >= first.stats.translation_s + first.stats.target_prefill_s
    else:
        assert first.stats.target_prompt_tokens == 0


def test_same_conversation_resume_and_fork_interface(port):
    chat = port.composite.session([("user", "Evidence")])
    assert not chat.ask("Q", max_new_tokens=1).finished
    assert chat.generate().finished
    before = chat.state
    left, right = chat.fork(2)
    assert left.ask("Left").finished and right.ask("Right").finished
    assert chat.state is before
    assert ("user", "Right") not in left.state.messages


@pytest.mark.parametrize("failure", ["translation", "allocation", "callback"])
@pytest.mark.parametrize("policy", ["cached_handoff", "strict"])
def test_shared_interface_keeps_failed_asks_retryable(port, failure, policy):
    chat = port.composite.session([("user", "Evidence")], policy=policy)
    parent = chat.state
    original = port.bridge.translate
    def translate(state):
        if failure == "translation":
            raise RuntimeError("translation failure")
        return original(state)
    def callback(token):
        raise RuntimeError("callback failure")
    port.source.backend.fail_target = failure == "allocation"
    with patch.object(port.bridge, "translate", side_effect=translate), pytest.raises(RuntimeError) as error:
        chat.ask("Retry", on_token=callback if failure == "callback" else None)
    assert chat.state is parent and chat.target_session is None
    if failure == "callback":
        assert isinstance(error.value, GenerationInterrupted)
        assert error.value.recovery_state is parent
    port.source.backend.fail_target = False
    assert chat.ask("Retry").state.messages.count(("user", "Retry")) == 1


def test_same_run_interface_splits_final_question_for_query_bridges(port):
    messages = [{"role": "user", "content": "Evidence"}, {"role": "user", "content": "Q"}]
    with patch.object(port.bridge, "translate", wraps=port.bridge.translate) as translate:
        answer = port.composite.run(messages)
    assert answer.finished
    prefix = translate.call_args.args[0]
    expected = (("user", "Evidence"),) if port.bridge.requires_request_suffix else (("user", "Evidence"), ("user", "Q"))
    assert prefix.messages == expected
    assert answer.state.messages == (("user", "Evidence"), ("user", "Q"), ("assistant", "A"))


def test_query_bridge_still_rejects_an_answer_ready_state(tmp_path):
    source, target, bridge, _ = pair(tmp_path)
    bridge.manifest["architecture"] = LAYER_ALIGNED
    ready = source.capture([("user", "Q")], generation_prompt=True)
    with pytest.raises(CompatibilityError, match="before the final request"):
        bridge.translate(ready)


def test_translate_accepts_a_completed_turn_as_the_next_question_prefix(port):
    source_chat = port.source.session()
    source_chat.ask("Previous")
    parent = source_chat.state
    translated = port.bridge.translate(parent)
    assert translated.phase == "turn_ready" and parent.phase == "turn_ready"
    assert port.target.session(translated).ask("Next").finished


@pytest.mark.parametrize("backend", ["nf4", "bf16", "gguf"])
@pytest.mark.parametrize("repo", list(BRIDGE_REVISIONS))
def test_chat_command_uses_same_flags_for_every_bridge(repo, backend):
    argv = ["chat.py", "--repo", repo, "--backend", backend,
            "--context", "Evidence", "--prompt", "Q", "--followup", "Next"]
    with patch("sys.argv", argv), patch("latentport.load") as load:
        context = load.return_value.__enter__.return_value
        chat = context.composite.session.return_value
        result = chat.ask.return_value
        result.stats.request_ttft_s, result.text, result.finished = 0.01, "A", True
        runpy.run_path(str(Path(__file__).parents[1] / "examples/chat.py"), run_name="__main__")
    assert load.call_args.args == (repo,)
    assert load.call_args.kwargs["backend"] == backend
    assert (load.call_args.kwargs["native_options"] is not None) == (backend == "gguf")
    context.composite.session.assert_called_once_with([("user", "Evidence")], policy="cached_handoff")
    assert [call.args[0] for call in chat.ask.call_args_list] == ["Q", "Next"]
