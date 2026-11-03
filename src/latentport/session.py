"""Mutable execution policy over immutable, forkable state snapshots."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable

from .errors import CapabilityError, CompatibilityError, HandoffError, StateError
from .identity import Answer, RunStats
from .state import State


class Session:
    def __init__(self, model: "Model", state: State | None = None):
        self.model, self.state = model, state
        self._pending_decode = None
        self.last_handoff_s = 0.0
        if state is not None:
            state.require_model(model.identity)

    def fork(self, count: int = 1) -> "Session | tuple[Session, ...]":
        if self.state is None:
            raise StateError("Capture state before forking")
        states = self.state.fork(count)
        if count == 1:
            return Session(self.model, states)
        return tuple(Session(self.model, state) for state in states)

    def prepare(self, question: str) -> State:
        """Consume a new user turn and native assistant prefix, without generating."""
        if not isinstance(question, str) or not question:
            raise ValueError("ask needs a nonempty user question")
        if self.state is None:
            self.state = self.model.capture((("user", question),), generation_prompt=True)
            self._pending_decode = None
            return self.state
        state = self.state
        if state.phase not in ("context_ready", "turn_ready"):
            raise StateError("A new question requires a context or completed turn")
        if not state.messages:
            # A captured raw prefix can still feed different branch questions.
            # Keep it a literal stream; no whole-history template is invented.
            if state.rendered_text is None:
                raise StateError("Raw state has no exact rendered stream")
            suffix = self.model.codec.render((("user", question),), generation_prompt=True)
            separator = self.model.codec.encode("\n") if not state.rendered_text.endswith("\n") else ()
            next_state = self.model.append(state, separator + suffix.token_ids, phase="answer_ready")
            self.state = next_state
            self._pending_decode = None
            return next_state
        messages = state.messages + (("user", question),)
        desired = self.model.codec.render(messages, generation_prompt=True)
        old_ids = state.rendered_token_ids
        if old_ids is None:
            raise StateError("The prior state has no exact token stream")
        if desired.token_ids[:len(old_ids)] == old_ids:
            delta = desired.token_ids[len(old_ids):]
            expected = desired.token_ids
        else:
            # Qwen's whole-history template strips thinking from earlier
            # assistant turns. A live state cannot be rewritten. Append only
            # the new native user/assistant boundary instead.
            if state.rendered_text is None or not (
                state.rendered_text.endswith("<|im_end|>") or
                state.rendered_text.endswith("<|im_end|>\n")
            ):
                raise CapabilityError("Prior state lacks a native turn boundary; replay is forbidden")
            suffix = self.model.codec.render((("user", question),), generation_prompt=True)
            newline = self.model.codec.encode("\n") if state.rendered_text.endswith("<|im_end|>") else ()
            delta = newline + suffix.token_ids
            expected = old_ids + delta
        if not delta:
            raise StateError("No new question tokens")
        next_state = self.model.append(state, delta, phase="answer_ready", messages=messages)
        if next_state.rendered_token_ids != expected:
            raise StateError("Native turn tokenization changed after append")
        self.state = next_state
        self._pending_decode = None
        return next_state

    def ask(self, question: str, *, max_new_tokens: int = 512, mtp: bool = False,
            on_token: Callable[[int], None] | None = None) -> Answer:
        started = perf_counter()
        self.prepare(question)
        prefill = perf_counter() - started
        result = self.generate(max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
        field = "reader_prefill_s" if self.model.role == "source" else "target_prefill_s"
        return replace(result, stats=replace(result.stats, **{field: prefill,
                                     "request_ttft_s": prefill + result.stats.first_token_s,
                                     "total_s": perf_counter() - started}))

    def generate(self, *, max_new_tokens: int = 512, mtp: bool = False,
                 on_token: Callable[[int], None] | None = None) -> Answer:
        from contextlib import nullcontext
        started = perf_counter()
        guard = (self.model.backend.generation_guard() if hasattr(self.model.backend, "generation_guard")
                 else nullcontext())
        with guard:
            queue = perf_counter() - started
            answer = self._generate(max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
        return replace(answer, stats=replace(answer.stats, queue_s=queue,
                       first_token_s=answer.stats.first_token_s+queue,
                       request_ttft_s=answer.stats.request_ttft_s+queue,
                       total_s=answer.stats.total_s+queue))

    def _generate(self, *, max_new_tokens: int = 512, mtp: bool = False,
                  on_token: Callable[[int], None] | None = None) -> Answer:
        if self.state is None or self.state.phase not in ("answer_ready", "in_answer"):
            raise StateError("generate requires an answer-ready or mid-answer state")
        if not 1 <= max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be 1..4096")
        if mtp and not self.model.backend.supports_mtp:
            raise CapabilityError("MTP is not available in this backend; no silent fallback")
        initial = self.state
        model = self.model
        started = perf_counter()
        transferred = started
        if self._pending_decode is not None:
            handle = self._pending_decode
            self._pending_decode = None
            if mtp and not getattr(handle, "mtp", False):
                if hasattr(handle, "enable_mtp"):
                    handle.enable_mtp()
                else:
                    raise CapabilityError("Prepared handoff did not initialize MTP")
        else:
            values = (initial._read_tensors() if getattr(model.backend, "copy_on_write", False)
                      else initial._tensors(device=getattr(model.backend, "device", None)))
            transferred = perf_counter()
            handle = model.backend.begin_decode(
                model.identity, values, initial.clock, mtp=mtp)
        installed = perf_counter()
        tokens = []
        eos = model.backend.eos_ids(model.identity)
        first_token_at = None
        try:
            while len(tokens) < max_new_tokens:
                if mtp and tokens and hasattr(handle, "greedy_batch"):
                    # The first token is emitted promptly from translated H.
                    # Later batches contain only target-verified committed tokens.
                    chunk = handle.greedy_batch(max_new_tokens-len(tokens), eos)
                    tokens.extend(chunk)
                    if on_token is not None:
                        for token in chunk:
                            on_token(token)
                    if chunk[-1] in eos:
                        break
                    continue
                logits = handle.logits
                token = int(logits[0, -1].argmax()) if logits.ndim == 3 else int(logits[0].argmax())
                tokens.append(token)
                if first_token_at is None:
                    first_token_at = perf_counter()
                if on_token is not None:
                    on_token(token)
                # Every committed token, including EOS and the cap token, is
                # ingested before state is published for resume/fork.
                handle.step(token)
                if token in eos:
                    break
            snapshot = handle.snapshot()
            native_stats = handle.stats() if hasattr(handle, "stats") else {}
        finally:
            # Public session retains its original snapshot if native work,
            # snapshotting or a streaming callback fails.
            if hasattr(handle, "close"):
                handle.close()
        finished = bool(tokens and tokens[-1] in eos)
        full_ids = (initial.rendered_token_ids or ()) + tuple(tokens) if initial.rendered_token_ids is not None else None
        current = State(identity=model.identity, clock=handle.clock,
                        phase="turn_ready" if finished else "in_answer",
                        origin="native_generation", target_prompt_tokens=initial.target_prompt_tokens,
                        rendered_text=model.codec.decode(full_ids, special=True) if full_ids is not None else None,
                        rendered_token_ids=full_ids, messages=initial.messages,
                        generated_token_ids=initial.generated_token_ids + tuple(tokens),
                        parent_id=initial.state_id, _components=snapshot)
        if finished and current.messages and tokens[-1] == model.codec.chat_eos_id:
            prefix = ""
            if initial.phase == "in_answer":
                base = model.codec.render(initial.messages, generation_prompt=True).decoded_text
                if initial.rendered_text is None or not initial.rendered_text.startswith(base):
                    raise StateError("Committed assistant prefix differs from native template")
                prefix = initial.rendered_text[len(base):]
            assistant = prefix + model.codec.decode(tokens[:-1], special=True)
            current = replace(current, messages=current.messages + (("assistant", assistant),))
        self.state = current
        elapsed = perf_counter()
        peak = getattr(model.backend, "peak_vram_bytes", lambda: None)()
        return Answer(text=model.codec.decode(tokens, special=False), token_ids=tuple(tokens),
                      state=current, finished=finished,
                      stats=RunStats(device_transfer_s=native_stats.pop("device_transfer_s",transferred-started),
                                     state_install_s=native_stats.pop("state_install_s",installed-transferred),
                                     first_token_s=(first_token_at-started) if first_token_at else 0.0,
                                     request_ttft_s=(first_token_at-started) if first_token_at else 0.0,
                                     generation_s=elapsed-installed,
                                     total_s=elapsed-started,
                                     peak_vram_bytes=peak,
                                     target_prompt_tokens=current.target_prompt_tokens,
                                     generated_tokens=len(tokens), backend=model.identity.backend,
                                     mtp_requested=mtp,
                                     mtp_used=bool(native_stats.get("mtp_batches", 0)),
                                     **native_stats))

    def escalate(self, *, target: "Model", bridge: "Bridge") -> State:
        if self.state is None:
            raise StateError("No state to escalate")
        if self.state.phase not in ("answer_ready", "in_answer", "context_ready"):
            raise StateError("Escalation requires a live context or answer boundary")
        if bridge.source.identity != self.model.identity or bridge.target.identity != target.identity:
            raise CompatibilityError("Bridge direction or exact model identity differs")
        started = perf_counter()
        translated = bridge.translate(self.state, target=target)
        # Translation and full installation must succeed before changing the
        # source session. An exception preserves it for retry or continuation.
        values = (translated._read_tensors() if getattr(target.backend, "copy_on_write", False)
                  else translated._tensors(device=getattr(target.backend, "device", None)))
        prepared = target.backend.begin_decode(
            target.identity, values,
            translated.clock)
        self.model, self.state = target, translated
        self._pending_decode = prepared
        self.last_handoff_s = perf_counter() - started
        return translated


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import Model
    from .bridge import Bridge
