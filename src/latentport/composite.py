"""Strict composite and cached handoff policies over a directional bridge."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable

from .errors import CapabilityError, GenerationInterrupted, StateError
from .identity import Answer


class Composite:
    def __init__(self, source: "Model", target: "Model", bridge: "Bridge"):
        self.source, self.target, self.bridge = source, target, bridge

    def session(self, context=None, *, policy: str = "cached_handoff") -> "CompositeSession":
        """strict translates each turn; cached_handoff translates only once."""
        if policy not in ("strict", "cached_handoff"):
            raise ValueError("policy must be 'strict' or 'cached_handoff'")
        reader = self.source.session(self.source.capture(context) if context is not None else None)
        return CompositeSession(self, reader, policy=policy)

    def run(self, messages, *, max_new_tokens: int = 512, mtp: bool = False,
            on_token: Callable[[int], None] | None = None) -> Answer:
        """Answer a prompt using the selected bridge's supported boundary."""
        if self.bridge.requires_request_suffix:
            from .model import _messages
            rows = _messages(messages)
            if rows is not None and rows[-1][0] != "user":
                raise ValueError("A query bridge requires a final user question")
            context = (rows[:-1] or None) if rows is not None else None
            question = rows[-1][1] if rows is not None else messages
            started = perf_counter()
            chat = self.session(context)
            captured = perf_counter() - started if context is not None else 0.0
            try:
                answer = chat.ask(question, max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
                return replace(answer, stats=replace(answer.stats,
                    reader_prefill_s=answer.stats.reader_prefill_s + captured,
                    request_ttft_s=answer.stats.request_ttft_s + captured,
                    total_s=perf_counter() - started))
            finally:
                chat.reader.close()
                if chat.target_session is not None:
                    chat.target_session.close()
        started = perf_counter()
        state = self.source.capture(messages, generation_prompt=True)
        reader_s = perf_counter() - started
        translated = self.bridge.translate(state)
        translation_s = perf_counter() - started - reader_s
        if translated.target_prompt_tokens != 0:
            raise StateError("Composite target prompt ownership violated")
        session = self.target.session(translated)
        del translated
        answer = session.generate(max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
        return replace(answer, stats=replace(answer.stats, reader_prefill_s=reader_s,
                                             translation_s=translation_s,
                                             request_ttft_s=reader_s+translation_s+answer.stats.first_token_s,
                                             total_s=perf_counter()-started))


class CompositeSession:
    def __init__(self, port: Composite, reader: "Session", *, policy: str = "cached_handoff"):
        if policy not in ("strict", "cached_handoff"):
            raise ValueError("policy must be 'strict' or 'cached_handoff'")
        self.port, self.reader = port, reader
        self._policy = policy
        self.target_session = None
        self._synced = True

    @property
    def policy(self):
        return self._policy

    @property
    def state(self):
        return self.target_session.state if self.target_session is not None else self.reader.state

    def fork(self, count: int = 1):
        """Branch a handed-off target state while retaining the cached policy."""
        if self.policy != "cached_handoff" or self.target_session is None:
            raise StateError("fork requires a cached_handoff session with an initialized target")
        sessions = self.target_session.fork(count)
        children = []
        for target in ((sessions,) if count == 1 else sessions):
            child = CompositeSession(self.port, self.port.source.session(), policy=self.policy)
            child.target_session = target
            children.append(child)
        return children[0] if count == 1 else tuple(children)

    def _sync(self):
        if self._synced:
            return 0.0
        if self.target_session is None or self.target_session.state is None:
            raise StateError("No target answer to synchronize")
        result = self.target_session.state
        if result.phase != "turn_ready" or not result.generated_token_ids:
            raise StateError("Complete the target turn before the next composite question")
        if result.generated_token_ids[-1] != self.port.target.codec.chat_eos_id:
            raise StateError("Target ended without a native chat-turn boundary")
        started = perf_counter()
        source = self.port.source
        old = self.reader.state
        if old is None or old.phase != "answer_ready":
            raise StateError("Reader is not at the corresponding answer boundary")
        assistant = self.port.target.codec.decode(result.generated_token_ids[:-1], special=True)
        messages = old.messages + (("assistant", assistant),) if old.messages else ()
        old_ids = old.rendered_token_ids
        if old_ids is None:
            raise StateError("Reader has no committed token stream")
        # Feed back the actual committed answer, including reasoning and EOS.
        # Re-rendering it through a template trims whitespace/thinking and may
        # merge the final prompt token with the first generated token.
        suffix = self.port.target.codec.decode(result.generated_token_ids, special=True)
        suffix_ids = source.codec.encode(suffix)
        if source.codec.decode(suffix_ids, special=True) != suffix:
            raise CapabilityError("Reader cannot represent the committed target answer exactly")
        next_state = source.append(old, suffix_ids,
                                   phase="turn_ready", messages=messages)
        if next_state.rendered_token_ids != old_ids + suffix_ids:
            raise StateError("Reader turn synchronization changed token IDs")
        self.reader.state = next_state
        self._synced = True
        return perf_counter() - started

    def ask(self, question: str, *, max_new_tokens: int = 512, mtp: bool = False,
            on_token: Callable[[int], None] | None = None) -> Answer:
        if self.policy == "cached_handoff" and self.target_session is not None:
            return self._ask(question, max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
        reader_state = self.reader._freeze()
        previous_target, synced = self.target_session, self._synced
        recovery_state = self.state
        emitted = []
        def emit(token):
            emitted.append(token)
            on_token(token)
        try:
            return self._ask(question, max_new_tokens=max_new_tokens, mtp=mtp,
                             on_token=emit if on_token is not None else None)
        except BaseException as exc:
            self.reader._state = reader_state
            if self.target_session is not previous_target and self.target_session is not None:
                self.target_session.close()
            self.target_session, self._synced = previous_target, synced
            if isinstance(exc, GenerationInterrupted):
                exc.recovery_state = recovery_state
            elif emitted:
                raise GenerationInterrupted(emitted, recovery_state) from exc
            raise

    def _ask(self, question: str, *, max_new_tokens: int, mtp: bool,
             on_token: Callable[[int], None] | None) -> Answer:
        if self.policy == "cached_handoff" and self.target_session is not None:
            state = self.target_session.state
            if (state.phase != "turn_ready" or not state.generated_token_ids or
                state.generated_token_ids[-1] != self.port.target.codec.chat_eos_id):
                raise StateError("Complete the target chat turn before asking the next question")
            return self.target_session.ask(question, max_new_tokens=max_new_tokens,
                                           mtp=mtp, on_token=on_token)
        started = perf_counter()
        sync_s = self._sync()
        if self.port.bridge.requires_request_suffix:
            return self._ask_after_prefix(question, max_new_tokens=max_new_tokens,
                                          mtp=mtp, on_token=on_token,
                                          started=started, sync_s=sync_s)
        source_state = self.reader.prepare(question)
        reader_s = perf_counter() - started - sync_s
        translated = self.port.bridge.translate(source_state)
        translation_s = perf_counter() - started - sync_s - reader_s
        if translated.target_prompt_tokens != 0:
            raise StateError("Composite target processed prompt tokens")
        target_session = self.port.target.session(translated)
        del translated
        try:
            result = target_session.generate(max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
        except BaseException:
            target_session.close()
            raise
        self.target_session, self._synced = target_session, False
        if self.policy == "cached_handoff":
            # Ownership moves to the target after successful generation. The
            # reader snapshot is no longer needed for this execution policy.
            self.reader.state = None
        return replace(result, stats=replace(result.stats, reader_prefill_s=reader_s,
                                             reader_sync_s=sync_s, translation_s=translation_s,
                                             request_ttft_s=sync_s+reader_s+translation_s+result.stats.first_token_s,
                                             total_s=perf_counter()-started))

    def _ask_after_prefix(self, question, *, max_new_tokens, mtp, on_token, started, sync_s):
        """Keep checkpoint-specific question ownership out of the public API."""
        if not isinstance(question, str) or not question:
            raise ValueError("ask needs a nonempty user question")
        prefix = self.reader.state
        translation_started = perf_counter()
        translated = None
        if prefix is not None:
            if prefix.phase not in ("context_ready", "turn_ready"):
                raise StateError("Query transfer requires a context or completed turn")
            translated = self.port.bridge.translate(prefix)
            if translated.target_prompt_tokens != 0:
                raise StateError("Composite target processed historical prompt tokens")
        translation_s = perf_counter() - translation_started if prefix is not None else 0.0
        reader_s = 0.0
        if self.policy == "strict":
            # Keep the reader at the answer boundary for the existing exact
            # answer synchronization path; this question is not translated.
            reader_started = perf_counter()
            self.reader.prepare(question)
            reader_s = perf_counter() - reader_started
        # With no context, there is no historical state to transfer. The target
        # reads only the new question; no synthetic prompt or replay is invented.
        candidate = self.port.target.session(translated)
        try:
            result = candidate.ask(question, max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
        except BaseException:
            candidate.close()
            raise
        self.target_session, self._synced = candidate, False
        if self.policy == "cached_handoff":
            self.reader.state = None
        return replace(result, stats=replace(result.stats,
            reader_prefill_s=reader_s, reader_sync_s=sync_s, translation_s=translation_s,
            request_ttft_s=sync_s + reader_s + translation_s + result.stats.request_ttft_s,
            total_s=perf_counter() - started))

    def generate(self, *, max_new_tokens: int = 512, mtp: bool = False,
                 on_token: Callable[[int], None] | None = None) -> Answer:
        if self.target_session is None:
            raise StateError("No target answer is in progress")
        result = self.target_session.generate(max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
        self._synced = False
        return result


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bridge import Bridge
    from .model import Model
    from .session import Session
