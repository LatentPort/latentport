"""Repeated small-reader/large-decoder policy over a directional bridge."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable

from .errors import CapabilityError, StateError
from .identity import Answer


class Composite:
    def __init__(self, source: "Model", target: "Model", bridge: "Bridge"):
        self.source, self.target, self.bridge = source, target, bridge

    def session(self, context=None) -> "CompositeSession":
        reader = self.source.session(self.source.capture(context) if context is not None else None)
        return CompositeSession(self, reader)

    def run(self, messages, *, max_new_tokens: int = 512, mtp: bool = False,
            on_token: Callable[[int], None] | None = None) -> Answer:
        """One complete prompt: source reads every token, target reads none."""
        started = perf_counter()
        state = self.source.capture(messages, generation_prompt=True)
        reader_s = perf_counter() - started
        translated = self.bridge.translate(state)
        translation_s = perf_counter() - started - reader_s
        if translated.target_prompt_tokens != 0:
            raise StateError("Composite target prompt ownership violated")
        answer = self.target.session(translated).generate(max_new_tokens=max_new_tokens,
                                                         mtp=mtp, on_token=on_token)
        return replace(answer, stats=replace(answer.stats, reader_prefill_s=reader_s,
                                             translation_s=translation_s,
                                             request_ttft_s=reader_s+translation_s+answer.stats.first_token_s,
                                             total_s=perf_counter()-started))


class CompositeSession:
    def __init__(self, port: Composite, reader: "Session"):
        self.port, self.reader = port, reader
        self.target_session = None
        self._synced = True

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
        native = source.codec.render(messages, generation_prompt=False) if messages else None
        old_ids = old.rendered_token_ids
        if old_ids is None:
            raise StateError("Reader has no committed token stream")
        if native is not None and native.token_ids[:len(old_ids)] == old_ids:
            suffix_ids = native.token_ids[len(old_ids):]
        else:
            # Earlier assistant turns may have been collapsed by the template;
            # use only the last-answer suffix so old live state is retained.
            anchor = (("user", ""),)
            start = source.codec.render(anchor, generation_prompt=True)
            end = source.codec.render(anchor + (("assistant", assistant),), generation_prompt=False)
            if end.token_ids[:len(start.token_ids)] != start.token_ids:
                raise CapabilityError("Reader cannot append the native assistant answer exactly")
            suffix_ids = end.token_ids[len(start.token_ids):]
        next_state = source.append(old, suffix_ids,
                                   phase="turn_ready", messages=messages)
        if next_state.rendered_token_ids != old_ids + suffix_ids:
            raise StateError("Reader turn synchronization changed token IDs")
        self.reader.state = next_state
        self._synced = True
        return perf_counter() - started

    def ask(self, question: str, *, max_new_tokens: int = 512, mtp: bool = False,
            on_token: Callable[[int], None] | None = None) -> Answer:
        started = perf_counter()
        sync_s = self._sync()
        source_state = self.reader.prepare(question)
        reader_s = perf_counter() - started - sync_s
        translated = self.port.bridge.translate(source_state)
        translation_s = perf_counter() - started - sync_s - reader_s
        if translated.target_prompt_tokens != 0:
            raise StateError("Composite target processed prompt tokens")
        target_session = self.port.target.session(translated)
        result = target_session.generate(max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
        self.target_session, self._synced = target_session, False
        return replace(result, stats=replace(result.stats, reader_prefill_s=reader_s,
                                             reader_sync_s=sync_s, translation_s=translation_s,
                                             request_ttft_s=sync_s+reader_s+translation_s+result.stats.first_token_s,
                                             total_s=perf_counter()-started))

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
