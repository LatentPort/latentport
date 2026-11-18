"""One model's capture, sequential append and session operations."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .errors import CompatibilityError, StateError
from .identity import ModelIdentity
from .state import State
from ._tokens import token_ids


def _messages(value):
    if isinstance(value, str):
        return None
    rows = []
    for item in value:
        role, content = (item["role"], item["content"]) if isinstance(item, dict) else item
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("Messages need string roles and string contents")
        rows.append((role, content))
    if not rows:
        raise ValueError("At least one message is required")
    return tuple(rows)


class Model:
    def __init__(self, identity: ModelIdentity, backend, *, role: str):
        if role not in ("source", "target") or identity.backend != backend.name or identity.state_abi != backend.state_abi:
            raise CompatibilityError("Model/backend role or state ABI mismatch")
        self.identity, self.backend, self.role = identity, backend, role
        self.closed = False
        self._owner = None

    def _check_open(self):
        if self.closed or (self._owner is not None and self._owner.closed):
            raise StateError("Model is closed")

    @property
    def codec(self):
        self._check_open()
        return self.backend.codec(self.identity)

    def capture(self, context: str | Sequence[tuple[str, str]], *, generation_prompt: bool = False) -> State:
        rows = _messages(context)
        record = self.codec.record(context) if rows is None else self.codec.render(rows, generation_prompt=generation_prompt)
        values = self.backend.capture(self.identity, record.token_ids)
        return State(identity=self.identity, clock=record.sequence_length,
                     phase="answer_ready" if generation_prompt else "context_ready",
                     origin="native_capture", target_prompt_tokens=record.sequence_length if self.role == "target" else 0,
                     rendered_text=record.decoded_text, rendered_token_ids=record.token_ids,
                     stream_tail_ids=record.token_ids[-32:],
                     messages=rows or (), messages_complete=bool(rows), _components=values)

    def append(self, state: State, text: str | Sequence[int], *, phase: str | None = None,
               messages: Sequence[tuple[str, str]] | None = None) -> State:
        self._check_open()
        state.require_model(self.identity)
        ids = self.codec.encode(text) if isinstance(text, str) else token_ids(text)
        if not ids:
            raise ValueError("append requires one or more tokens")
        ids = token_ids(ids, vocabulary_size=getattr(self.codec, "vocabulary_size", None))
        check = getattr(self.backend, "check_capacity", None)
        if check is not None:
            check(state.clock, state.physical_kv_length, len(ids))
        working = (state._read_tensors() if getattr(self.backend, "copy_on_write", False)
                   else state._tensors(device=getattr(self.backend, "device", None)))
        values = self.backend.advance(self.identity, working,
                                      state.clock, ids)
        full_ids = (state.rendered_token_ids or ()) + ids if state.rendered_token_ids is not None else None
        rendered = self.codec.decode(full_ids, special=True) if full_ids is not None else None
        return State(identity=self.identity, clock=state.clock + len(ids), phase=phase or state.phase,
                     origin="native_append", target_prompt_tokens=state.target_prompt_tokens +
                     (len(ids) if self.role == "target" and state.phase != "in_answer" else 0),
                     rendered_text=rendered, rendered_token_ids=full_ids,
                     stream_tail_ids=((state.stream_tail_ids or (state.rendered_token_ids or ())[-32:]) + ids)[-32:],
                     messages=tuple(messages) if messages is not None else state.messages,
                     messages_complete=state.messages_complete and (messages is not None or not state.messages),
                     generated_token_ids=state.generated_token_ids,
                     parent_id=state.state_id, _components=values)

    def combine(self, left: State, right: State, *, method: str = "auto") -> State:
        """Continue left with the ordered history represented by right."""
        from ._combine import combine
        return combine(self, left, right, method=method)

    def session(self, state: State | None = None, *, resident: bool = False,
                retain_replay: bool = False) -> "Session":
        self._check_open()
        if state is not None:
            state.require_model(self.identity)
        from .session import Session
        return Session(self, state, resident=resident, retain_replay=retain_replay)

    def live(self, state: State | None = None, *, retain_replay: bool = False):
        """Keep a resident cache and optionally retain token IDs for combination."""
        return self.session(state, resident=True, retain_replay=retain_replay)

    def prefill(self, context: str | Sequence[tuple[str, str]]) -> State:
        return self.capture(context, generation_prompt=False)

    def close(self):
        """Close this view; only standalone models own their backend."""
        if self.closed:
            return
        self.closed = True
        if self._owner is None and hasattr(self.backend, "close"):
            self.backend.close()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .session import Session
