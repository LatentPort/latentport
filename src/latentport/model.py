"""One model's capture, sequential append and session operations."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .errors import CompatibilityError, StateError
from .identity import ModelIdentity
from .state import State


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

    @property
    def codec(self):
        return self.backend.codec(self.identity)

    def capture(self, context: str | Sequence[tuple[str, str]], *, generation_prompt: bool = False) -> State:
        rows = _messages(context)
        record = self.codec.record(context) if rows is None else self.codec.render(rows, generation_prompt=generation_prompt)
        values = self.backend.capture(self.identity, record.token_ids)
        return State(identity=self.identity, clock=record.sequence_length,
                     phase="answer_ready" if generation_prompt else "context_ready",
                     origin="native_capture", target_prompt_tokens=record.sequence_length if self.role == "target" else 0,
                     rendered_text=record.decoded_text, rendered_token_ids=record.token_ids,
                     messages=rows or (), _components=values)

    def append(self, state: State, text: str | Sequence[int], *, phase: str | None = None,
               messages: Sequence[tuple[str, str]] | None = None) -> State:
        state.require_model(self.identity)
        ids = self.codec.encode(text) if isinstance(text, str) else tuple(int(x) for x in text)
        if not ids:
            raise ValueError("append requires one or more tokens")
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
                     messages=tuple(messages) if messages is not None else state.messages,
                     generated_token_ids=state.generated_token_ids,
                     parent_id=state.state_id, _components=values)

    def session(self, state: State | None = None) -> "Session":
        if state is not None:
            state.require_model(self.identity)
        from .session import Session
        return Session(self, state)

    def prefill(self, context: str | Sequence[tuple[str, str]]) -> State:
        return self.capture(context, generation_prompt=False)

    def close(self):
        """Release this model's backend; independently owned snapshots survive."""
        if hasattr(self.backend, "close"):
            self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .session import Session
