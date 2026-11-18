"""Incremental ingestion over a resident cache and immutable State boundaries."""
from dataclasses import replace

from .errors import CapabilityError, StateError
from .state import INPUT_EMBEDDINGS
from ._tokens import token_ids

TAIL_TOKENS = 32


class ResidentCache:
    """Append fresh tokens without retaining or rendering the historical prompt.

    Private ingestion adapter owned by Session. Snapshots remain immutable
    and contain enough tail metadata to ask a new question. Session owns
    transformations, model changes and failure recovery.
    """
    def __init__(self, model, state=None, *, retain_replay=False):
        if not hasattr(model.backend, "live"):
            raise CapabilityError("This backend does not support resident ingestion")
        if state is not None:
            state.require_model(model.identity)
        self.model = model
        self._live = model.backend.live(model.identity, state)
        self._tail = (state.stream_tail_ids or (state.rendered_token_ids or ())[-TAIL_TOKENS:]) if state else ()
        self._phase = state.phase if state else "context_ready"
        self._prompt_tokens = state.target_prompt_tokens if state else 0
        self._generated = state.generated_token_ids if state else ()
        self._embedding_blocks = None
        if model._embedding_cache is not None:
            prefix = state._input_embeddings() if state is not None else None
            if state is None or prefix is not None:
                self._embedding_blocks = [] if prefix is None else [prefix]
        self._replay_ids = [] if retain_replay else None
        if self._replay_ids is not None and state is not None:
            history = state.rendered_token_ids
            self._replay_ids = list(history) if history is not None and len(history) == state.clock else None
        self.closed = False

    @property
    def clock(self):
        return self._live.clock

    @property
    def physical(self):
        return self._live.physical

    def _check_open(self):
        if self.closed:
            raise StateError("Live session is closed or failed; restore an immutable snapshot")
        self.model._check_open()

    def append(self, text):
        self._check_open()
        ids = self.model.codec.encode(text) if isinstance(text, str) else token_ids(text)
        if not ids:
            raise ValueError("append requires one or more tokens")
        check = getattr(self.model.backend, "check_capacity", None)
        if check is not None:
            check(self.clock, self.physical, len(ids))
        try:
            if self._embedding_blocks is not None:
                embeddings = self.model._embedding_cache.capture(ids)
            for start in range(0, len(ids), 1024):
                self._live.append(ids[start:start + 1024])
        except BaseException:
            self.close()
            raise
        if self._embedding_blocks is not None:
            self._embedding_blocks.append(embeddings)
        self._tail = (self._tail + ids)[-TAIL_TOKENS:]
        if self._replay_ids is not None:
            self._replay_ids.extend(ids)
        if self.model.role == "target" and self._phase != "in_answer":
            self._prompt_tokens += len(ids)
        return self.clock

    def _decorate(self, state):
        values = state._components
        if self._embedding_blocks:
            import torch

            embeddings = (self._embedding_blocks[0] if len(self._embedding_blocks) == 1
                          else torch.cat(self._embedding_blocks))
            values = {**values, INPUT_EMBEDDINGS: embeddings}
        return replace(state, stream_tail_ids=self._tail, phase=self._phase,
                       rendered_token_ids=tuple(self._replay_ids) if self._replay_ids is not None else None,
                       messages=(), messages_complete=False,
                       target_prompt_tokens=self._prompt_tokens, generated_token_ids=self._generated,
                       _components=values)

    def snapshot(self, *, release=False):
        self._check_open()
        state = self._live.snapshot(release=release)
        if release:
            self.closed = True
        try:
            return self._decorate(state)
        finally:
            if release:
                self._embedding_blocks = None

    def close(self):
        self._live.close()
        self._embedding_blocks = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
