"""Incremental ingestion over a resident cache and immutable State boundaries."""
from dataclasses import replace

from .context import ContextOptions
from .errors import CapabilityError, StateError

TAIL_TOKENS = 32


class LiveSession:
    """Append fresh tokens without retaining or rendering the historical prompt.

    ``compact`` delegates to the same ``State.compact`` operation. Snapshots
    remain immutable, and contain enough tail metadata to ask a new question.
    """
    def __init__(self, model, state=None):
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

    def append(self, text):
        self._check_open()
        ids = self.model.codec.encode(text) if isinstance(text, str) else tuple(int(x) for x in text)
        if not ids:
            raise ValueError("append requires one or more tokens")
        options = getattr(self.model.backend, "options", None)
        context = options.context_options if hasattr(options, "context_options") else options or ContextOptions()
        context.check(self.clock, self.physical, len(ids))
        try:
            for start in range(0, len(ids), 1024):
                self._live.append(ids[start:start + 1024])
        except BaseException:
            self.close()
            raise
        self._tail = (self._tail + ids)[-TAIL_TOKENS:]
        if self.model.role == "target" and self._phase != "in_answer":
            self._prompt_tokens += len(ids)
        return self.clock

    def _decorate(self, state):
        return replace(state, stream_tail_ids=self._tail, phase=self._phase,
                       target_prompt_tokens=self._prompt_tokens, generated_token_ids=self._generated)

    def snapshot(self, *, release=False):
        self._check_open()
        state = self._live.snapshot(release=release)
        if release:
            self.closed = True
        return self._decorate(state)

    def compact(self, *, ratio=8, force=False):
        self._check_open()
        # Backend adapters freeze, call State.compact, then install a private
        # working cache. There is no second selector or eviction policy here.
        state = self._live.compact(ratio=ratio, force=force)
        return self._decorate(state)

    def close(self):
        self._live.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
