"""Mutable execution policy over immutable, forkable state snapshots."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable

from .errors import CapabilityError, CompatibilityError, HandoffError, StateError
from .identity import Answer, RunStats
from .state import State


class Session:
    def __init__(self, model: "Model", state: State | None = None, *, resident: bool = False):
        self._resident = None
        self.resident = resident
        self.closed = False
        if resident and not hasattr(model.backend, "live"):
            raise CapabilityError("This backend does not support resident execution")
        self.model, self.state = model, state
        self._pending_decode = None
        self.last_handoff_s = 0.0
        if state is not None:
            state.require_model(model.identity)

    @property
    def state(self):
        return self._resident.snapshot() if self._resident is not None else self._state

    @state.setter
    def state(self, value):
        # Replacing a checkpoint must also replace any mutable working handle.
        # Validate first so a rejected assignment leaves the session untouched.
        self._check_open()
        if value is not None:
            value.require_model(self.model.identity)
        if self._resident is not None:
            self._resident.close()
            self._resident = None
        pending = getattr(self, "_pending_decode", None)
        if pending is not None and hasattr(pending, "close"):
            pending.close()
        self._pending_decode = None
        self._state = value

    def _check_open(self):
        if self.closed:
            raise StateError("Session is closed")

    def _freeze(self):
        self._check_open()
        if self._resident is not None:
            self._state = self._resident.snapshot(release=True)
            self._resident = None
        if self._pending_decode is not None:
            if hasattr(self._pending_decode, "close"):
                self._pending_decode.close()
            self._pending_decode = None
        return self._state

    @property
    def clock(self):
        return self._resident.clock if self._resident is not None else self._state.clock if self._state else 0

    @property
    def physical(self):
        return self._resident.physical if self._resident is not None else self._state.physical_kv_length if self._state else 0

    def append(self, text):
        self._check_open()
        if self.resident:
            if self._resident is None:
                self._freeze()
                from .stream import ResidentCache
                self._resident = ResidentCache(self.model, self._state)
                self._state = None
            try:
                return self._resident.append(text)
            except BaseException:
                if self._resident.closed:
                    self.close()
                raise
        self._freeze()
        if self._state is None:
            if not isinstance(text, str):
                raise ValueError("An initial nonresident append requires text; use resident=True for token IDs")
            self._state = self.model.prefill(text)
        else:
            self._state = self.model.append(self._state, text)
        return self.clock

    def snapshot(self, *, release=False):
        self._check_open()
        state = self._freeze() if release else self.state
        if state is None:
            raise StateError("Capture state before snapshotting")
        if release:
            self.close()
        return state

    def save(self, path):
        return self.snapshot().save(path)

    def _transform(self, operation, *, target=None, decode=False):
        """Freeze, transform and prepare installation before committing ownership.

        A failure retains the exact source snapshot. Resident execution resumes
        lazily from it, so recovery does not require a second allocation to work.
        """
        from ._transaction import transform_install
        parent = self._freeze()
        if parent is None:
            raise StateError("Capture state before transforming")
        destination = target or self.model
        if self.resident and not hasattr(destination.backend, "live"):
            raise CapabilityError("Destination backend does not support resident execution")
        def install(child):
            if self.resident:
                from .stream import ResidentCache
                return ResidentCache(destination, child)
            if decode:
                values = (child._read_tensors() if getattr(destination.backend, "copy_on_write", False)
                          else child._tensors(device=getattr(destination.backend, "device", None)))
                return destination.backend.begin_decode(destination.identity, values, child.clock)
            destination.backend.install(destination.identity, child._read_tensors(), child.clock)
        child, handle = transform_install(parent, operation, destination.identity, install)
        self.model, self._state = destination, child
        if self.resident:
            self._resident = handle
            self._state = None
        else:
            self._pending_decode = handle
        return child

    def compact(self, *, ratio=8, force=False):
        return self._transform(lambda state: state.compact(ratio=ratio, force=force))

    def close(self):
        if self._resident is not None:
            self._resident.close()
            self._resident = None
        if self._pending_decode is not None and hasattr(self._pending_decode, "close"):
            self._pending_decode.close()
        self._pending_decode = None
        self.closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def fork(self, count: int = 1) -> "Session | tuple[Session, ...]":
        self._check_open()
        parent = self.snapshot()
        states = parent.fork(count)
        if count == 1:
            return Session(self.model, states, resident=self.resident)
        return tuple(Session(self.model, state, resident=self.resident) for state in states)

    def prepare(self, question: str) -> State:
        """Consume a new user turn and native assistant prefix, without generating."""
        self._freeze()
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
            if state.rendered_text is None and not state.stream_tail_ids:
                raise StateError("Raw state has no exact rendered stream")
            tail = state.rendered_text if state.rendered_text is not None else self.model.codec.decode(state.stream_tail_ids, special=True)
            suffix = self.model.codec.render((("user", question),), generation_prompt=True)
            separator = self.model.codec.encode("\n") if not tail.endswith("\n") else ()
            next_state = self.model.append(state, separator + suffix.token_ids, phase="answer_ready")
            self.state = replace(next_state, generated_token_ids=())
            self._pending_decode = None
            return self.state
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
        self.state = replace(next_state, generated_token_ids=())
        self._pending_decode = None
        return self.state

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
        self._check_open()
        if self._resident is not None:
            self._freeze()
        from contextlib import nullcontext
        started = perf_counter()
        guard = (self.model.backend.generation_guard() if hasattr(self.model.backend, "generation_guard")
                 else nullcontext())
        with guard:
            queue = perf_counter() - started
            answer = self._generate(max_new_tokens=max_new_tokens, mtp=mtp, on_token=on_token)
            cleanup_started = perf_counter()
            if hasattr(self.model.backend, "finish_request"):
                self.model.backend.finish_request()
            cleanup_s = perf_counter() - cleanup_started
        return replace(answer, stats=replace(answer.stats, queue_s=queue,
                       first_token_s=answer.stats.first_token_s+queue,
                       request_ttft_s=answer.stats.request_ttft_s+queue,
                       generation_s=answer.stats.generation_s+cleanup_s,
                       total_s=answer.stats.total_s+queue+cleanup_s))

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
                         stream_tail_ids=((initial.stream_tail_ids or (initial.rendered_token_ids or ())[-32:]) + tuple(tokens))[-32:],
                        parent_id=initial.state_id, _components=snapshot)
        if finished and current.messages and tokens[-1] == model.codec.chat_eos_id:
            # The committed stream is authoritative. Re-rendering old turns
            # can strip thinking and cannot reconstruct a resumed live answer.
            assistant = model.codec.decode(current.generated_token_ids[:-1], special=True)
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
        self._freeze()
        if self.state is None:
            raise StateError("No state to escalate")
        if self.state.phase not in ("answer_ready", "in_answer", "context_ready"):
            raise StateError("Escalation requires a live context or answer boundary")
        if bridge.source.identity != self.model.identity or bridge.target.identity != target.identity:
            raise CompatibilityError("Bridge direction or exact model identity differs")
        started = perf_counter()
        translated = self._transform(lambda state: bridge.translate(state, target=target),
                                     target=target, decode=True)
        self.last_handoff_s = perf_counter() - started
        return translated


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import Model
    from .bridge import Bridge
