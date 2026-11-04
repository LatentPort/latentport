"""Directional, pair-locked complete-state translation."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from .codec import numeric_alignment, require_equivalent_content
from .errors import CompatibilityError, HandoffError, StateError
from .identity import ModelIdentity
from .state import State


class Bridge:
    def __init__(self, *, bundle_root: str | Path, source: "Model", target: "Model", translator=None):
        self.bundle_root = Path(bundle_root).resolve()
        self.source, self.target = source, target
        manifest = json.loads((self.bundle_root / "latentport.json").read_text(encoding="utf-8"))
        if manifest["schema"] != "latentport.bridge-bundle.v1" or manifest["candidate"] != "E119_joint_step0016":
            raise CompatibilityError("Unsupported bridge bundle")
        for side, model in (("source", source), ("target", target)):
            ref = manifest["models"][side]
            if (model.identity.repo_id, model.identity.revision) != (ref["repo_id"], ref["revision"]):
                raise CompatibilityError(f"{side} checkpoint differs from bridge bundle")
        if source.identity.backend != target.identity.backend:
            raise CompatibilityError("Cross-backend state conversion is not implemented")
        self.manifest = manifest
        self._translator = translator

    def prepare(self):
        """Load the selected bridge without translating a prompt."""
        if self._translator is None:
            if self.source.identity.backend == "reference":
                from ._reference.bridge import ReferenceTranslator
                if not (self.bundle_root / "bridge/parent.safetensors").exists():
                    from .hub import bundle_root
                    self.bundle_root = bundle_root(self.manifest["repo_id"], full=True)
                self._translator = ReferenceTranslator(self.bundle_root)
            elif self.source.identity.backend == "gguf":
                self._translator = self.source.backend.translator()
            else:
                raise HandoffError("No complete-state translator for this backend")
        if hasattr(self._translator, "warm"):
            self._translator.warm()

    def _record_pair(self, state: State):
        source_codec = self.source.codec
        target_codec = self.target.codec
        if state.rendered_text is None:
            raise StateError("Bridge needs the exact rendered source stream for numeric alignment")
        if state.origin in ("native_append", "native_generation"):
            # Live state contains exactly these bytes. Whole-history rendering
            # would silently remove old thinking during multi-turn handoffs.
            target_record = target_codec.record(state.rendered_text)
        elif state.phase == "in_answer" and state.messages:
            target_base = target_codec.render(state.messages, generation_prompt=True)
            committed = source_codec.decode(state.generated_token_ids, special=True)
            target_record = target_codec.record(target_base.decoded_text + committed)
            if target_record.decoded_text != target_base.decoded_text + committed:
                raise CompatibilityError("Target tokenizer changed the committed assistant prefix")
        elif state.messages:
            target_record = target_codec.render(state.messages, generation_prompt=state.phase == "answer_ready")
        else:
            target_record = target_codec.record(state.rendered_text)
        try:
            source_record = source_codec.record(state.rendered_text)
        except CompatibilityError:
            source_record = _record_from_ids(source_codec, state.rendered_token_ids or ())
        if tuple(source_record.token_ids) != tuple(state.rendered_token_ids or ()):
            # A live multi-turn stream may differ from whole-history
            # re-tokenization because earlier thinking is retained.
            source_record = _record_from_ids(source_codec, state.rendered_token_ids or ())
        if source_record.sequence_length != state.clock:
            raise StateError("Source state clock and rendered stream disagree")
        if state.phase != "in_answer" and not any(role == "assistant" for role, _ in state.messages):
            require_equivalent_content(source_record, target_record, state.messages)
        return source_record, target_record

    def translate(self, state: State, *, target: "Model | None" = None) -> State:
        state.require_model(self.source.identity)
        destination = target or self.target
        destination.identity.require_same(self.target.identity)
        source_record, target_record = self._record_pair(state)
        metadata = numeric_alignment(source_record, target_record)
        if self._translator is None:
            self.prepare()
        started = perf_counter()
        source_values = state._read_tensors(device=getattr(self.source.backend, "translation_device", None))
        versions = {name: _tensor_version(value) for name, value in source_values.items()}
        values = self._translator.translate(source_values, state.clock, metadata)
        if any(_tensor_version(value) != versions[name]
               for name, value in source_values.items()):
            raise StateError("Translator mutated an immutable source state")
        elapsed = perf_counter() - started
        generated = ()
        if state.phase == "in_answer" and state.generated_token_ids:
            committed = self.source.codec.decode(state.generated_token_ids, special=True)
            generated = self.target.codec.encode(committed)
            if target_record.token_ids[-len(generated):] != generated:
                raise CompatibilityError("Translated assistant suffix has no exact token boundary")
        return State(identity=destination.identity, clock=target_record.sequence_length,
                     phase=state.phase, origin="translated:" + self.manifest["release_id"],
                     target_prompt_tokens=0, rendered_text=target_record.decoded_text,
                     rendered_token_ids=target_record.token_ids, messages=state.messages,
                     generated_token_ids=generated, parent_id=state.state_id,
                     _components=values)


def _tensor_version(value):
    # Torch inference tensors deliberately have no version counter. Native
    # backends and translators obey the readonly borrowed-state contract.
    try:
        return getattr(value, "_version", None)
    except RuntimeError:
        return None


def _record_from_ids(codec, ids):
    from .codec import BYTE_DECODER, TokenRecord
    tokens = codec.tokenizer.convert_ids_to_tokens(list(ids))
    specials = set(codec.tokenizer.all_special_ids)
    pieces = [token.encode() if ident in specials else bytes(BYTE_DECODER[c] for c in token)
              for ident, token in zip(ids, tokens)]
    text = codec.decode(ids, special=True)
    if b"".join(pieces) != text.encode():
        raise CompatibilityError("Source token bytes do not round-trip")
    spans, cursor = [], 0
    for piece in pieces:
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return TokenRecord(text, text, tuple(ids), tuple(spans))


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import Model
