"""Exact tokenizer streams and numeric byte alignment for the supported pair."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

from .errors import CompatibilityError


def _byte_decoder() -> dict[str, int]:
    values = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): value for value, char in zip(values, chars)}


BYTE_DECODER = _byte_decoder()


@dataclass(frozen=True, slots=True)
class TokenRecord:
    rendered_text: str
    decoded_text: str
    token_ids: tuple[int, ...]
    token_byte_spans: tuple[tuple[int, int], ...]

    @property
    def sequence_length(self) -> int:
        return len(self.token_ids)


class Codec:
    def __init__(self, tokenizer: Any, template_kwargs: Mapping[str, Any] | None = None):
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})

    @property
    def chat_eos_id(self) -> int:
        return int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))

    def render(self, messages: Sequence[tuple[str, str]], *, generation_prompt: bool) -> TokenRecord:
        rows = [{"role": role, "content": content} for role, content in messages]
        text = self.tokenizer.apply_chat_template(rows, tokenize=False,
                                                  add_generation_prompt=generation_prompt,
                                                  **self.template_kwargs)
        native = self.tokenizer.apply_chat_template(rows, tokenize=True,
                                                    add_generation_prompt=generation_prompt,
                                                    return_dict=True, **self.template_kwargs)["input_ids"]
        record = self.record(text)
        if list(record.token_ids) != list(native):
            raise CompatibilityError("Native chat template and tokenizer disagree")
        return record

    def record(self, text: str) -> TokenRecord:
        ids = tuple(int(x) for x in self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if not ids:
            raise ValueError("Cannot capture an empty token sequence")
        decoded = self.decode(ids, special=True)
        normalizer = getattr(getattr(self.tokenizer, "backend_tokenizer", None), "normalizer", None)
        normalized = normalizer.normalize_str(text) if normalizer else text
        if decoded != normalized:
            raise CompatibilityError("Tokenizer normalization does not reproduce the rendered stream")
        special_ids = set(self.tokenizer.all_special_ids)
        pieces = []
        for ident, token in zip(ids, self.tokenizer.convert_ids_to_tokens(list(ids))):
            pieces.append(token.encode() if ident in special_ids else bytes(BYTE_DECODER[c] for c in token))
        if b"".join(pieces) != decoded.encode():
            raise CompatibilityError("Tokenizer byte spans do not round-trip")
        spans = []
        cursor = 0
        for piece in pieces:
            spans.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        return TokenRecord(text, decoded, ids, tuple(spans))

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(int(x) for x in self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, token_ids: Sequence[int], *, special: bool = False) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=not special,
                                     clean_up_tokenization_spaces=False)


def _byte_operations(a: bytes, b: bytes) -> list[tuple[str, int, int, int, int]]:
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(a), len(b)) - prefix and a[-suffix - 1] == b[-suffix - 1]:
        suffix += 1
    ae, be = len(a) - suffix, len(b) - suffix
    if max(ae - prefix, be - prefix) > 16384:
        raise CompatibilityError("Changed template byte region exceeds the supported alignment bound")
    ops = [("equal", 0, prefix, 0, prefix)] if prefix else []
    for tag, i, j, k, l in SequenceMatcher(None, a[prefix:ae], b[prefix:be], autojunk=False).get_opcodes():
        ops.append((tag, i + prefix, j + prefix, k + prefix, l + prefix))
    if suffix:
        ops.append(("equal", ae, len(a), be, len(b)))
    return ops


def numeric_alignment(source: TokenRecord, target: TokenRecord) -> dict[str, Any]:
    """Produce one monotone source coordinate per native target token."""
    a, b = source.decoded_text.encode(), target.decoded_text.encode()
    n, m = source.sequence_length, target.sequence_length
    if n <= 0 or m <= 0:
        raise CompatibilityError("Both tokenizer streams must be nonempty")
    midpoints = [(x + y) / 2 for x, y in source.token_byte_spans]
    nonempty = [op for op in _byte_operations(a, b) if op[4] > op[3]]
    if not nonempty:
        raise CompatibilityError("No target byte spans were mapped")
    ends = [op[4] for op in nonempty]
    centers = []
    for start, end in target.token_byte_spans:
        point = (start + end) / 2
        _, i, j, k, l = nonempty[min(bisect_right(ends, point), len(nonempty) - 1)]
        mapped = i + (point - k) * (j - i) / (l - k)
        right = bisect_right(midpoints, mapped)
        if right == 0:
            center = 0.0
        elif right == n:
            center = float(n - 1)
        else:
            center = right - 1 + (mapped - midpoints[right - 1]) / (midpoints[right] - midpoints[right - 1])
        centers.append(max(centers[-1] if centers else 0.0, min(float(n - 1), center)))
    return {"source_length": n, "target_length": m,
            "centers": centers, "target_positions": list(range(m))}


def require_equivalent_content(source: TokenRecord, target: TokenRecord,
                               messages: Sequence[tuple[str, str]]) -> None:
    if source.decoded_text == target.decoded_text:
        return
    empty = "<|im_start|>system\n<|im_end|>\n"
    if messages and messages[0] == ("system", "") and source.decoded_text.startswith(empty):
        if source.decoded_text[len(empty):] == target.decoded_text:
            return
    raise CompatibilityError("Source and target templates consume different content")
