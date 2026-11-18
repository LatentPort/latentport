"""Complete Gemma 4 global and sliding KV state for standalone BF16 text."""

import re

import torch
from transformers import AutoTokenizer, DynamicCache

from .._models import GEMMA_ABI
from ..codec import Codec, TokenRecord
from ..errors import CapabilityError, CompatibilityError, StateError
from ..state import State
from .bf16 import BF16Backend
from .gemma_decoder import Decoder
from .live import LiveCache, LiveDecode
from .native import KERNEL_LOCK


class GemmaCodec(Codec):
    initial_bos_id = 2
    turn_end_text = "<turn|>"

    @property
    def chat_eos_id(self):
        return self.tokenizer.convert_tokens_to_ids("<turn|>")

    def record(self, text):
        ids = self.encode(text)
        if not ids:
            raise ValueError("Cannot capture an empty token sequence")
        special = set(self.tokenizer.all_special_ids)
        spans = []
        data = bytearray()
        for idx, piece in zip(ids, self.tokenizer.convert_ids_to_tokens(list(ids))):
            start = len(data)
            if idx not in special and re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
                data.append(int(piece[3:5], 16))
            else:
                data.extend((piece if idx in special else piece.replace("▁", " ")).encode())
            spans.append((start, len(data)))
        decoded = self.decode(ids, special=True)
        if data != decoded.encode():
            raise CompatibilityError("Gemma tokenizer byte spans do not round-trip")
        return TokenRecord(text, decoded, ids, tuple(spans))


class GemmaLive(LiveCache):
    new_cache = staticmethod(lambda config: DynamicCache(config=config))

    @property
    def physical(self):
        return self.cache.get_seq_length(5)

    @torch.inference_mode()
    def _append(self, ids):
        if self.failed or self.cache is None or not ids:
            raise StateError("Gemma append requires an open cache and nonempty input")
        self.backend.options.check(self.clock, self.physical, len(ids))
        try:
            self.terminal = self.model.step(ids, self.cache, self.clock).contiguous()
            if self.positions is not None:
                more = torch.arange(self.clock, self.clock + len(ids), device=self.positions.device)
                self.positions = torch.cat((self.positions, more))
            self.clock += len(ids)
            return self.model.logits(self.terminal)[:, -1]
        except BaseException:
            self.failed = True
            raise


class GemmaDecode(LiveDecode):
    @torch.inference_mode()
    def __init__(self, live):
        self.live = live
        self.logits = live.model.logits(live.terminal)[:, -1]


class GemmaBackend(BF16Backend):
    state_abi = GEMMA_ABI
    copy_on_write = True

    def _model(self, identity):
        side = self._side(identity)
        with KERNEL_LOCK:
            if side not in self._models:
                self._models[side] = Decoder(self._path(side), self.device, options=self.options)
        return self._models[side]

    def codec(self, identity):
        side = self._side(identity)
        if side not in self._codecs:
            tok = AutoTokenizer.from_pretrained(
                self._path(side, metadata_only=True), local_files_only=True
            )
            self._codecs[side] = GemmaCodec(tok, self.specs[side].get("template_kwargs"))
        return self._codecs[side]

    def _extract(self, model, cache, terminal):
        values = {"H": terminal}
        for i, kind in enumerate(model.cfg.layer_types):
            prefix = "S" if kind == "sliding_attention" else ""
            values[f"{prefix}K.{i}"] = cache.layers[i].keys
            values[f"{prefix}V.{i}"] = cache.layers[i].values
        return values

    def _cache(self, model, values, clock):
        cache = DynamicCache(config=model.cfg)
        for i, kind in enumerate(model.cfg.layer_types):
            sliding = kind == "sliding_attention"
            prefix = "S" if sliding else ""
            layer = cache.layers[i]
            layer.keys, layer.values = values[f"{prefix}K.{i}"], values[f"{prefix}V.{i}"]
            layer.dtype, layer.device, layer.is_initialized = layer.keys.dtype, layer.keys.device, True
            if sliding:
                layer.cumulative_length = clock
        return cache

    def live(self, identity, state=None):
        return GemmaLive(self, identity, state)

    def _run(self, identity, values, clock, ids):
        parent = (
            State(identity, clock, "context_ready", "install", _components=values)
            if values is not None
            else None
        )
        with GemmaLive(self, identity, parent, _owned=True) as live:
            live.append(ids)
            return live.snapshot(release=True)._read_tensors()

    def eos_ids(self, identity):
        return frozenset((1, 106))

    @torch.inference_mode()
    def first_logits(self, identity, components, clock):
        self.validate(components, clock, self._side(identity))
        return self._model(identity).logits(components["H"])[:, -1]

    def begin_decode(self, identity, components, clock, *, mtp=False):
        if mtp:
            raise CapabilityError("Gemma text execution disables MTP")
        parent = State(identity, clock, "answer_ready", "decode", _components=components)
        return GemmaDecode(GemmaLive(self, identity, parent, _owned=True))
