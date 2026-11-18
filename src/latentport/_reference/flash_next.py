"""Indexed sparse attention and complete-state execution for Flash-Next."""

from contextlib import contextmanager, nullcontext
import math
from types import MethodType
from unittest.mock import patch

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from .._models import FLASH_ABI
from ..errors import StateError, CapabilityError
from ..state import State
from . import cache_abi
from .bf16 import BF16Backend
from .live import LiveCache, LiveDecode
from .native import KERNEL_LOCK
from .rotary import configure_global_yarn
from .flash_weights import StreamExperts, load_weights


def index_mask(indexer, hidden, position_embeddings, attention_mask, past_key_values):
    # All callers here have one unpadded sequence, a causal prefix and contiguous suffix.
    count = hidden.shape[1]
    ratio = indexer.compress_ratio
    cos, sin = position_embeddings
    q, keys = indexer.index_qk_proj(hidden).split(
        [indexer.index_n_heads * indexer.index_head_dim, indexer.index_head_dim], -1
    )
    q = indexer.q_layernorm(q.reshape(1, count, indexer.index_n_heads, -1))
    q = qwen.apply_rotary_pos_emb(q, cos=cos[:, -count:], sin=sin[:, -count:], unsqueeze_dim=2)
    keys = past_key_values.update_indexer(keys, indexer.layer_idx)
    length = keys.size(1)
    if length <= indexer.token_budget:
        return attention_mask
    blocks = length // ratio
    pooled = keys[0, : blocks * ratio].reshape(blocks, ratio, -1).float().mean(1).to(keys.dtype)
    pooled = indexer.k_layernorm(pooled)
    pooled = qwen.apply_rotary_pos_emb(
        pooled[:, None], cos=cos[0, : blocks * ratio : ratio], sin=sin[0, : blocks * ratio : ratio]
    ).squeeze(1)
    visible = torch.arange(length - count + 1, length + 1, device=hidden.device)
    nblocks = visible // ratio
    scores = (q[0].float() @ pooled.float().T).relu_().sum(1) / math.sqrt(indexer.index_head_dim)
    block_ids = torch.arange(blocks, device=hidden.device)
    scores.masked_fill_(block_ids[None] >= nblocks[:, None], -torch.inf)
    # Include all visible complete blocks when fewer than the routing budget exist.
    selected = scores.topk(min(blocks, indexer.block_topk), dim=-1).indices
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, selected, True)
    keep &= block_ids[None] < nblocks[:, None]
    physical = torch.arange(length, device=hidden.device)
    mask = (physical[None] >= nblocks[:, None] * ratio) & (physical[None] < visible[:, None])
    mask[:, : blocks * ratio] |= keep.repeat_interleave(ratio, dim=1)
    return mask[None, None]


@contextmanager
def resident_experts(layer, config):
    original = layer.mlp.experts
    if not isinstance(original, StreamExperts) or original.quantization is not None:
        yield
        return
    with torch.device("meta"):
        module = qwen.Qwen4ExpTextExperts(config)
    module.gate_up_proj = nn.Parameter(original.gate.to(layer.mlp.gate.weight.device), requires_grad=False)
    module.down_proj = nn.Parameter(original.down.to(layer.mlp.gate.weight.device), requires_grad=False)
    layer.mlp.experts = module
    try:
        yield
    finally:
        layer.mlp.experts = original


def configure(model):
    model.config._experts_implementation = "grouped_mm"
    for layer in model.model.layers:
        if hasattr(layer, "self_attn"):
            indexer = layer.self_attn.indexer
            indexer.forward = MethodType(index_mask, indexer)


def append_layer(layer, hidden, ids, cache, rope, prior):
    n = ids.size(-1)
    visible = (
        torch.arange(prior + n, device=ids.device)[None, :]
        <= torch.arange(prior, prior + n, device=ids.device)[:, None]
    )
    return layer(
        hidden,
        position_embeddings=rope,
        attention_mask=visible[None, None],
        conv_mask=None,
        past_key_values=cache,
        ple_input_ids=ids,
    )


class FlashLiveCache(LiveCache):
    @staticmethod
    def new_cache(config):
        return DynamicCache(config=config)

    @torch.inference_mode()
    def _append(self, ids):
        if self.failed or self.cache is None or not ids:
            raise StateError("Flash-Next append requires an open cache and nonempty input")
        self.backend.options.check(self.clock, self.physical, len(ids))
        tokens = torch.as_tensor(ids, device=self.backend.device).reshape(1, -1)
        next_positions = torch.arange(self.clock, self.clock + len(ids), device=tokens.device)
        previous = getattr(self.cache, "position_ids", None)
        logical = next_positions.reshape(1, 1, -1).expand(3, 1, -1)
        positions = torch.cat((previous, logical), dim=-1) if previous is not None else logical
        prior = self.physical
        try:
            with self.backend.kernels():
                text = self.model.model
                hidden = text.embed_tokens(tokens).repeat(1, 1, text.config.hc_count)
                cos, sin = text.rotary_emb(hidden, positions)
                for layer in text.layers:
                    # Short decode steps transfer only the experts selected by the router.
                    transfer = resident_experts(layer, text.config) if len(ids) >= 128 else nullcontext()
                    with transfer:
                        for start in range(0, len(ids), 1024):
                            end = min(start + 1024, len(ids))
                            rope = cos[:, : prior + end], sin[:, : prior + end]
                            hidden[:, start:end] = append_layer(
                                layer,
                                hidden[:, start:end],
                                tokens[:, start:end],
                                self.cache,
                                rope,
                                prior + start,
                            )
                self.terminal = text.hyper_connection_mixer(hidden[:, -1:]).contiguous()
                self.cache.position_ids = positions
                if self.positions is not None:
                    self.positions = positions[0, 0]
                self.clock += len(ids)
                return self.model.lm_head(self.terminal)[:, -1]
        except BaseException:
            self.failed = True
            raise


class FlashNextBackend(BF16Backend):
    state_abi = FLASH_ABI

    @staticmethod
    @contextmanager
    def kernels():
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

        with (
            patch.object(qwen, "torch_chunk_gated_delta_rule", chunk_gated_delta_rule),
            patch.object(qwen, "torch_recurrent_gated_delta_rule", fused_recurrent_gated_delta_rule),
        ):
            yield

    def _model(self, identity):
        side = self._side(identity)
        with KERNEL_LOCK, torch.cuda.device(self.device):
            if side not in self._models:
                model = load_weights(self._path(side), nf4=self.name == "nf4")
                configure_global_yarn(model.model.rotary_emb, self.options)
                configure(model)
                model.generation_config.eos_token_id = [248044, 248046]
                self._models[side] = model
        return self._models[side]

    def _extract(self, model, cache, terminal):
        values = {"H": terminal}
        for i, layer in enumerate(cache.layers):
            if hasattr(layer, "indexer_keys"):
                values[f"K.{i}"], values[f"V.{i}"], values[f"I.{i}"] = (
                    layer.keys,
                    layer.values,
                    layer.indexer_keys,
                )
            else:
                values[f"C.{i}"], values[f"R.{i}"] = layer.conv_states[0], layer.recurrent_states[0]
        values["PLE"], values["NGRAM"] = cache.layers[1].conv_states[1], cache.layers[1].conv_states[2]
        return values

    def _cache(self, model, values, clock):
        cache = DynamicCache(config=model.config)
        for i, layer in enumerate(cache.layers):
            if hasattr(layer, "indexer_keys"):
                cache_abi.assign(cache, i, "full_attention", values[f"K.{i}"], values[f"V.{i}"])
                layer.indexer_keys = values[f"I.{i}"]
                layer.indexer_dtype, layer.indexer_device = (
                    layer.indexer_keys.dtype,
                    layer.indexer_keys.device,
                )
                layer.is_indexer_initialized = True
            else:
                cache_abi.assign(cache, i, "linear_attention", values[f"C.{i}"], values[f"R.{i}"])
        ple = cache.layers[1]
        for j, field in ((1, "PLE"), (2, "NGRAM")):
            ple.conv_states[j] = values[field]
            ple.conv_kernel_size[j] = values[field].shape[-1]
            ple.has_previous_state[j] = ple.is_conv_states_initialized[j] = True
        positions = values.get("P")
        if positions is None:
            positions = torch.arange(clock, device=values["H"].device)
        cache.position_ids = positions.reshape(1, 1, -1).expand(3, 1, -1)
        return cache

    def live(self, identity, state=None):
        return FlashLiveCache(self, identity, state)

    def _run(self, identity, values, clock, ids):
        parent = (
            State(identity, clock, "context_ready", "install", _components=values)
            if values is not None
            else None
        )
        with FlashLiveCache(self, identity, parent, _owned=True) as live:
            live.append(ids)
            return live.snapshot(release=True)._read_tensors()

    @torch.inference_mode()
    def first_logits(self, identity, components, clock):
        self.validate(components, clock, self._side(identity))
        return self._model(identity).lm_head(components["H"])[:, -1]

    def begin_decode(self, identity, components, clock, *, mtp=False):
        if mtp:
            raise CapabilityError("Standalone Transformers execution disables MTP")
        parent = State(identity, clock, "answer_ready", "decode", _components=components)
        return FlashDecode(FlashLiveCache(self, identity, parent, _owned=True))


class FlashDecode(LiveDecode):
    @torch.inference_mode()
    def __init__(self, live):
        self.live = live
        self.logits = live.model.lm_head(live.terminal)[:, -1]
