"""Flash-Next readout over the first attention block's values and routing keys."""
from contextlib import ExitStack, contextmanager
import os
import sys

from ._models import FLASH_NEXT
from ._combine_readout import _readout, _rank, profile
from ._reference.native import KERNEL_LOCK
from .errors import CompatibilityError, StateError


@contextmanager
def early_experts(text):
    import torch
    from ._reference.flash_next import resident_experts
    from ._reference.flash_weights import StreamExperts

    layers = text.layers[:3]
    pending = [(layer, sum(t.numel() * t.element_size() for t in (experts.gate, experts.down)))
               for layer in layers if isinstance(experts := layer.mlp.experts, StreamExperts)]
    device = text.embed_tokens.weight.device
    reserve = 1024 ** 3
    free = torch.cuda.mem_get_info(device)[0]
    if pending and sum(size for _, size in pending) + reserve >= free:
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(device)[0]
    with ExitStack() as stack:
        for layer, size in pending:
            if size + reserve < free:
                stack.enter_context(resident_experts(layer, text.config))
                free -= size
        yield


@contextmanager
def random_ngram_reads(text):
    if sys.platform != "linux":
        yield
        return

    import ctypes
    import mmap
    from ._reference.flash_weights import ShardedEmbedding

    advise = ctypes.CDLL(None).madvise
    advise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    regions = []
    # The verifier visits unrelated token IDs, so sequential read-ahead on
    # these CPU mappings would read many pages between the requested rows.
    for layer in text.layers[:3]:
        for module in layer.modules():
            if not isinstance(module, ShardedEmbedding):
                continue
            for shard in module.shards:
                start = shard.data_ptr()
                base = start - start % mmap.PAGESIZE
                size = start - base + shard.numel() * shard.element_size()
                if advise(base, size, mmap.MADV_RANDOM) == 0:
                    regions.append((base, size))
    try:
        yield
    finally:
        for base, size in regions:
            advise(base, size, mmap.MADV_NORMAL)


def shallow_fingerprints(text, ids, *, first_keys=False):
    from transformers import DynamicCache
    from ._reference.flash_next import append_layer

    hidden = text.embed_tokens(ids).repeat(1, 1, text.config.hc_count)
    cache = DynamicCache(config=text.config)
    for layer in text.layers[:3]:
        hidden = append_layer(layer, hidden, ids, cache, None, 0)
    first_attention = text.layers[3]
    hidden, _, _ = first_attention.attn_hyper_connection(hidden)
    attention = first_attention.self_attn
    values = attention.v_proj(hidden).float()
    indexer = attention.indexer
    index = indexer.index_qk_proj(hidden)[..., -indexer.index_head_dim:].float()
    if first_keys:
        shape = (*hidden.shape[:2], -1, attention.head_dim)
        keys = attention.k_norm(attention.k_proj(hidden).reshape(shape)).flatten(2).float()
        return values, index, keys
    return values, index


def features(values, index):
    import torch
    from torch.nn.functional import normalize

    return torch.cat([normalize(x, dim=-1) * x.size(-1) ** 0.5 for x in (values, index)], dim=-1)


def _fingerprint_error(values, target):
    return sum((x[:, -1] - y).square().mean(-1) / y.square().mean().clamp_min(1e-9)
               for x, y in zip(values, target))


def _verify(text, sources, first_key, candidates, special, common):
    import torch
    ids = []
    for position, row in enumerate(candidates):
        target = [bank[position] for bank in sources]
        used = set()
        best_error, winner = float("inf"), None
        first = position == 0
        tolerance = 0.0003 if first else 0.006
        if first:
            target.append(first_key)
        batch_size = max(8, min(512, 512 // (position + 1)))
        for shortlist in (row[:16], row[:512], torch.cat((common, special)), row):
            pool = [i for i in dict.fromkeys(shortlist.tolist()) if i not in used]
            used.update(pool)
            for group in torch.tensor(pool, device=row.device, dtype=torch.long).split(batch_size):
                if not len(group):
                    continue
                prefix = torch.tensor(ids, dtype=torch.long, device=row.device).expand(len(group), -1)
                sequences = torch.cat((prefix, group[:, None]), dim=1)
                values = shallow_fingerprints(text, sequences, first_keys=first)
                error = _fingerprint_error(values, target)
                minimum, offset = error.min(0)
                score = float(minimum)
                if not first and tolerance < score < 0.03:
                    # Expert routing can change slightly with the batch shape.
                    # Confirm a near match in a batch containing only that prefix.
                    repeated = sequences[offset].expand(4, -1).contiguous()
                    confirmed = shallow_fingerprints(text, repeated)
                    score = min(score, float(_fingerprint_error(confirmed, target).min()))
                if score < best_error:
                    best_error, winner = score, int(group[offset])
                if best_error <= tolerance:
                    break
            if best_error <= tolerance:
                break
        if best_error > (0.0015 if first else tolerance):
            raise StateError(f"LPCombine could not verify right-state token {position}. Use a retained token history.")
        ids.append(winner)
    return tuple(ids)


def recover_tokens(model, state):
    import torch
    from torch.nn.functional import normalize
    from transformers import DynamicCache
    from ._reference.flash_next import append_layer

    spec = profile(FLASH_NEXT)
    runtime = model.backend
    if (model.identity.weights.partition(":execution:")[0] != spec["bf16_weights"] or
            runtime.options.position_policy != "default"):
        raise CompatibilityError("Flash-Next cache readout requires the pinned BF16 weights and native text positions")
    values = state._read_tensors(device=runtime.device)
    if "P" in values or state.physical_kv_length != state.clock:
        raise StateError("Flash-Next cache-only combine requires a dense right state")
    attention = values["V.3"][0].transpose(0, 1).flatten(1).float()
    index = values["I.3"][0].float()
    first_key = values["K.3"][0, :, 0].flatten().float()
    if attention.shape != (state.clock, 512) or index.shape != (state.clock, 128):
        raise StateError("Unexpected Flash-Next attention or index cache geometry")
    text = runtime._model(model.identity).model
    special = torch.arange(248044, min(model.codec.vocabulary_size, text.embed_tokens.num_embeddings),
                           device=attention.device)
    common = torch.arange(min(4096, model.codec.vocabulary_size), device=attention.device)
    weight, bias = _readout(FLASH_NEXT, os.environ.get("LATENTPORT_COMBINER"))
    with KERNEL_LOCK, torch.inference_mode(), runtime.kernels(), early_experts(text), random_ngram_reads(text):
        x = features(attention, index)
        projected = normalize(x @ weight.to(x.device) + bias.to(x.device), dim=-1).to(torch.bfloat16)
        candidates = _rank(text.embed_tokens.weight, projected, 16384)
        recovered = _verify(text, (attention, index), first_key, candidates, special, common)
        ids = torch.tensor([recovered], device=attention.device, dtype=torch.long)
        cache = DynamicCache(config=text.config)
        for start in range(0, ids.size(1), 1024):
            tokens = ids[:, start : start + 1024]
            hidden = text.embed_tokens(tokens).repeat(1, 1, text.config.hc_count)
            append_layer(text.layers[0], hidden, tokens, cache, None, start)
        first = cache.layers[0]
        recurrence = values["R.0"].float()
        residual = (first.recurrent_states[0].float() - recurrence).square().mean()
        if (not torch.equal(first.conv_states[0], values["C.0"]) or
                not bool(residual <= 0.0002 * recurrence.square().mean().clamp_min(1e-9))):
            raise StateError("LPCombine could not verify the complete Flash-Next token stream. Use retained tokens.")
        return recovered
