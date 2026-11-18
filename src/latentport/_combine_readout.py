"""Pinned NF4 LPCombine readout and causal verification of candidate tokens."""
from copy import copy
from functools import lru_cache
import json
import os
from pathlib import Path

from ._models import PINNED_REVISION
from .errors import CompatibilityError, StateError, CapabilityError
from .hub import verified


@lru_cache(maxsize=1)
def profile():
    return json.loads((Path(__file__).parent / "configs/lpcombine-qwen3.8-27b.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _readout(override):
    from safetensors.torch import load_file
    spec = profile()
    if override:
        path = Path(override).expanduser()
    else:
        from huggingface_hub import hf_hub_download
        from .hub import bulk_home
        path = hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"],
                               cache_dir=bulk_home() / "huggingface")
    tensors = load_file(str(verified(path, spec)), device="cpu")
    if set(tensors) != {"weight", "bias"} or tensors["weight"].shape != (2048, 5120) or tensors["bias"].shape != (5120,):
        raise CompatibilityError("LPCombine readout has unexpected tensor shapes")
    return tensors["weight"], tensors["bias"]


def _shortlist(language, keys, values, width):
    import torch
    from torch.nn.functional import normalize
    from transformers.models.qwen3_5.modeling_qwen3_5 import rotate_half
    device = keys.device
    length = keys.shape[2]
    positions = torch.arange(length, device=device).expand(3, 1, -1)
    cos, sin = language.rotary_emb(keys.float(), positions)
    dim = cos.size(-1)
    rot = keys[..., :dim].float()
    unrotated = torch.cat((rot * cos[:, None] - rotate_half(rot) * sin[:, None], keys[..., dim:].float()), dim=-1)

    def feature(tensor):
        flat = tensor.transpose(1, 2).flatten(2).float()[0]
        return normalize(flat, dim=-1) * flat.shape[-1] ** 0.5

    weight, bias = _readout(os.environ.get("LATENTPORT_COMBINER"))
    features = torch.cat((feature(values), feature(unrotated)), dim=-1)
    projected = normalize(features @ weight.to(device) + bias.to(device), dim=-1).to(torch.bfloat16)
    # Normalize embeddings in blocks to avoid a vocabulary-sized FP32 temporary.
    table = language.get_input_embeddings().weight
    unit = torch.cat([normalize(chunk.float(), dim=-1).to(torch.bfloat16) for chunk in table.split(8192)])
    candidates = []
    for chunk in projected.split(64):
        scores = chunk @ unit.T
        scores[:, 248044:] = torch.finfo(scores.dtype).min
        candidates.append(scores.topk(width, dim=-1).indices)
    return torch.cat(candidates)


def _cache_rows(cache, rows):
    # Copies only the first four initialized layers and never writes the source.
    next_cache = copy(cache)
    next_cache.layers = list(cache.layers)
    for i, layer in enumerate(cache.layers[:4]):
        child = copy(layer)
        if hasattr(layer, "recurrent_states"):
            child.conv_states = {0: layer.conv_states[0].index_select(0, rows)}
            child.recurrent_states = {0: layer.recurrent_states[0].index_select(0, rows)}
            child.has_previous_state = layer.has_previous_state.copy()
            child.is_conv_states_initialized = layer.is_conv_states_initialized.copy()
            child.is_recurrent_states_initialized = layer.is_recurrent_states_initialized.copy()
            child.conv_kernel_size = layer.conv_kernel_size.copy()
        else:
            child.keys = layer.keys.index_select(0, rows)
            child.values = layer.values.index_select(0, rows)
        next_cache.layers[i] = child
    return next_cache


def _verify(language, keys, values, candidates, special):
    import torch
    from transformers import DynamicCache
    # Share the immutable modules, with a private layer list for this readout.
    decoder = copy(language)
    decoder._modules = language._modules.copy()
    decoder.layers = language.layers[:4]
    prefix = None
    ids = []
    for position, row in enumerate(candidates):
        best_error = float("inf")
        selected = set()
        next_cache = None
        token = None
        reference = (keys[0, :, position, :].float(), values[0, :, position, :].float())
        for shortlist in (row[:16], torch.cat((row, special))):
            pool = [i for i in dict.fromkeys(shortlist.tolist()) if i not in selected]
            selected.update(pool)
            for group in torch.tensor(pool, dtype=torch.long, device=keys.device).split(64):
                if not len(group):
                    continue
                trial = (DynamicCache(config=language.config) if prefix is None else
                         _cache_rows(prefix, torch.zeros_like(group)))
                trial = decoder(input_ids=group[:, None], past_key_values=trial,
                                use_cache=True).past_key_values
                layer = trial.layers[3]
                error = sum((tensor[:, :, -1, :].float() - target).square().mean((1, 2)) /
                            target.square().mean().clamp_min(1e-9)
                            for tensor, target in zip((layer.keys, layer.values), reference))
                minimum, index = error.min(0)
                if float(minimum) < best_error:
                    best_error, token = float(minimum), int(group[index])
                    next_cache = _cache_rows(trial, index.reshape(1))
                if best_error <= 0.02:
                    break
            if best_error <= 0.02:
                break
        if next_cache is None or best_error > 0.02:
            raise StateError(f"LPCombine could not verify right-state token {position}. Use a retained token history for exact replay.")
        prefix = next_cache
        ids.append(token)
    return tuple(ids)


def recover_tokens(model, state):
    import torch
    spec = profile()
    identity = model.identity
    runtime = model.backend
    if (identity.revision != PINNED_REVISION or
            identity.weights.partition(":execution:")[0] != spec["nf4_weights"] or
            getattr(getattr(runtime, "options", None), "position_policy", None) != "default"):
        raise CompatibilityError("NF4 cache-only combine requires the pinned model weights and native text positions")
    if state.physical_kv_length != state.clock:
        raise StateError("Cache-only combine requires a dense right state. Use its retained token history after compaction.")
    values = state._read_tensors(device=runtime.device)
    keys, attention_values = values["K.3"], values["V.3"]
    if (keys.shape != (1, 4, state.clock, 256) or keys.shape != attention_values.shape or
            "P" in values):
        raise StateError("LPCombine requires a dense, zero-start Qwen3.8 text state")
    language = runtime._model(identity).model
    maximum = model.codec.vocabulary_size
    special = torch.arange(248044, min(maximum, language.get_input_embeddings().num_embeddings),
                           device=keys.device)
    with torch.inference_mode(), runtime.kernels():
        candidates = _shortlist(language, keys, attention_values, 1024)
        return _verify(language, keys, attention_values, candidates, special)
