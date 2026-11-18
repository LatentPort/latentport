"""Read Qwen3 and Gemma tokens from the context-independent first value projection."""
from ._models import GEMMA
from ._compact.compatibility import _standalone_fingerprint
from ._reference.native import KERNEL_LOCK
from .errors import CompatibilityError, StateError


def _bank(decoder, size):
    import torch
    blocks = []
    with decoder.layer(0) as layer:
        attention = layer.self_attn
        for ids in torch.arange(size, device=decoder.device).split(4096):
            hidden = layer.input_layernorm(decoder.text.embed_tokens(ids))
            projection = attention.v_proj if attention.v_proj is not None else attention.k_proj
            values = projection(hidden)
            if hasattr(attention, "v_norm"):
                values = attention.v_norm(values.reshape(len(ids), -1, attention.head_dim)).flatten(1)
            blocks.append(values)
    return torch.cat(blocks)


def recover_tokens(model, state):
    import torch
    from torch.nn.functional import normalize

    identity = model.identity
    if identity.weights.partition(":execution:")[0] != "bf16-sha256:" + _standalone_fingerprint(identity.repo_id):
        raise CompatibilityError("Cache readout requires the exact pinned BF16 checkpoint")
    values = state._read_tensors(device=model.backend.device)
    field = "SV.0" if identity.repo_id == GEMMA else "V.0"
    first = values[field]
    if "P" in values or first.ndim != 4 or first.shape[0] != 1 or first.shape[2] != state.clock:
        raise StateError("The complete first-layer value cache is required. Use retained tokens for a sparse or truncated right state.")

    decoder = model.backend._model(identity)
    with KERNEL_LOCK, torch.inference_mode():
        bank = getattr(decoder, "_combine_value_bank", None)
        if bank is None:
            table = _bank(decoder, model.codec.vocabulary_size)
            unit = torch.cat([normalize(x.float(), dim=-1).to(table.dtype) for x in table.split(4096)])
            decoder._combine_value_bank = bank = (table, unit)
        table, unit = bank
        rows = first[0].transpose(0, 1).flatten(1).float()
        result = []
        for group in rows.split(64):
            similarity = normalize(group, dim=-1).to(unit.dtype) @ unit.T
            choices = similarity.topk(8, dim=-1).indices
            error = (table[choices].float() - group[:, None]).square().mean(-1)
            error /= group.square().mean(-1, keepdim=True).clamp_min(1e-9)
            minimum, offset = error.min(-1)
            unique = error.le(0.0001).sum(-1).eq(1)
            if not bool((minimum.le(0.0001) & unique).all()):
                raise StateError("LPCombine could not uniquely verify a right-state token. Use a retained token history.")
            result.extend(choices.gather(1, offset[:, None]).flatten().tolist())
        return tuple(result)
