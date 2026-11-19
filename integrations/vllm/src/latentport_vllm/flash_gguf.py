"""Translate Flash-Next GGUF tensors to the pinned Qwen4Exp layout."""

from types import MethodType

import gguf
import torch
from vllm_gguf_plugin.weight_utils import get_gguf_tensor_names
from vllm_gguf_plugin.weights_adapter import Qwen35GGUFAdapter
from vllm_gguf_plugin.weights_adapter.qwen3_5 import (
    build_qwen35_text_mapper,
    qwen35_layer_substr,
)

_PLE_TABLE = "per_layer_token_embd.weight"
_TEXT = "model.language_model."
_FLASH_TOP = {
    "output_hc_norm.weight": "hyper_connection_mixer.hc_norm.weight",
    "output_hc_down.weight": "hyper_connection_mixer.gguf_hc_down.weight",
    "output_hc_up.weight": "hyper_connection_mixer.gguf_hc_up.weight",
}
_FLASH_LAYER = {
    "attn_qkv_gate.weight": "linear_attn.in_proj_qkvz.weight",
    "ssm_beta_alpha.weight": "linear_attn.in_proj_ba.weight",
    "indexer.q_proj.weight": "self_attn.indexer.gguf_q.weight",
    "indexer.k_proj.weight": "self_attn.indexer.gguf_k.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
    "hc_attn_down.weight": "attn_hyper_connection.gguf_hc_down.weight",
    "hc_attn_down_inject.weight": "attn_hyper_connection.gguf_hc_down_inject.weight",
    "hc_attn_inject.weight": "attn_hyper_connection.gguf_hc_inject.weight",
    "hc_attn_up.weight": "attn_hyper_connection.gguf_hc_up.weight",
    "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
    "hc_ffn_down.weight": "mlp_hyper_connection.gguf_hc_down.weight",
    "hc_ffn_down_inject.weight": "mlp_hyper_connection.gguf_hc_down_inject.weight",
    "hc_ffn_inject.weight": "mlp_hyper_connection.gguf_hc_inject.weight",
    "hc_ffn_up.weight": "mlp_hyper_connection.gguf_hc_up.weight",
    "ple_key.weight": "ple.key_proj.weight",
    "ple_value.weight": "ple.value_proj.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
    "ple_conv1d.weight": "ple.conv1d.weight",
}
_FLOAT_PROJECTIONS = {
    "gguf_hc_down": "input_mix_weight_down",
    "gguf_hc_up": "input_mix_weight_up",
    "gguf_hc_inject": "block_inject_weight",
    "gguf_hc_down_inject": None,
    "gguf_q": None,
    "gguf_k": None,
}


class FlashNextGGUFAdapter(Qwen35GGUFAdapter):
    extra_unquantized_modules = ("indexer.index_qk_proj",)

    def build_name_map(self, files, model_config):
        mapper = build_qwen35_text_mapper(is_multimodal=True, is_moe=True)
        text_config = model_config.hf_config.get_text_config()
        common_layer_names = tuple(qwen35_layer_substr(True))
        mapped = {}
        for name in sorted(get_gguf_tensor_names(files.backbone)):
            if name == _PLE_TABLE:
                continue
            if name in _FLASH_TOP:
                mapped[name] = _TEXT + _FLASH_TOP[name]
                continue
            if name.startswith("blk."):
                _, layer, suffix = name.split(".", 2)
                if int(layer) >= text_config.num_hidden_layers:
                    continue
                if suffix in _FLASH_LAYER:
                    mapped[name] = f"{_TEXT}layers.{layer}.{_FLASH_LAYER[suffix]}"
                    continue
                if not suffix.startswith(common_layer_names):
                    raise ValueError(f"Unrecognized Flash-Next GGUF tensor: {name}")
            translated = mapper.apply_list([name])[0]
            if translated == name:
                raise ValueError(f"Unrecognized Flash-Next GGUF tensor: {name}")
            mapped[name] = translated
        return mapped

    def transform_weights(self, weights, model_config):
        prepared = self._split_flash_projections(weights, model_config)
        yield from super().transform_weights(prepared, model_config)

    def _split_flash_projections(self, weights, model_config):
        config = model_config.hf_config.get_text_config()
        value_rows = config.linear_num_value_heads * config.linear_value_head_dim
        qkv_rows = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim + value_rows
        )
        packed = {
            "in_proj_qkvz": (("in_proj_qkv", qkv_rows), ("in_proj_z", value_rows)),
            "in_proj_ba": (
                ("in_proj_b", config.linear_num_value_heads),
                ("in_proj_a", config.linear_num_value_heads),
            ),
        }
        quant_types = {}
        indexer_parts = {}
        for name, tensor in weights:
            base, _, field = name.rpartition(".")
            prefix, _, projection = base.rpartition(".")
            if projection in packed:
                parts = packed[projection]
                shards = (
                    (tensor,) * len(parts)
                    if field == "weight_type"
                    else tensor.split([size for _, size in parts], dim=0)
                )
                for (target, _), shard in zip(parts, shards, strict=True):
                    yield f"{prefix}.{target}.{field}", shard
                continue
            if projection in _FLOAT_PROJECTIONS:
                if field == "weight_type":
                    quant_types[prefix, projection] = gguf.GGMLQuantizationType(
                        tensor.item()
                    )
                    continue
                kind = quant_types.pop((prefix, projection), None)
                if kind is not None:
                    tensor = torch.from_numpy(
                        gguf.quants.dequantize(tensor.numpy(), kind)
                    )
                if projection in ("gguf_q", "gguf_k"):
                    pair = indexer_parts.setdefault(prefix, {})
                    pair[projection] = tensor
                    if len(pair) == 2:
                        yield (
                            f"{prefix}.index_qk_proj.weight",
                            torch.cat((pair["gguf_q"], pair["gguf_k"])),
                        )
                        del indexer_parts[prefix]
                elif projection == "gguf_hc_down_inject":
                    down, inject = tensor.split((config.hc_lowrank, config.hc_count))
                    yield f"{prefix}.input_mix_weight_down.weight", down
                    yield f"{prefix}.block_inject_weight.weight", inject
                else:
                    yield f"{prefix}.{_FLOAT_PROJECTIONS[projection]}.weight", tensor
                continue
            if name.endswith(
                (
                    ".ple.norm_conv.weight",
                    ".ple.norm_key.weight",
                    ".ple.norm_query.weight",
                )
            ):
                tensor = tensor - 1
            yield name, tensor
        if indexer_parts or quant_types:
            raise ValueError("Incomplete Flash-Next GGUF projections")


def bind_ple_table(model, paths):
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        Qwen4ExpPLEDiskOffloadEmbedding,
    )

    (layer,) = [
        module
        for module in model.modules()
        if isinstance(module, Qwen4ExpPLEDiskOffloadEmbedding)
    ]
    (table,) = [
        tensor
        for path in paths
        for tensor in gguf.GGUFReader(path).tensors
        if tensor.name == _PLE_TABLE
    ]
    if (
        tuple(table.shape) != (layer.embedding_dim, layer.org_vocab_size)
        or table.tensor_type != gguf.GGMLQuantizationType.Q4_0
    ):
        raise ValueError("Unexpected Flash-Next GGUF n-gram table")

    # The table stays in the mmap. Each step dequantizes only the rows it uses
    # into vLLM's existing BF16 host and device staging buffers.
    layer.stage_rows = MethodType(_stage_ple_rows, layer)
    layer._lpcompact_gguf_table = table
    layer._bound = True


def _stage_ple_rows(layer, num_tokens):
    table = layer._lpcompact_gguf_table
    ids = layer._host_ids[:num_tokens].flatten()
    distinct, inverse = ids.unique(return_inverse=True)
    if distinct.numel() and not 0 <= distinct[0] <= distinct[-1] < layer.org_vocab_size:
        raise IndexError("Flash-Next GGUF n-gram id is out of range")
    staging, host, _ = layer._planes["weight"]
    if distinct.numel():
        packed = table.data[distinct.numpy()]
        decoded = torch.from_numpy(gguf.quants.dequantize(packed, table.tensor_type))
        host[:num_tokens].flatten(0, 1).copy_(decoded.index_select(0, inverse))
    staging[:num_tokens].copy_(host[:num_tokens], non_blocking=True)
