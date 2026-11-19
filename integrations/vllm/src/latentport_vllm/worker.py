"""Worker-owned cache layout, selection, and page copy."""

from dataclasses import dataclass
from itertools import pairwise

import torch

from latentport.selector import load_kv_selector


@dataclass
class PageRows:
    group: int
    tensor: torch.Tensor
    stride: int = 1
    layer: int | None = None

    def _address(self, pages, rows):
        width = self.tensor.shape[1]
        return pages[rows // width], rows % width

    def read(self, pages, rows):
        return self.tensor[self._address(pages, rows)]

    def write(self, pages, rows, values):
        self.tensor[self._address(pages, rows)] = values


@dataclass
class CompactedRequest:
    logical_rows: int
    positions: torch.Tensor

    @property
    def dropped(self):
        return self.logical_rows - self.positions.numel()


def _layer_index(name):
    fields = name.split(".")
    if "layers" not in fields:
        return None
    return int(fields[fields.index("layers") + 1])


def _cache_layout(runner, settings):
    """Resolve the cache views the runner bound to the actual model layers."""
    state = runner.vllm_config.compilation_config.static_forward_context
    layers = []
    indexers = []
    raw_groups = set()
    movable_names = set()
    expected_layers = set(settings.selector.attention_layers)
    backends = (
        {"QWEN4_EXP_QSA_TRITON"}
        if settings.profile.indexed
        else {"FLASH_ATTN", "TRITON_ATTN"}
    )
    for group, config in enumerate(runner.kv_cache_config.kv_cache_groups):
        for name in config.layer_names:
            layer = state[name]
            number = _layer_index(name)
            backend = layer.get_attn_backend().get_name()
            if backend in backends and number in expected_layers:
                cache = layer.kv_cache.transpose(1, 2)
                head_size = (
                    layer.head_dim if settings.profile.indexed else layer.head_size
                )
                if cache.shape[-1] != 2 * head_size:
                    raise ValueError(f"Unexpected KV layout for {name}")
                layers.append(PageRows(group, cache, layer=number))
                movable_names.add(name)
            elif settings.profile.indexed and name.endswith(".compressed_key_cache"):
                indexers.append(PageRows(group, layer.kv_cache, 4, number))
                movable_names.add(name)
            elif settings.profile.indexed and name.endswith(".raw_key_cache"):
                if layer.rope_position_cache is None:
                    raise ValueError(
                        "Flash-Next needs original RoPE coordinates in its QSA ring"
                    )
                raw_groups.add(group)

    numbers = {row.layer for row in layers}
    if len(layers) != len(numbers) or numbers != expected_layers:
        raise ValueError(
            f"Unexpected compactable attention layers. Missing: {sorted(expected_layers - numbers)}"
        )
    if settings.profile.indexed and (
        len(indexers) != len(layers)
        or {row.layer for row in indexers} != numbers
        or not raw_groups
    ):
        raise ValueError("Flash-Next has an incomplete paged QSA cache")
    if any(
        row.tensor.ndim != 4
        or row.tensor.dtype != torch.bfloat16
        or row.tensor.shape[1] * row.stride
        != runner.block_tables.block_sizes[row.group]
        for row in (*layers, *indexers)
    ):
        raise ValueError(
            "LPCompact requires BF16 cache pages with the expected row width"
        )

    move = {row.group for row in (*layers, *indexers)}
    if any(
        name not in movable_names
        for group in move
        for name in runner.kv_cache_config.kv_cache_groups[group].layer_names
    ):
        raise ValueError("LPCompact found an unexpected cache owner in a movable group")
    by_layer = {row.layer: row for row in layers}
    features = tuple(by_layer[number] for number in settings.selector.feature_layers)
    return (
        (*layers, *indexers),
        features,
        tuple(sorted(move)),
        tuple(sorted(move | raw_groups)),
    )


def _validate_keep(keep, rows, target, granularity):
    if (
        len(keep) != target
        or not keep
        or keep[0] < 0
        or keep[-1] >= rows
        or any(a >= b for a, b in pairwise(keep))
    ):
        raise ValueError("LPCompact returned invalid retained rows")
    if granularity > 1:
        tail = rows % granularity
        full = len(keep) - tail
        if keep[full:] != list(range(rows - tail, rows)) or any(
            keep[i] % granularity
            or keep[i + granularity - 1] != keep[i] + granularity - 1
            for i in range(0, full, granularity)
        ):
            raise ValueError("LPCompact split a native QSA index group")


class WorkerCompaction:
    def __init__(self, runner, settings):
        from vllm.distributed.parallel_state import get_tp_group

        self.runner = runner
        self.device = runner.device
        self.settings = settings
        self.tp = get_tp_group()
        self.kv_heads = settings.selector.attention_kv_heads
        self.rows, self.features, self.movable_groups, self.packed_groups = (
            _cache_layout(runner, settings)
        )
        if any(
            heads < self.kv_heads or heads % self.kv_heads
            for heads in (
                row.tensor.shape[2] * self.tp.world_size for row in self.features
            )
        ) or sum(
            row.tensor.shape[-1] // 2 * self.kv_heads for row in self.features
        ) != (settings.selector.feature_width):
            raise ValueError(
                "LPCompact cache geometry does not match the selector features"
            )
        self.selector = load_kv_selector(
            settings.selector.base_model,
            settings.selector.base_revision,
            settings.selector.artifact,
            device=str(self.device),
            weights_path=settings.weights_path,
        )
        self.requests = {}
        self.pending = None

    def _global_features(self, view, pages, rows):
        values = view.read(pages, rows)[..., view.tensor.shape[-1] // 2 :].contiguous()
        if self.tp.world_size > 1:
            values = self.tp.all_gather(values, dim=1)
            replicas = values.shape[1] // self.kv_heads
            values = values[:, ::replicas]
        return values

    @torch.inference_mode()
    def stage(self, request_id, logical_rows, rows, target, source, destination):
        if self.pending is not None:
            raise RuntimeError("LPCompact copy already staged")
        if request_id not in self.runner.req_states.req_id_to_index:
            raise RuntimeError("LPCompact request is absent from the worker")
        prior = self.requests.get(request_id)
        if rows != logical_rows - (prior.dropped if prior else 0):
            raise RuntimeError(
                "LPCompact worker and scheduler disagree on packed length"
            )
        available = {
            group: torch.tensor(source[group], device=self.device, dtype=torch.long)
            for group in self.movable_groups
        }
        reserved = {
            group: torch.tensor(
                destination[group], device=self.device, dtype=torch.long
            )
            for group in self.movable_groups
        }

        def read_values(start, stop):
            offsets = torch.arange(start, stop, device=self.device)
            return [
                self._global_features(view, available[view.group], offsets)
                for view in self.features
            ]

        def copy_rows(positions):
            for view in self.rows:
                group_rows = (
                    positions
                    if view.stride == 1
                    else (
                        positions[: target - target % view.stride : view.stride]
                        // view.stride
                    )
                )
                for start in range(0, group_rows.numel(), 8192):
                    source_rows = group_rows[start : start + 8192]
                    output_rows = torch.arange(
                        start,
                        start + source_rows.numel(),
                        device=self.device,
                    )
                    view.write(
                        reserved[view.group],
                        output_rows,
                        view.read(available[view.group], source_rows),
                    )

        try:
            keep = self.selector.select(read_values, rows, self.settings.ratio)
            _validate_keep(keep, rows, target, self.settings.selector.granularity)
            positions = torch.tensor(keep, device=self.device, dtype=torch.long)
            if self.tp.world_size > 1:
                positions = self.tp.broadcast(positions, src=0)
            copy_rows(positions)
            if prior is not None:
                recent = torch.arange(
                    prior.logical_rows, logical_rows, device=self.device
                )
                positions = torch.cat((prior.positions, recent))[positions]
            torch.cuda.current_stream(self.device).synchronize()
        except (torch.OutOfMemoryError, ValueError) as exc:
            # The active block table still points at the old cache.
            torch.cuda.current_stream(self.device).synchronize()
            return str(exc)
        self.pending = request_id, CompactedRequest(logical_rows, positions)
        return None

    def finish_stage(self, request_id, replacement):
        pending, self.pending = self.pending, None
        if replacement is None:
            return
        if pending is None or pending[0] != request_id:
            raise RuntimeError("LPCompact has no matching staged copy")
        index = self.runner.req_states.req_id_to_index[request_id]
        self.runner.block_tables.append_block_ids(index, replacement, overwrite=True)
        self.runner.block_tables.apply_staged_writes()
        self.requests[request_id] = pending[1]

    def forget(self, request_ids):
        for request_id in request_ids:
            self.requests.pop(request_id, None)

    def prepare_attention(self, batch, tables, slots):
        if not self.requests:
            return
        offsets = torch.tensor(
            [
                self.requests[req_id].dropped if req_id in self.requests else 0
                for req_id in batch.req_ids
            ],
            dtype=torch.int32,
        )
        if not offsets.any():
            return
        if batch.num_reqs != batch.num_reqs_after_padding:
            raise RuntimeError("LPCompact requires unpadded eager batches")
        gpu_offsets = offsets.to(self.device, non_blocking=True)
        packed_lens = batch.seq_lens - gpu_offsets
        packed_upper = batch.seq_lens_cpu_upper_bound - offsets
        batch.packed_attention_lengths = {
            group: (packed_lens, packed_upper, int(packed_upper.max()))
            for group in self.packed_groups
        }
        row_to_request = torch.repeat_interleave(
            torch.arange(batch.num_reqs, device=self.device),
            torch.diff(batch.query_start_loc),
            output_size=batch.num_tokens,
        )
        packed_positions = (
            batch.positions[: batch.num_tokens] - gpu_offsets[row_to_request]
        )
        for group in self.movable_groups:
            if self.runner.block_tables._slot_mapping_enabled[group]:
                size = self.runner.block_tables.block_sizes[group]
                block = tables[group][row_to_request, packed_positions // size]
                slots[group, : batch.num_tokens] = (
                    block.long() * size + packed_positions % size
                )


class WorkerExtension:
    def latentport_init(self, settings):
        compactor = WorkerCompaction(self.model_runner, settings)
        self.model_runner.latentport_compactor = compactor
        return compactor.movable_groups

    def latentport_stage(self, *args):
        return self.model_runner.latentport_compactor.stage(*args)

    def latentport_finish_stage(self, *args):
        self.model_runner.latentport_compactor.finish_stage(*args)
