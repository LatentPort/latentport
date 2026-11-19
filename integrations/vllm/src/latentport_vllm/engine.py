"""Allocate and commit a replacement cache between completed vLLM steps."""

import logging
import math
from time import perf_counter

logger = logging.getLogger("vllm.latentport")


class EngineCompaction:
    def __init__(self, engine, config, settings):
        self.engine = engine
        self.settings = settings
        self.scheduler = engine.scheduler
        cache = self.scheduler.kv_cache_manager
        self.cache = cache
        self.pool = cache.block_pool
        self.managers = cache.coordinator.single_type_managers
        layouts = engine.collective_rpc("latentport_init", args=(settings,))
        self.groups = layouts[0]
        if not self.groups or any(groups != self.groups for groups in layouts):
            raise ValueError("LPCompact workers disagree on movable cache groups")

        model_len = config.model_config.max_model_len
        scheduler_config = config.scheduler_config
        slots = scheduler_config.max_num_seqs
        sequence_rows = model_len
        if settings.max_sessions:
            slots = scheduler_config.max_num_active_seqs or slots
            if settings.max_sessions > slots:
                raise ValueError(
                    "LPCompact session limit exceeds vLLM's active sequence limit"
                )
            # One completed synchronous scheduler step separates checks. Give
            # low thresholds another block so the replacement can reclaim a page.
            block_size = max(self.managers[group].block_size for group in self.groups)
            trigger = max(
                settings.minimum_rows,
                1024 + settings.selector.granularity + block_size,
                2 * block_size,
            )
            sequence_rows = min(
                model_len, trigger + scheduler_config.max_num_batched_tokens - 1
            )
        self.reduced_reservation = sequence_rows < model_len
        self.sequence_rows = sequence_rows

        # Keep the native bounds for recurrent and sliding-window cache groups.
        per_request = sum(
            math.ceil(sequence_rows / self.managers[index].block_size)
            if self.reduced_reservation and index in self.groups
            else math.ceil(
                group.kv_cache_spec.max_memory_usage_bytes(config)
                / group.kv_cache_spec.page_size_bytes
            )
            for index, group in enumerate(cache.kv_cache_config.kv_cache_groups)
        )
        staged_rows = (
            max(1024 + settings.selector.granularity - 1, sequence_rows // settings.ratio)
            if self.reduced_reservation
            else settings.selector.target_rows(model_len, settings.ratio)
        )
        staging = (
            sum(
                math.ceil(staged_rows / self.managers[group].block_size)
                for group in self.groups
            )
            if model_len >= settings.minimum_rows
            else 0
        )
        watermark = cache.watermark_blocks if settings.max_sessions else 0
        needed = per_request * slots + staging + watermark
        available = self.pool.get_num_free_blocks()
        if needed > available:
            raise ValueError(
                f"LPCompact needs {needed} available cache pages for the configured "
                f"sequence limit and staging reserve, but vLLM has {available}. "
                "Reduce --max-num-seqs or --max-model-len."
            )
        if settings.max_sessions:
            logger.info(
                "LPCompact residency slots=%d rows_per_slot=%d pages_per_slot=%d "
                "staging_pages=%d watermark_pages=%d available_pages=%d",
                slots, sequence_rows, per_request, staging, watermark, available,
            )
        if self.reduced_reservation:
            # The pool already reserves each slot's full physical envelope.
            # A check against the uncompressed logical prompt would reject
            # valid admissions even when chunked prefill has room to progress.
            self.scheduler.scheduler_reserve_full_isl = False

    def after_step(self, request_ids):
        for request_id in request_ids:
            request = self.scheduler.requests.get(request_id)
            if request is None or request.is_finished():
                continue
            previous = self.managers[self.groups[0]].packed_rows_dropped.get(
                request_id, 0
            )
            logical = request.num_computed_tokens
            physical = logical - previous
            if physical < self.settings.minimum_rows:
                continue
            target = self.settings.selector.target_rows(physical, self.settings.ratio)
            if target < physical:
                self._compact(request_id, logical, physical, target)

    def _compact(self, request_id, logical, physical, target):
        source = self.cache.get_block_ids(request_id)
        demand = {
            group: math.ceil(target / self.managers[group].block_size)
            for group in self.groups
        }
        required = sum(demand.values())
        if (
            required >= sum(len(source[group]) for group in self.groups)
            or self.pool.get_num_free_blocks() < required
        ):
            if self.reduced_reservation and physical >= self.sequence_rows:
                raise RuntimeError("LPCompact exhausted its replacement page reserve")
            return
        start = perf_counter()
        staged = {
            group: self.pool.get_new_blocks(count) for group, count in demand.items()
        }
        destination = tuple(
            [block.block_id for block in staged[group]] if group in staged else ids
            for group, ids in enumerate(source)
        )
        # An RPC exception is ambiguous: another worker may still be writing.
        # Let the engine fail with both page sets still reserved.
        errors = self.engine.collective_rpc(
            "latentport_stage",
            args=(request_id, logical, physical, target, source, destination),
        )
        if any(error is not None for error in errors):
            self.engine.collective_rpc(
                "latentport_finish_stage", args=(request_id, None)
            )
            for blocks in staged.values():
                self.pool.free_blocks(blocks)
            if self.reduced_reservation:
                raise RuntimeError(
                    f"LPCompact could not stage resident attention pages: {errors}"
                )
            logger.warning("LPCompact could not stage attention pages: %s", errors)
            return

        self.engine.collective_rpc(
            "latentport_finish_stage", args=(request_id, destination)
        )
        old_count = 0
        for group, blocks in staged.items():
            manager = self.managers[group]
            old = manager.req_to_blocks[request_id]
            old_count += len(old)
            manager.req_to_blocks[request_id] = blocks
            manager.packed_rows_dropped[request_id] = logical - target
            self.pool.free_blocks(reversed(old))
        spec = self.settings.selector
        logger.info(
            "LPCompact artifact=%s revision=%s ratio=%d logical=%d rows=%d->%d "
            "pages_reclaimed=%d elapsed_ms=%.1f",
            spec.artifact,
            spec.artifact_revision,
            self.settings.ratio,
            logical,
            physical,
            target,
            old_count - required,
            1000 * (perf_counter() - start),
        )
