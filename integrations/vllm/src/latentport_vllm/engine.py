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

        # Reserve uncompacted requests and one replacement. The physical budget
        # for a sliding-window group covers its live pages and in-flight tokens.
        model_len = config.model_config.max_model_len
        per_request = sum(
            math.ceil(spec.max_memory_usage_bytes(config) / spec.page_size_bytes)
            for spec in (
                group.kv_cache_spec for group in cache.kv_cache_config.kv_cache_groups
            )
        )
        staged_rows = settings.selector.target_rows(model_len, settings.ratio)
        staging = (
            sum(
                math.ceil(staged_rows / self.managers[group].block_size)
                for group in self.groups
            )
            if model_len >= settings.minimum_rows
            else 0
        )
        needed = per_request * config.scheduler_config.max_num_seqs + staging
        available = self.pool.get_num_free_blocks()
        if needed > available:
            raise ValueError(
                f"LPCompact needs {needed} available cache pages for the configured "
                f"sequence limit and staging reserve, but vLLM has {available}. "
                "Reduce --max-num-seqs or --max-model-len."
            )

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
            return
        start = perf_counter()
        staged = {
            group: self.pool.get_new_blocks(count) for group, count in demand.items()
        }
        destination = tuple(
            [block.block_id for block in staged[group]] if group in staged else ids
            for group, ids in enumerate(source)
        )
        try:
            errors = self.engine.collective_rpc(
                "latentport_stage",
                args=(request_id, logical, physical, target, source, destination),
            )
        except Exception:
            for blocks in staged.values():
                self.pool.free_blocks(blocks)
            raise
        if any(error is not None for error in errors):
            self.engine.collective_rpc(
                "latentport_finish_stage", args=(request_id, None)
            )
            for blocks in staged.values():
                self.pool.free_blocks(blocks)
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
