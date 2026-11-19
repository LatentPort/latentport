"""The release's pinned model and cache layouts."""

from dataclasses import dataclass
from pathlib import Path

from latentport.selector import SelectorSpec, get_selector_spec

CONFIG_KEY = "latentport_lpcompact"
QWEN_27B = "Qwen/Qwen3.8-27B"
FLASH_NEXT = "Qwen/Qwen3.8-Flash-Next"
FLASH_NEXT_NVFP4 = "Inferact/Qwen3.8-Flash-Next-NVFP4"
QWEN_27B_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
FLASH_NEXT_REVISION = "812d96e10a0dd82b69d749e68693698cf6c36daa"


@dataclass(frozen=True)
class ModelProfile:
    revision: str
    selector_base: str
    selector_base_revision: str
    attention_layers: int
    indexed: bool = False
    quantization: str | None = None


PROFILES = {
    QWEN_27B: ModelProfile(QWEN_27B_REVISION, QWEN_27B, QWEN_27B_REVISION, 16),
    FLASH_NEXT: ModelProfile(
        FLASH_NEXT_REVISION, FLASH_NEXT, FLASH_NEXT_REVISION, 12, True
    ),
    FLASH_NEXT_NVFP4: ModelProfile(
        "103a7608316173ca6edd49929544244de7ffda70",
        FLASH_NEXT,
        FLASH_NEXT_REVISION,
        12,
        True,
        "modelopt_fp4",
    ),
}


@dataclass(frozen=True)
class Settings:
    model: str
    profile: ModelProfile
    selector: SelectorSpec
    ratio: int
    minimum_rows: int
    weights_path: str | None

    @classmethod
    def load(cls, model, options):
        if model not in PROFILES:
            raise ValueError(f"Unsupported LPCompact vLLM model: {model}")
        profile = PROFILES[model]
        selector = get_selector_spec(
            profile.selector_base,
            profile.selector_base_revision,
            options["artifact"],
        )
        minimum_rows = options.get("minimum_rows")
        if minimum_rows is None:
            minimum_rows = selector.minimum_rows
        ratio = options.get("ratio", 8)
        selector.target_rows(0, ratio)
        if type(minimum_rows) is not int or minimum_rows < 1056:
            raise ValueError("LPCompact minimum rows must be at least 1056")
        return cls(
            model,
            profile,
            selector,
            ratio,
            minimum_rows,
            options.get("weights_path"),
        )


def get_settings(config):
    options = config.additional_config.get(CONFIG_KEY)
    if options is None:
        return None
    model = config.model_config
    settings = Settings.load(options.get("base_model", model.model), options)
    parallel = config.parallel_config
    cache = config.cache_config
    scheduler = config.scheduler_config
    if model.revision != settings.profile.revision:
        raise ValueError("LPCompact requires the pinned base model revision")
    if model.model != settings.model:
        snapshot = Path(model.model)
        repo_cache = "models--" + settings.model.replace("/", "--")
        if tuple(snapshot.parts[-3:]) != (
            repo_cache,
            "snapshots",
            settings.profile.revision,
        ):
            raise ValueError(
                "LPCompact requires the pinned Hugging Face base checkpoint"
            )
    if (
        model.quantization != settings.profile.quantization
        or str(model.dtype) != "torch.bfloat16"
    ):
        raise ValueError("LPCompact requires the pinned weight format and BF16 compute")
    if model.hf_overrides and config.load_config.load_format != "dummy":
        raise ValueError("LPCompact requires the base checkpoint's model configuration")
    if (
        cache.cache_dtype not in ("auto", "bfloat16")
        or cache.mamba_cache_mode != "none"
    ):
        raise ValueError("LPCompact requires BF16 KV and Mamba cache mode 'none'")
    if (
        cache.enable_prefix_caching
        or scheduler.async_scheduling
        or scheduler.disable_hybrid_kv_cache_manager
        or cache.kv_sharing_fast_prefill
        or parallel.use_ubatching
        or config.kv_transfer_config
        or config.speculative_config
        or config.lora_config
        or config.max_concurrent_batches != 1
        or not model.enforce_eager
        or any(
            size != 1
            for size in (
                parallel.pipeline_parallel_size,
                parallel.data_parallel_size,
                parallel.decode_context_parallel_size,
                parallel.prefill_context_parallel_size,
            )
        )
    ):
        raise ValueError(
            "LPCompact requires eager, synchronous HMA serving without prefix "
            "caching, KV transfer, speculative decoding, LoRA, PP, DP or CP"
        )
    mm = model.multimodal_config
    if mm is not None and not mm.language_model_only:
        raise ValueError("LPCompact currently requires text-only serving")
    return settings
