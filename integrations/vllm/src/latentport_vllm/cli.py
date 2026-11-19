"""Launch a pinned vLLM server with LatentPort's compaction configuration."""

import argparse
import json
import sys

from latentport.errors import CompatibilityError

from .settings import CONFIG_KEY, FLASH_NEXT_NVFP4, PROFILES, Settings


def build_args(argv):
    parser = argparse.ArgumentParser(prog="latentport-vllm")
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("model", choices=tuple(PROFILES))
    parser.add_argument("--lpcompact", required=True, metavar="HF_REPO")
    parser.add_argument("--lpcompact-ratio", type=int, choices=(4, 8), default=8)
    parser.add_argument("--lpcompact-min-rows", type=int, metavar="ROWS")
    parser.add_argument("--lpcompact-weights-path", metavar="PATH")
    settings, passthrough = parser.parse_known_args(argv)
    options = {
        "base_model": settings.model,
        "artifact": settings.lpcompact,
        "ratio": settings.lpcompact_ratio,
        "minimum_rows": settings.lpcompact_min_rows,
        "weights_path": settings.lpcompact_weights_path,
    }
    try:
        resolved = Settings.load(settings.model, options)
    except (CompatibilityError, ValueError) as exc:
        parser.error(str(exc))
    revision = resolved.profile.revision
    max_model_len = "139264"
    memory_utilization = "0.91"
    variant_args = []
    if settings.model == FLASH_NEXT_NVFP4:
        max_model_len = "69632"
        memory_utilization = "0.94"
        variant_args = [
            "--engram-config",
            '{"disk_offload":true}',
            "--max-num-batched-tokens",
            "2048",
            "--moe-backend",
            "marlin",
            "--gdn-prefill-backend",
            "triton",
        ]
    fixed = {
        "--revision",
        "--tokenizer-revision",
        "--additional-config",
        "--worker-extension-cls",
    }
    if any(flag.partition("=")[0] in fixed for flag in passthrough):
        parser.error(
            "the launcher sets model pins, additional config, and the worker extension"
        )
    return [
        "serve",
        settings.model,
        "--revision",
        revision,
        "--tokenizer-revision",
        revision,
        "--dtype",
        "bfloat16",
        "--language-model-only",
        "--enforce-eager",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--max-model-len",
        max_model_len,
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        memory_utilization,
        "--worker-extension-cls",
        "latentport_vllm.worker.WorkerExtension",
        "--additional-config",
        json.dumps({CONFIG_KEY: options}),
        *variant_args,
        *passthrough,
    ]


def main():
    args = build_args(sys.argv[1:])
    try:
        from vllm.entrypoints.cli.main import main as vllm_main
        from vllm.v1.engine.core import EngineCore
    except ModuleNotFoundError as exc:
        if exc.name != "vllm":
            raise
        raise SystemExit(
            "The pinned latentport-vllm image supplies vLLM 0.34.0."
        ) from None

    if getattr(EngineCore, "latentport_compaction_api", None) != 1:
        raise SystemExit(
            "Use the pinned latentport-vllm image or apply its vLLM patch."
        )
    sys.argv[1:] = args
    return vllm_main()
