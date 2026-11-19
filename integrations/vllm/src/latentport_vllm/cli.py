"""Launch a pinned vLLM server with LatentPort's compaction configuration."""

import argparse
import json
import os
import sys
from importlib.metadata import entry_points

from latentport._models import FLASH_NEXT, GEMMA, QWEN3, QWEN36
from latentport.errors import CompatibilityError

from .settings import (
    CONFIG_KEY,
    FLASH_NEXT_NVFP4,
    PROFILES,
    QUANTIZED_LOADERS,
    SESSION_PLUGIN,
    SESSION_TTL_DEFAULT,
    WEIGHT_FORMATS,
    Settings,
)


def build_args(argv):
    parser = argparse.ArgumentParser(prog="latentport-vllm")
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("model", choices=tuple(PROFILES))
    parser.add_argument("--lpcompact", required=True, metavar="HF_REPO")
    parser.add_argument("--lpcompact-ratio", type=int, choices=(4, 8), default=8)
    parser.add_argument("--lpcompact-min-rows", type=int, metavar="ROWS")
    parser.add_argument("--lpcompact-weights-path", metavar="PATH")
    parser.add_argument("--weight-format", choices=WEIGHT_FORMATS)
    parser.add_argument("--gguf-path", metavar="PATH")
    parser.add_argument("--lpcompact-max-sessions", type=int, default=0, metavar="COUNT")
    parser.add_argument(
        "--lpcompact-session-ttl",
        type=int,
        default=SESSION_TTL_DEFAULT,
        metavar="SECONDS",
    )
    settings, passthrough = parser.parse_known_args(argv)
    options = {
        "base_model": settings.model,
        "artifact": settings.lpcompact,
        "ratio": settings.lpcompact_ratio,
        "minimum_rows": settings.lpcompact_min_rows,
        "weights_path": settings.lpcompact_weights_path,
        "weight_format": settings.weight_format,
        "gguf_path": settings.gguf_path,
    }
    if (
        settings.lpcompact_max_sessions
        or settings.lpcompact_session_ttl != SESSION_TTL_DEFAULT
    ):
        options.update(
            max_sessions=settings.lpcompact_max_sessions,
            session_ttl=settings.lpcompact_session_ttl,
        )
    try:
        resolved = Settings.load(settings.model, options)
    except (CompatibilityError, ValueError) as exc:
        parser.error(str(exc))
    revision = resolved.profile.revision
    max_model_len = str(min(139264, resolved.selector.native_positions))
    memory_utilization = "0.91"
    text_only_args = [] if settings.model == QWEN3 else ["--language-model-only"]
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
    elif settings.model == FLASH_NEXT and resolved.load_format:
        variant_args = [
            "--engram-config",
            '{"disk_offload":true}',
            "--max-num-batched-tokens",
            "2048",
            "--gdn-prefill-backend",
            "triton",
        ]
    elif settings.model == GEMMA:
        variant_args = ["--attention-backend", "TRITON_ATTN"]
    elif settings.model == QWEN36:
        variant_args = [
            "--gdn-prefill-backend",
            "triton",
        ]
        if resolved.weight_format == "bf16":
            variant_args += ["--moe-backend", "triton"]
    fixed = {
        "--revision",
        "--tokenizer-revision",
        "--additional-config",
        "--worker-extension-cls",
    }
    if resolved.max_sessions:
        fixed.update({"--api-server-count", "--headless", "--grpc", "--rust-frontend"})
        variant_args += ["--api-server-count", "1"]
    elif settings.lpcompact_session_ttl != SESSION_TTL_DEFAULT:
        parser.error("--lpcompact-session-ttl requires --lpcompact-max-sessions")
    if resolved.load_format is not None:
        fixed.update(
            {
                "--quantization",
                "-q",
                "--quantization-config",
                "--load-format",
                "--model-loader-extra-config",
                "--hf-config-path",
                "--config-format",
                "--tokenizer",
                "--model-impl",
            }
        )
        if settings.model == FLASH_NEXT:
            fixed.add("--engram-config")
        variant_args += [
            "--quantization",
            resolved.quantization,
            "--load-format",
            resolved.load_format,
            "--model-loader-extra-config",
            json.dumps({"base_model": settings.model, "gguf_path": settings.gguf_path}),
        ]
    if any(flag.partition("=")[0].replace("_", "-") in fixed for flag in passthrough):
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
        *text_only_args,
        "--enforce-eager",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--max-model-len",
        max_model_len,
        "--max-num-seqs",
        str(resolved.max_sessions or 1),
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
    options = json.loads(args[args.index("--additional-config") + 1])[CONFIG_KEY]
    if options.get("max_sessions"):
        enabled = os.environ.get("VLLM_PLUGINS")
        names = (
            [item.name for item in entry_points(group="vllm.general_plugins")]
            if enabled is None
            else enabled.split(",")
        )
        os.environ["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*names, SESSION_PLUGIN]))
        os.environ["VLLM_USE_RUST_FRONTEND"] = "0"
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
    if "--load-format" in args:
        load_format = args[args.index("--load-format") + 1]
        if load_format in QUANTIZED_LOADERS.values():
            from .weight_plugins import require_plugin

            try:
                require_plugin(load_format)
            except ValueError as exc:
                raise SystemExit(str(exc)) from None
    sys.argv[1:] = args
    return vllm_main()
