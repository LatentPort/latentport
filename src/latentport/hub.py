"""Pinned Hugging Face asset resolution with a bulk cache outside the code tree."""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_REPO = "latentport/latentport-4-27b-research-preview"
DEFAULT_REVISION = "08570694b0fef9ca79430cc9dcddb7d365cf84f4"
GGUF_REPO = "latentport/latentport-4-27b-research-preview-GGUF"
GGUF_REVISION = "880c3f9aeb30b929346ba3f68b719d8bf21756a9"
GGUF_NAME = "latentport-4-27b-bridge.F32.gguf"


def bulk_home() -> Path:
    override = os.environ.get("LATENTPORT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    drive = Path("F:/LatentPort")
    return (drive / "runtime" if drive.exists() else Path.home() / ".cache" / "latentport").resolve()


def _snapshot(repo_id: str, revision: str, *, patterns=None) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, revision=revision,
                                  cache_dir=str(bulk_home() / "huggingface"),
                                  allow_patterns=patterns)).resolve()


def bundle_root(value: str | Path = DEFAULT_REPO, *, revision: str = DEFAULT_REVISION,
                full: bool = False) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    local = Path("F:/LatentPort/releases/latentport-4-27b-research-preview")
    if (str(value) == DEFAULT_REPO and revision == DEFAULT_REVISION and
        local.is_dir() and (local / "latentport.json").exists()):
        return local
    patterns = None if full else ["latentport.json", "bridge_config.json"]
    return _snapshot(str(value), revision, patterns=patterns)


def model_snapshot(spec: dict, *, metadata_only: bool) -> Path:
    patterns = (["config.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "special_tokens_map.json", "generation_config.json"]
                if metadata_only else None)
    return _snapshot(spec["repo_id"], spec["revision"], patterns=patterns)


def gguf_bridge_file(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    local = Path("F:/LatentPort/releases/latentport-4-27b-research-preview-GGUF") / GGUF_NAME
    if local.is_file():
        return local
    root = _snapshot(GGUF_REPO, GGUF_REVISION, patterns=[GGUF_NAME, "latentport.json"])
    return root / GGUF_NAME
