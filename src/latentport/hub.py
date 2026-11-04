"""Pinned Hugging Face asset resolution with a bulk cache outside the code tree."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path


DEFAULT_REPO = "latentport/latentport-4-27b-research-preview"
DEFAULT_REVISION = "cf50f0c61c55454cbfce4b2ef334632ae461bb54"
GGUF_REPO = "latentport/latentport-4-27b-research-preview-GGUF"
GGUF_REVISION = "880c3f9aeb30b929346ba3f68b719d8bf21756a9"
GGUF_NAME = "latentport-4-27b-bridge.F32.gguf"


def bulk_home() -> Path:
    override = os.environ.get("LATENTPORT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "latentport").resolve()


def _snapshot(repo_id: str, revision: str, *, patterns=None) -> Path:
    from huggingface_hub import snapshot_download, constants
    return Path(snapshot_download(repo_id=repo_id, revision=revision,
                                  cache_dir=str(bulk_home() / "huggingface"),
                                  allow_patterns=patterns,
                                  local_files_only=constants.HF_HUB_OFFLINE)).resolve()


def bundle_root(value: str | Path = DEFAULT_REPO, *, revision: str = DEFAULT_REVISION,
                full: bool = False) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    patterns = None if full else ["latentport.json", "bridge_config.json"]
    return _snapshot(str(value), revision, patterns=patterns)


def model_snapshot(spec: dict, *, metadata_only: bool) -> Path:
    # A fully cached pinned tokenizer must work without a Hub directory-list
    # request. Also reuse the user's existing HF cache before downloading.
    if metadata_only and spec.get("metadata_files"):
        from huggingface_hub import try_to_load_from_cache
        for cache in (str(bulk_home() / "huggingface"), None):
            cached = try_to_load_from_cache(spec["repo_id"], "config.json",
                                           revision=spec["revision"], cache_dir=cache)
            if isinstance(cached, str):
                root = Path(cached).parent
                if all((root / name).is_file() and
                       (root / name).stat().st_size == receipt["bytes"] and
                       hashlib.sha256((root / name).read_bytes()).hexdigest() == receipt["sha256"]
                       for name, receipt in spec["metadata_files"].items()):
                    return root.resolve()
    patterns = (["config.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "special_tokens_map.json", "generation_config.json"]
                if metadata_only else None)
    return _snapshot(spec["repo_id"], spec["revision"], patterns=patterns)


def gguf_bridge_file(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    root = _snapshot(GGUF_REPO, GGUF_REVISION, patterns=[GGUF_NAME, "latentport.json"])
    return root / GGUF_NAME


def gguf_model_file(side: str) -> Path:
    """Download the exact qualified GGUF, never substitute another quantization."""
    import json
    from huggingface_hub import hf_hub_download
    spec = json.loads((Path(__file__).parent / "configs/gguf-models.json").read_text())[side]
    path = Path(hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"],
                               cache_dir=bulk_home()/"huggingface"))
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Downloaded {side} GGUF has an unexpected size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"Downloaded {side} GGUF failed SHA-256 verification")
    return path
