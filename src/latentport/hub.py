"""Pinned Hugging Face asset resolution with a bulk cache outside the code tree."""
from __future__ import annotations

import os
import hashlib
import shutil
from pathlib import Path


DEFAULT_REPO = "latentport/latentport-4-27b-research-preview"
DEFAULT_REVISION = "cf50f0c61c55454cbfce4b2ef334632ae461bb54"
GGUF_REPO = "latentport/latentport-4-27b-research-preview-GGUF"
GGUF_REVISION = "880c3f9aeb30b929346ba3f68b719d8bf21756a9"
GGUF_NAME = "latentport-4-27b-bridge.F32.gguf"
GGUF_BYTES = 2578356288


def cached_file(repo_id, filename, revision):
    from huggingface_hub import try_to_load_from_cache
    for cache in (str(bulk_home()/"huggingface"), None):
        cached = try_to_load_from_cache(repo_id, filename, revision=revision, cache_dir=cache)
        if isinstance(cached, str) and Path(cached).is_file():
            return Path(cached)
    return None


def check_download_space(required):
    """Check the destination filesystem, including a modest metadata/scratch margin."""
    if required <= 0:
        return
    directory = bulk_home()
    while not directory.exists():
        directory = directory.parent
    free = shutil.disk_usage(directory).free
    if free < required + 1024**3:
        raise RuntimeError(f"Insufficient disk space at {directory}: need at least "
                           f"{(required + 1024**3)/1e9:.2f} GB free; have {free/1e9:.2f} GB.")


def transfer_download_bytes(manifest_path=None):
    specs = gguf_model_specs(manifest_path)
    required = sum(spec["bytes"] for spec in (specs['source'], specs['target'])
                   if cached_file(spec["repo_id"], spec["filename"], spec["revision"]) is None)
    if cached_file(GGUF_REPO, GGUF_NAME, GGUF_REVISION) is None:
        required += GGUF_BYTES
    return required


def bulk_home() -> Path:
    override = os.environ.get("LATENTPORT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "latentport").resolve()


def _snapshot(repo_id: str, revision: str, *, patterns=None) -> Path:
    from huggingface_hub import snapshot_download, constants
    # Reuse complete snapshots already present in the standard HF cache. A local-only
    # lookup must never turn into a network request or accept a partial snapshot.
    if patterns and not any('*' in pattern for pattern in patterns):
        from huggingface_hub import try_to_load_from_cache
        paths = [try_to_load_from_cache(repo_id, name, revision=revision) for name in patterns]
        if all(isinstance(path, str) and Path(path).is_file() for path in paths):
            return Path(paths[0]).parent.resolve()
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
    cached = cached_file(GGUF_REPO, GGUF_NAME, GGUF_REVISION)
    if cached is not None:
        return cached
    check_download_space(GGUF_BYTES)
    root = _snapshot(GGUF_REPO, GGUF_REVISION, patterns=[GGUF_NAME, "latentport.json"])
    return root / GGUF_NAME


def gguf_model_specs(manifest_path=None):
    """Resolve an explicit pinned profile; repositories need not be owned by us."""
    import json
    import re
    path = Path(manifest_path) if manifest_path else Path(__file__).parent / "configs/gguf-models.json"
    specs = json.loads(path.read_text())
    if not manifest_path:
        return specs
    for side in ("source", "target"):
        spec = specs[side]
        if manifest_path:  # Existing release manifests remain backwards compatible.
            if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
                raise ValueError(f"{side}: model revision must be a full commit SHA")
            if not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")):
                raise ValueError(f"{side}: model SHA-256 is required")
            if type(spec.get("bytes")) is not int or spec["bytes"] <= 0:
                raise ValueError(f"{side}: positive model byte size is required")
            if not spec.get("repo_id") or not spec.get("filename"):
                raise ValueError(f"{side}: repository and filename are required")
    return specs


def gguf_model_file(side: str, *, manifest_path=None) -> Path:
    """Download the exact qualified GGUF, never substitute another quantization."""
    from huggingface_hub import hf_hub_download
    spec = gguf_model_specs(manifest_path)[side]
    # Reuse an exact cached upstream file before populating our own cache.
    cached = cached_file(spec["repo_id"], spec["filename"], spec["revision"])
    if cached is None:
        check_download_space(spec["bytes"])
    path = Path(cached if cached is not None else hf_hub_download(
        spec["repo_id"], spec["filename"], revision=spec["revision"], cache_dir=bulk_home()/"huggingface"))
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Downloaded {side} GGUF has an unexpected size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"Downloaded {side} GGUF failed SHA-256 verification")
    return path
