"""Pinned Hugging Face asset resolution with a bulk cache outside the code tree."""
from __future__ import annotations

import os
import hashlib
import shutil
from pathlib import Path


DEFAULT_REPO = "latentport/LPBridge-Qwen3.5-4b-Qwen3.8-27b"
DEFAULT_REVISION = "c8607cfac644eddb8ebdb8a9dc6079873c2015a4"


def verified(path, spec):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Missing file or size mismatch: {path}")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    return path


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


def bundle_root(value: str | Path = DEFAULT_REPO, *, revision=None, full=False, backend=None) -> Path:
    import json
    from pathlib import PurePosixPath
    local = Path(value).expanduser().is_dir()
    if not local and revision is None and str(value) == DEFAULT_REPO:
        revision = DEFAULT_REVISION
    root = Path(value).resolve() if local else _snapshot(str(value), revision, patterns=["latentport.json"])
    manifest = json.loads((root / "latentport.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "latentport.bridge-bundle.v2":
        raise ValueError("This runtime requires an LPBridge v2 bundle")
    if not local and manifest["repo_id"] != str(value):
        raise ValueError("Bridge repository identity mismatch")
    files = manifest["artifacts"]
    if not {"bridge_config.json", "configs/geometry.json", "configs/calibration.json",
            "configs/architecture_v2.json"}.issubset(files):
        raise ValueError("Bridge configuration is missing a checksum")
    for name in files:
        parts = PurePosixPath(name)
        if not parts.parts or parts.is_absolute() or ".." in parts.parts or "\\" in name or ":" in name or str(parts) != name:
            raise ValueError("Invalid bridge artifact path")
    for endpoint in manifest["models"].values():
        from re import fullmatch
        if not fullmatch("[0-9a-f]{40}", endpoint["revision"]):
            raise ValueError("Each base model requires an immutable revision")
        if any(Path(name).name != name or "/" in name or "\\" in name or ":" in name or name in ("", ".", "..")
               for name in (*endpoint["metadata_files"], *endpoint["weight_files"])):
            raise ValueError("Invalid base model filename")
    gguf = manifest["gguf_bridge"]
    if gguf not in files:
        raise ValueError("Missing GGUF receipt")
    names = [name for name in files if name.endswith(".json") or
             (full and (name == gguf if backend == "gguf" else name.endswith(".safetensors")))]
    if not local:
        # HF snapshot directory names are immutable commit IDs, even for tags.
        root = _snapshot(str(value), root.name, patterns=["latentport.json", *names])
    for name in names:
        verified(root / name, files[name])
    config = json.loads((root / "bridge_config.json").read_text(encoding="utf-8"))
    if config["architecture"] != manifest["architecture"] or any(
            name not in files or not name.endswith(".safetensors") for name in config["weights"].values()):
        raise ValueError("Bridge graph refers to an unverified weight file")
    return root


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
    if not metadata_only and spec.get("weight_files"):
        patterns = [*spec["metadata_files"], *spec["weight_files"]]
    root = _snapshot(spec["repo_id"], spec["revision"], patterns=patterns)
    for name, record in spec.get("metadata_files", {}).items():
        verified(root / name, record)
    return root


def gguf_bridge_file(value=None, *, root=None):
    import json
    root = Path(root) if root is not None else bundle_root(full=True, backend="gguf")
    manifest = json.loads((root / "latentport.json").read_text())
    name = manifest["gguf_bridge"]
    return verified(value or root / name, manifest["artifacts"][name])


def gguf_model_specs(manifest_path=None, *, root=None):
    """Resolve an explicit pinned profile; repositories need not be owned by us."""
    import json
    import re
    if manifest_path:
        specs = json.loads(Path(manifest_path).read_text())
    else:
        root = Path(root) if root is not None else bundle_root()
        specs = json.loads((root / "latentport.json").read_text())["gguf_models"]
    for side in ("source", "target"):
        spec = specs[side]
        if manifest_path or spec.get("repo_id"):
            if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
                raise ValueError(f"{side}: model revision must be a full commit SHA")
            if not spec.get("repo_id"):
                raise ValueError(f"{side}: repository is required")
        if not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")):
            raise ValueError(f"{side}: model SHA-256 is required")
        if type(spec.get("bytes")) is not int or spec["bytes"] <= 0 or not spec.get("filename"):
            raise ValueError(f"{side}: positive model byte size and filename are required")
    return specs


def gguf_model_file(side: str, *, manifest_path=None, root=None, local_path=None) -> Path:
    """Download the exact qualified GGUF, never substitute another quantization."""
    from huggingface_hub import hf_hub_download
    expected = gguf_model_specs(root=root)[side]
    spec = gguf_model_specs(manifest_path, root=root)[side] if manifest_path else expected
    if (expected["bytes"], expected["sha256"]) != (spec["bytes"], spec["sha256"]):
        raise ValueError(f"{side}: external model profile differs from bridge manifest")
    if local_path is not None:
        return verified(local_path, expected)
    candidate = bulk_home() / "models" / Path(expected["filename"]).name
    if candidate.is_file() and candidate.stat().st_size == expected["bytes"]:
        return verified(candidate, expected)
    if not spec.get("repo_id"):
        raise FileNotFoundError(f"Supply the pinned {side} GGUF via {side}_path, "
                                f"a verified external model manifest, or {candidate}")
    # Reuse an exact cached upstream file before populating our own cache.
    cached = cached_file(spec["repo_id"], spec["filename"], spec["revision"])
    if cached is None:
        check_download_space(spec["bytes"])
    path = Path(cached if cached is not None else hf_hub_download(
        spec["repo_id"], spec["filename"], revision=spec["revision"], cache_dir=bulk_home()/"huggingface"))
    return verified(path, spec)
