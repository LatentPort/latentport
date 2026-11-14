# Changelog

## 0.2.1

- Move native-runtime downloads to a dedicated Hugging Face repository with
  immutable revision pins and the original SHA-256 checksums.
- Present LPCompact as LatentPort's compaction feature and retire its separate
  GitHub product release. Upgrade from 0.2.0 before a fresh native installation.

## 0.2.0

- Lightweight PyPI package with GGUF/BF16 backend extras and dependency-aware diagnostics.
- Packaged browser demo with automatic runtime setup, model caching, disk checks,
  Docker-compatible binding, and a headless verification command.
- Versioned native installation and pinned Windows CUDA runtime components.
- External GGUF manifests with immutable revisions/checksums and standard HF cache reuse.
- GGUF/BF16 container build targets, package CI, and manual Trusted Publishing workflows.
- Existing model pins remain unchanged pending external-model GPU qualification.

## 0.1.0.dev1

- Composite sessions now default to `cached_handoff`: translate the initial prompt once,
  then keep the 27B state for follow-up turns. Select `policy="strict"` explicitly for
  reader synchronization and translation on every turn.
- Accept valid GGUF files stored as extensionless Hugging Face cache blobs; check the
  file signature instead of rejecting a resolved path without a `.gguf` suffix.
- Pinned public downloads for the tested Q4_K_M model pair and Windows native runtime.
- Installation diagnostics, composite examples, cookbook and qualified support matrix.

## 0.1.0.dev0

Initial development wheel. Superseded by dev1 for direct Hub-cache execution and the
new default conversation policy. Existing immutable artifact URLs remain available.
