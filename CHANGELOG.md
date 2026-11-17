# Changelog

## 0.4.6

- Add standalone BF16 text inference and LPCompact for Gemma 4 31B IT. Global attention uses the learned selector and sliding attention retains its native window.

## 0.4.5

- Preserve the authoritative native working cache when parking fails during snapshot allocation.
- Cache GGUF tokenizers only after all compatibility probes pass, including retries.
- Reject token IDs outside int32 and native vocabulary bounds before modifying caches.
- Preserve Unicode text across bounded generation calls, forks and snapshot restoration.
- Check compaction base-weight fingerprints against an offline pinned catalog; unknown weights require explicit `allow_unqualified=True` and still enforce revision and ABI checks.

- Standardize public model cards and operation examples while preserving creator attribution and checkpoint-specific evaluation results.
- Replace internal experiment labels with descriptive names in public documentation, metadata and private runtime attributes.
- Repair native installation after removal of the GitHub release by hosting the exact checksum-verified archives on Hugging Face.

## 0.4.4

- Use the standard standalone model, state and session APIs for Qwen3.8-Flash-Next BF16 compaction.

## 0.4.3

- Add `compact_flash_next_cache` for the pinned Qwen3.8-Flash-Next BF16
  Transformers text cache. The child retains original logical positions,
  sparse indexer keys and independent recurrent, convolution and n-gram state.

## 0.4.2

- Use the same capture/translate/ask workflow and composite conversation API for
  every supported bridge, with checkpoint-specific question boundaries handled
  by the runtime. Add common repository, backend and context flags to the chat example.
- Organize the README around state operations, with one transfer recipe, one
  compaction recipe and a shared catalog of supported transformations.
- Preserve prominent creator attribution and align Docker wheel paths with the release version.

## 0.4.1

- Support the Qwen3.5-4B to Qwen3.5-9B step-64 bridge with BF16, NF4 and Q4_K_M GGUF query handoff.

- Restore request boundaries after failed ordinary and composite asks. Emit tokens
  only after backend commit, and report failed streams with `GenerationInterrupted`.
- Protect translation inputs with private tensor storage and give `Port` sole
  ownership of its shared runtime; model views close independently.
- Reject invalid exact token IDs and explicitly mark incomplete handoff transcripts.
- Write v2 snapshots with manifest integrity receipts; trusted v1 snapshots require
  explicit `allow_legacy=True` when loading.
- Preserve resident handles on no-op transforms, add session compaction token
  reservations, and enforce the trained revision for the default compaction model.
- Repair compact example backend options and MTP allocation in the chat example.

## 0.4.0

- Load standalone Qwen3.5-4B and Qwen3.5-9B by model ID and compact their
  immutable states with dedicated learned selectors. Model identity selects
  the pinned base revision, complete-state geometry and selector checksum.

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
