# Changelog

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
