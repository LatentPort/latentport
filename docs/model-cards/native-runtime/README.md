---
tags:
- latentport
- gguf
- native-runtime
---

# native-runtime

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

[GitHub](https://github.com/LatentPort/latentport) | [PyPI](https://pypi.org/project/latentport/) | [Installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) | [API](https://github.com/LatentPort/latentport/blob/main/docs/api.md)

## Overview

Prebuilt native binaries for the shared LatentPort Python library. They implement native inference-state operations, including compaction; this repository does not contain a language model.

## Compatibility

| Platform | CUDA runtime | GPU architecture | Native ABI |
| --- | --- | --- | --- |
| Windows x64 | 13.0 | NVIDIA Blackwell SM120 | 2 |
| Linux x64 | 12.8 | NVIDIA Blackwell SM120 | 2 |

These are ABI-2 builds. Check `native-runtime.json` and the package installation guide for compatibility with your package version, GPU and CUDA runtime.

## Usage

```shell
python -m pip install --upgrade latentport
latentport install-native
```

Install CUDA libraries and backend dependencies as described in the installation guide. The package pins the native download to an immutable repository revision.

## Evaluation

The archives listed in `package-native.json` are byte-identical to the native builds pinned by LatentPort 0.4.2. Their SHA-256 receipts verify identity, not model answer quality. The older archives listed in `native-runtime.json` remain available for existing pinned clients.

## Limitations

Prebuilt binaries support only the listed platforms and GPU architecture. Other configurations require a compatible build and validation. Base models, bridge weights and NVIDIA drivers are distributed separately.

## Artifacts and provenance

`package-native.json` records the platform, CUDA version, ABI, byte counts and SHA-256 values of the currently selected archives. Older archives have a separate `native-runtime.json`, `SHA256SUMS.txt` and embedded `native-build.json`; the newer package archives do not contain that embedded build receipt. [Source and build instructions](https://github.com/LatentPort/latentport/tree/main/native).

## License

LatentPort source is Apache-2.0 licensed ([LICENSE](LICENSE)). Patched llama.cpp/ggml components retain their MIT license ([LICENSE.llama.cpp](LICENSE.llama.cpp)), also included in each archive. Preserve these notices when redistributing binaries.
