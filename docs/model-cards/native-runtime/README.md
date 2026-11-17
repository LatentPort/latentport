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

The archives are byte-identical to the previously qualified native builds. Their build metadata identifies the qualified platform and runtime; binary availability is not evidence of model answer quality.

## Limitations

Prebuilt binaries support only the listed platforms and GPU architecture. Other configurations require a compatible build and validation. Base models, bridge weights and NVIDIA drivers are distributed separately.

## Artifacts and provenance

`native-runtime.json` records platform, compiler/runtime compatibility, source revisions, byte counts and SHA-256 values. Each archive includes `native-build.json`; `SHA256SUMS.txt` verifies the ZIP files. [Source and build instructions](https://github.com/LatentPort/latentport/tree/main/native).

## License

LatentPort source is Apache-2.0 licensed ([LICENSE](LICENSE)). Patched llama.cpp/ggml components retain their MIT license ([LICENSE.llama.cpp](LICENSE.llama.cpp)), also included in each archive. Preserve these notices when redistributing binaries.
