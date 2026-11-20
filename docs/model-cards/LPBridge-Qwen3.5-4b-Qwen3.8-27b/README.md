---
library_name: latentport
license: apache-2.0
base_model:
- Qwen/Qwen3.5-4B
- Qwen/Qwen3.8-27B
tags:
- state-transfer
- gguf
---

# LPBridge-Qwen3.5-4b-Qwen3.8-27b

[LPBridge overview](https://latentport.com/lpbridge)

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

[GitHub](https://github.com/LatentPort/latentport) | [PyPI](https://pypi.org/project/latentport/) | [Installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) | [API](https://github.com/LatentPort/latentport/blob/main/docs/api.md)

## Overview

A directional learned bridge from **Qwen/Qwen3.5-4B** to **Qwen/Qwen3.8-27B**. It translates attention keys and values, convolution history, recurrent memory and terminal hidden state so the target can continue without prefilling the historical context. Base-model weights are obtained separately.

## Compatibility

Use LatentPort 0.4.2 or later with Python 3.11+, CUDA PyTorch and the dependencies for `bf16`, `nf4` or `gguf`. Exact source and target revisions are recorded in `latentport.json`. GGUF requires a compatible LatentPort native library and qualified model files. Use `thinking=False` or `thinking=True` to select the corresponding bridge parameters. The bridge requires complete dense source state and its aligned token history.

## Usage

Install the runtime and backend dependencies using the installation guide above. Select the bridge repository; the operation sequence is the same for every supported pair.

```python
import latentport as lp

with lp.load("latentport/LPBridge-Qwen3.5-4b-Qwen3.8-27b", backend="nf4", thinking=False) as port:
    state = port.source.capture([("user", "The project codename is Juniper.")])
    transferred = port.bridge.translate(state)
    session = port.target.session(transferred)
    answer = session.ask("What is the project codename?")
    print(answer.text)
```

For a managed conversation, use `port.composite.session()`. Backend selection changes setup requirements, not the `bridge.translate(state)` operation.

## Evaluation

This checkpoint has separate corrections for thinking enabled and disabled. Historical task-success and TTFT results for an earlier 4B-to-27B bridge are recorded in the [benchmarks](https://github.com/LatentPort/latentport/blob/main/docs/benchmarks.md). Those scores do not establish the quality of this checkpoint. See the linked benchmark records for the scope of native GGUF runtime checks.

## Limitations

This is a directional state transform, not a standalone language model or a reverse bridge. Compatibility is specific to the pinned model pair. Compacted and bounded-suffix resident source states are not supported for translation. Results from small task panels do not establish broad capability, and runtime smoke checks do not qualify answer quality across backends. Historical results from earlier bridge weights must not be attributed to this checkpoint.

## Artifacts and provenance

`latentport.json` records the model identities, artifact filenames and SHA-256 receipts; `bridge_config.json` describes the operation. Safetensors contain inference parameters and population statistics. `bridge/translator.F32.gguf` contains the corresponding native bridge tensors. Use immutable repository revisions and artifact hashes for reproducibility.

## License

LatentPort bridge weights and code are licensed under Apache-2.0; see [LICENSE](LICENSE). Base models and runtime dependencies retain their upstream licenses.
