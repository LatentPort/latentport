---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: latentport
tags:
- latentport
- state-compaction
- kv-cache
- qwen
- qwen3.8
- gguf
---

# LPCompact-Qwen3.8-27B

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

[GitHub](https://github.com/LatentPort/latentport) | [PyPI](https://pypi.org/project/latentport/) | [Installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) | [API](https://github.com/LatentPort/latentport/blob/main/docs/api.md)

## Overview

A learned attention-memory compaction selector for **Qwen/Qwen3.8-27B**, exposed through `state.compact()`. It scores attention values at layers 3, 15 and 31. LatentPort retains selected keys and values while preserving recurrent state, convolution history and the logical continuation clock. It does not summarize text or replay historical tokens.

## Compatibility

Use LatentPort 0.4.2 or later with Python 3.11+, CUDA PyTorch and the chosen backend dependencies. The selector serves BF16, NF4 and supported GGUF weights of the compatible base checkpoint. Backend availability does not imply identical quality qualification; see Evaluation. For GGUF, supply `model_path` and a compatible native library. Exact model and selector pins are recorded in the runtime configuration.

## Usage

Install the runtime and backend dependencies using the installation guide above. Set `history` to the context you want to retain.

```python
import latentport as lp

history = "The project codename is Juniper."  # Replace with your long context.
with lp.load_model("Qwen/Qwen3.8-27B", backend="bf16") as model:
    state = model.prefill(history)
    compacted = state.compact()
    answer = model.session(compacted).ask("What is the project codename?")
    print(answer.text)
```

Short histories are unchanged by default. Normal activation starts at 65,536 physical positions. The default ratio is 8; pass `ratio=4` to retain more memory. The policy preserves the first and last 512 positions, with a 1,024-position floor. Use session-level capacity controls when reserving room for more input. The result supports append, save/load and fork.

## Evaluation

Runtime qualification passed at 64K and 128K with BF16, Q4_K_M and Q8_0, including continued generation, save/load, forks and repeated compaction. At 128K, 8x selection reduces BF16 attention tensor storage from 8 GiB to 1 GiB. See the [validation record](https://github.com/LatentPort/latentport/blob/main/docs/compaction-validation.json). These structural and continuation checks are not a broad answer-quality benchmark.

## Limitations

Compaction is lossy and task dependent. It reduces attention KV storage, not base-model weights or all peak working memory. A compacted source cannot currently be passed to a bridge. Quality measured on one model, backend or task panel must not be assumed for another. Repeated-operation quality and broader workloads need separate evaluation.

## Artifacts and provenance

`model.safetensors` contains the selector, and `config.json` records architecture, policy and checkpoint fingerprints. Base-model weights are separate.

The selector has 393,473 FP32 parameters: 3,072 input features, a 128-unit hidden layer with GELU and one output score. The `v0.1` tag identifies the original checkpoint.

## License

The selector and LatentPort code are licensed under Apache-2.0; see [LICENSE](LICENSE). Qwen base models and runtime dependencies retain their upstream licenses.
