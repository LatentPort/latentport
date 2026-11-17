---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: latentport
tags:
- kv-cache
- state-compaction
- qwen
---

# LPCompact-Qwen3.5-9B

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

[GitHub](https://github.com/LatentPort/latentport) | [PyPI](https://pypi.org/project/latentport/) | [Installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) | [API](https://github.com/LatentPort/latentport/blob/main/docs/api.md)

## Overview

A learned attention-memory compaction selector for **Qwen/Qwen3.5-9B**, exposed through `state.compact()`. It scores attention values at layers 3, 15 and 31. LatentPort retains selected keys and values while preserving recurrent state, convolution history and the logical continuation clock. It does not summarize text or replay historical tokens.

## Compatibility

Use LatentPort 0.4.2 or later with Python 3.11+, CUDA PyTorch and the chosen backend dependencies. The selector serves BF16, NF4 and supported GGUF weights of the compatible base checkpoint. Backend availability does not imply identical quality qualification; see Evaluation. For GGUF, supply `model_path` and a compatible native library. Exact model and selector pins are recorded in the runtime configuration.

## Usage

Install the runtime and backend dependencies using the installation guide above. Set `history` to the context you want to retain.

```python
import latentport as lp

history = "The project codename is Juniper."  # Replace with your long context.
with lp.load_model("Qwen/Qwen3.5-9B", backend="bf16") as model:
    state = model.prefill(history)
    compacted = state.compact()
    answer = model.session(compacted).ask("What is the project codename?")
    print(answer.text)
```

Short histories are unchanged by default. Normal activation starts at 65,536 physical positions. The default ratio is 8; pass `ratio=4` to retain more memory. The policy preserves the first and last 512 positions, with a 1,024-position floor. Use session-level capacity controls when reserving room for more input. The result supports append, save/load and fork.

## Evaluation

The frozen selector and uncompressed BF16 model answered the same eight probes on each of eight independent histories per length. The rapid evaluation uses a fixed subset of the original held-out evaluation seeds. Each of four task families has one familiar-format and one held-out-format history at each length. The final column counts answers solved by both models among those solved with full memory.

| History tokens | Full memory | 8× compaction | Full-memory answers retained |
| ---: | ---: | ---: | ---: |
| 65,536 | 49/64 | 50/64 | 47/49 |
| 131,072 | 48/64 | 48/64 | 48/48 |

| Task, both lengths combined | Full memory | 8× compaction | Lost after compaction | Gained after compaction |
| --- | ---: | ---: | ---: | ---: |
| Repository function lookup | 32/32 | 32/32 | 0 | 0 |
| Latest decision | 32/32 | 30/32 | 2 | 0 |
| Structured record lookup | 32/32 | 32/32 | 0 | 0 |
| Two-hop document lookup | 1/32 | 4/32 | 0 | 3 |

Results are from one training seed and these synthetic retrieval tasks. Exact answers depend on the base model and the history.

## Limitations

Compaction is lossy and task dependent. It reduces attention KV storage, not base-model weights or all peak working memory. A compacted source cannot currently be passed to a bridge. Quality measured on one model, backend or task panel must not be assumed for another. Results use one training seed and synthetic retrieval tasks.

## Artifacts and provenance

`model.safetensors` contains the selector, and `config.json` records architecture, policy and checkpoint fingerprints. Base-model weights are separate.

The frozen BF16 base supplied all features and attention-derived supervision. A record-label warm-up used 16 independent 16K histories and 256 AdamW updates. Utility training used 32 additional histories spanning 16K, 32K, 64K and 128K, 32 disjoint study questions per history and 1,024 updates. Labels estimate the change in a head's value-space output when a historical entry is removed. The deployed selector takes state tensors alone.

## License

The selector and LatentPort code are licensed under Apache-2.0; see [LICENSE](LICENSE). Qwen base models and runtime dependencies retain their upstream licenses.
