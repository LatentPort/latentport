---
license: apache-2.0
base_model: Qwen/Qwen3-14B
library_name: latentport
tags:
- kv-cache
- state-compaction
- qwen
---

# LPCompact-Qwen3-14B

[LPCompact overview](https://latentport.com/lpcompact)

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

[GitHub](https://github.com/LatentPort/latentport) | [PyPI](https://pypi.org/project/latentport/) | [Installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) | [API](https://github.com/LatentPort/latentport/blob/main/docs/api.md)

## Overview

A learned attention-memory compaction selector for **Qwen/Qwen3-14B**, exposed through `state.compact()`. It scores attention values at layers 3, 15 and 31. LatentPort gathers selected keys and values in all forty attention layers while preserving original positions and the logical continuation clock.

## Compatibility

Use LatentPort 0.4.7 or later with Python 3.11+, CUDA PyTorch and the BF16 backend dependencies. The qualified path is single-sequence BF16 text inference with the pinned checkpoint and native position settings. The maximum total logical length is 40,960 tokens. The runtime configuration records the exact base and selector revisions.

## Usage

Install the runtime and backend dependencies using the installation guide above. Set `history` to the context you want to retain.

```python
import latentport as lp

history = "The project codename is Juniper."  # Replace with your long context.
with lp.load_model("Qwen/Qwen3-14B", backend="bf16") as model:
    state = model.prefill(history)
    compacted = state.compact()
    answer = model.session(compacted).ask("What is the project codename?")
    print(answer.text)
```

Short histories are unchanged by default. Normal activation starts at 32,768 physical positions. The default ratio is 8; pass `ratio=4` to retain more memory. The policy preserves the first and last 512 positions, with a 1,024-position floor. Use session-level capacity controls when reserving room for more input. The result supports append, save/load and fork.

## Evaluation

The 16K panel deliberately forces compaction below the release activation threshold; its answer loss is why normal activation starts at 32K. The frozen selector and uncompressed BF16 model answered the same eight probes on each of eight independent histories per length. The rapid evaluation uses a fixed subset of the original held-out evaluation seeds. Each of four task families has one familiar-format and one held-out-format history at each length. A correct answer requires all canonical answer tokens and a valid stop token under greedy decoding. The final column counts answers solved by both models among those solved with full memory.

| History tokens | Full memory | 8× compaction | Full-memory answers retained |
| ---: | ---: | ---: | ---: |
| 16,384 | 44/64 | 24/64 | 21/44 |
| 32,768 | 55/64 | 50/64 | 50/55 |

| Task, both lengths combined | Full memory | 8× compaction | Lost after compaction | Gained after compaction |
| --- | ---: | ---: | ---: | ---: |
| Repository function lookup | 32/32 | 23/32 | 9 | 0 |
| Latest decision | 32/32 | 18/32 | 14 | 0 |
| Structured record lookup | 26/32 | 25/32 | 3 | 2 |
| Two-hop document lookup | 9/32 | 8/32 | 2 | 1 |

Results are from one training seed and these synthetic retrieval tasks. Exact answers depend on the base model and the history.

## Limitations

Compaction is lossy and task dependent. It reduces attention KV storage, not base-model weights or all peak working memory. A compacted source cannot currently be passed to a bridge. Quality measured on one model, backend or task panel must not be assumed for another. Results use one training seed and synthetic retrieval tasks.

## Artifacts and provenance

`model.safetensors` contains the selector, and `config.json` records architecture, policy and checkpoint fingerprints. Base-model weights are separate.

The frozen BF16 base supplied all features and attention-derived supervision. A record-label warm-up used 16 independent 16K histories and 256 AdamW updates. Utility training used 32 additional histories spanning 4K, 8K, 16K and 32K, 32 disjoint study questions per history and 1,024 updates. Labels estimate the change in a head's value-space output when a historical entry is removed. The deployed selector takes state tensors alone.

## License

The selector and LatentPort code are licensed under Apache-2.0; see [LICENSE](LICENSE). Qwen base models and runtime dependencies retain their upstream licenses.
