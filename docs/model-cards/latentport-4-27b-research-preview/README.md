---
license: apache-2.0
base_model:
- Qwen/Qwen3.5-4B
- Qwen/Qwen3.8-27B
language:
- en
tags:
- latentport
- hybrid-composite
- state-transfer
- research-preview
---

# latentport-4-27b-research-preview

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

[GitHub](https://github.com/LatentPort/latentport) | [PyPI](https://pypi.org/project/latentport/) | [Installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) | [API](https://github.com/LatentPort/latentport/blob/main/docs/api.md)

## Overview

An earlier research-preview bridge from **Qwen3.5-4B** to **Qwen3.8-27B**, retained with its original evaluation evidence. It transfers attention, recurrent, convolution and terminal hidden state. For current package use, select the [canonical LPBridge repository](https://huggingface.co/latentport/LPBridge-Qwen3.5-4b-Qwen3.8-27b).

## Compatibility

The preview is tied to the exact pair and runtime recorded in its manifest. It is not the current package default. Its original GGUF preview targeted Windows x64, Python 3.11, CUDA 13.0 and NVIDIA RTX 5090. Do not infer compatibility with arbitrary checkpoints or stock GGUF loaders.

## Usage

For new applications, use the current package and canonical bridge with the standard API:

```python
import latentport as lp

with lp.load("latentport/LPBridge-Qwen3.5-4b-Qwen3.8-27b", backend="nf4", thinking=False) as port:
    state = port.source.capture([("user", "The project codename is Juniper.")])
    transferred = port.bridge.translate(state)
    session = port.target.session(transferred)
    answer = session.ask("What is the project codename?")
    print(answer.text)
```

This example loads the current bridge, not the archived preview evaluated below. Reproducing preview measurements requires its original pinned runtime and artifacts.

## Evaluation

The frozen LatentPort-4+27B bridge was evaluated against both native models in the published composite evaluation.
The 4B reads the **entire prompt, including the question**; the 27B processes **zero prompt tokens**.

| Completed tasks | Native 4B | LatentPort-4+27B | Native 27B |
| --- | ---: | ---: | ---: |
| Panel A (64 tasks) | 9 / 64 | **26 / 64** | 29 / 64 |
| Independent confirmation B (64 tasks) | 11 / 64 | **23 / 64** | 29 / 64 |

On confirmation B, the composite improved strict task success by **18.75 percentage points**
over native 4B (paired 95% interval: +0.74 to +33.92 points). Native 27B remained stronger overall.
These are adapted public and authored tasks, not official leaderboard scores.

### Time to first token

| Prompt length | Native 27B median | Composite median | Paired TTFT speedup (95% CI) |
| --- | ---: | ---: | ---: |
| Short | 0.603 s | 0.513 s | 1.06x (0.87–1.29) |
| ~2K | 2.148 s | 1.295 s | **1.65x** (1.61–1.69) |
| ~8K | 8.871 s | 4.459 s | **2.01x** (1.96–2.05) |
| ~16K | 24.663 s | 11.087 s | **2.19x** (2.15–2.23) |

Across all 32 requests, paired TTFT speedup was **1.67x** (95% CI: 1.49–1.86).
Eight independent requests per length, three paired timing repetitions per request;
ratios are paired geometric estimates, not ratios of the displayed medians.

**Configuration:** RTX 5090, pinned NF4/BF16 research runtime, resident models and bridge.
TTFT includes source prefill, state translation, installation and first emitted token;
it excludes cold loading. A first emitted token may be a reasoning/control token, not
useful answer text. These results do not establish faster full answers or decode throughput.
Quality scores have not been re-established for the GGUF runtime. Prose grading was
identity-blinded AI grading; template differences and ambiguous judgments limit generalization.

Machine-readable composite results and evidence fingerprints: [COMPOSITE_RESULTS.json](COMPOSITE_RESULTS.json).
The evaluation above uses the NF4/BF16 research configuration; it must not be presented
as a GGUF quality benchmark. Runtime correctness checks and research task evaluation
serve different purposes.

## Limitations

Preview results apply only to the evaluated checkpoint and configuration. NF4/BF16 task scores are not GGUF quality scores and do not describe the current canonical bridge. Warm TTFT excludes cold loading and does not establish full-answer speedups.

## Artifacts and provenance

The repository retains its preview manifests, bridge artifacts and `COMPOSITE_RESULTS.json`. Exact revisions and SHA-256 receipts identify the weights independently of descriptive release labels. The GGUF companion contains native-format artifacts; neither repository is a standalone language model.

## License

The bridge weights are licensed under Apache-2.0, including the unchanged preview artifacts and earlier published revisions; see [LICENSE](LICENSE). Upstream base models and native dependencies retain their respective licenses.
