---
license: apache-2.0
base_model:
- Qwen/Qwen3.5-4B
- Qwen/Qwen3.8-27B
language:
- en
tags:
- latentport
- gguf
- hybrid-composite
- state-transfer
- research-preview
---
# LatentPort-4+27B

**Small-model prefill. Large-model generation.**

[Website](https://latentport.com) · [Python runtime](https://github.com/LatentPort/latentport) · [Installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) · [Cookbook](https://github.com/LatentPort/latentport/blob/main/docs/cookbook.md)

LatentPort-4+27B is a hybrid composite of **Qwen3.5-4B** and **Qwen3.8-27B**.
The 4B reads the entire prompt, including the question. A learned bridge translates
its complete working state into the 27B's state space, so the 27B can generate without
processing the prompt itself. The transfer includes attention K/V, recurrent state,
convolution history and the terminal hidden state.

The runtime can retain the translated context for cached conversations, save it for
later, or fork it into independent sessions with different questions. In cached-handoff
mode, later questions are processed by the 27B; the original history is not replayed.
The initial supported direction is 4B → 27B.

## Artifacts

This is the GGUF distribution of the [canonical bridge](https://huggingface.co/latentport/latentport-4-27b-research-preview).

| Files | Contents |
| --- | --- |
| `latentport-4-27b-bridge.F32.gguf` | Complete bridge tensors in FP32; approximately 2.58 GB |
| `models/Qwen3.5-4B-Q4_K_M.gguf` | Tested reader quantization of the pinned upstream model |
| `models/Qwen3.8-27B-MTP-Q4_K_M.gguf` | Tested target quantization, including its trained MTP head |
| `models/*-LICENSE` | Upstream Qwen Apache-2.0 licenses |
| `runtime/` | Pinned Windows/CUDA native runtime, corresponding sources and dependency licenses |
| `tensor_map.json`, `FORMAT_VERIFICATION.json` | Tensor mapping and canonical-to-GGUF conversion verification |
| `latentport.json`, `SHA256SUMS.json` | Model identities, bridge metadata and file integrity |

**GGUF is the bridge container, not its quantization.** The bridge remains FP32; only
the language models use Q4_K_M. The custom `latentport_bridge_v1` architecture cannot
generate on its own. Use the LatentPort-enabled llama.cpp backend through the Python package.

The GGUF bridge embeds the integrity hash of the original canonical artifact revision,
which remains pinned in `latentport.json`. Later card/license updates do not change its
neural weights or require re-encoding the bridge.

## Run the composite

[Download the Python preview wheel](https://huggingface.co/latentport/latentport-4-27b-research-preview-GGUF/resolve/1415eb9591dfaf15355d8bba5ed38f64792026e0/runtime/latentport-0.1.0.dev1-py3-none-any.whl). Install CUDA-enabled Torch 2.11.0 first,
then `pip install` the wheel. Full prerequisites are in the installation guide below.

Install the [LatentPort runtime](https://github.com/LatentPort/latentport/blob/main/docs/installation.md),
then install its pinned native backend with `python -m latentport install-native`.
The GGUF preview targets Windows x64, Python 3.11 and an RTX 5090 (32 GB), with CUDA 13.0
runtime libraries. Model files download automatically and are cached in `LATENTPORT_HOME`.

```python
import latentport as lp

with lp.load(backend="gguf") as port:
    port.warm()
    chat = port.composite.session()
    answer = chat.ask("Analyze this document: ...")
    print(answer.text, end="")
    while not answer.finished:
        answer = chat.generate(max_new_tokens=512)
        print(answer.text, end="")
```

Cached handoff is the default. Use `policy="strict"` to keep every new user input on the 4B. Strict multi-turn execution
synchronizes the completed answer back to the reader and retranslates its updated state.
A compatible runtime is required: neither Transformers nor a stock GGUF loader can
execute the bridge as a standalone language model.

## Composite results

The frozen LatentPort-4+27B bridge was evaluated against both native models in E120.
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

## License and model identity

The LatentPort bridge weights are released under **Apache-2.0**, including the unchanged
artifacts from release `e119-final16-b960c37364bc`. See [LICENSE](LICENSE).
Qwen base models retain their upstream Apache-2.0 licenses; native runtime dependencies
retain their respective licenses. This is a research preview for the pinned model pair,
not a promise of native-27B equivalence or compatibility with arbitrary models.
