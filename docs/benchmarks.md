# Benchmarks

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

## GGUF runtime checks (separate from E120)

Q4_K_M base models, FP32 bridge container, BF16 native KV, Windows RTX 5090:

- Initial ~16K warm TTFT: 2.367 s composite vs 5.582 s native 27B (2.36x).
- Cached-handoff follow-up after ~16K: 123 ms vs 127 ms for a persistent native 27B session.
- The follow-up consumed 23 new target prompt tokens, preserved the old prefix, used
  no reader synchronization or bridge translation, and recalled the planted code word.

These are bounded engineering samples, not a new quality benchmark or a statistical
speedup guarantee. The cached follow-up result establishes reuse, not a meaningful
speed advantage over native caching. Strict-mode follow-ups retranslate the whole state
and can be slower than native cached follow-ups.

Receipts: [initial and strict checks](runtime-validation.json),
[cached handoff](cached-handoff-validation.json). Warm request TTFT excludes downloads,
file hashing and model loading. Keep the port resident to measure request latency.
Do not compare a resident composite with a cold native model. Native prefix caching is
also available: multiple agents sharing one context save the first prefill, not N full
prefills against a properly cached native baseline.
