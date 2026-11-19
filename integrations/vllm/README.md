# LPCompact for vLLM

This integration serves the LPCompact text models on NVIDIA GPUs. It pins the
served checkpoint and downloads the selector directly from Hugging Face.
Serving uses BF16 compute and KV. For an offline deployment, prewarm the base checkpoint's
Hugging Face cache and LatentPort's selector cache under
`LATENTPORT_HOME/huggingface`, or supply `--lpcompact-weights-path` with the pinned
selector file.

Build from the repository root with vLLM 0.34.0's upstream image.

```sh
docker build -f integrations/vllm/Dockerfile -t latentport-vllm:dev .
docker run --gpus all --ipc=host -p 8000:8000 latentport-vllm:dev \
  serve Qwen/Qwen3.8-27B \
  --lpcompact latentport/LPCompact-Qwen3.8-27B --lpcompact-ratio 8
```

For Flash-Next on a single RTX PRO 6000, the launcher can use the pinned Inferact
NVFP4 checkpoint with disk-backed embedding lookups. Put the Hugging Face cache
on fast local NVMe and mount it into the container.

```sh
docker run --gpus all --ipc=host -p 8000:8000 \
  -e LATENTPORT_HOME=/root/.cache \
  -v /path/to/huggingface:/root/.cache/huggingface latentport-vllm:dev \
  serve Inferact/Qwen3.8-Flash-Next-NVFP4 \
  --lpcompact latentport/LPCompact-Qwen3.8-Flash-Next
```

The same selector is used with the `Qwen/Qwen3.8-Flash-Next` BF16 profile. Its full
checkpoint needs substantially more memory and still requires full-weight
qualification. See the
[vLLM hardware recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) for
tested base-model hardware combinations.

### Install in an existing vLLM environment

In a dedicated Linux environment with vLLM 0.34.0 and its CUDA dependencies
installed, install the adapter and apply its bundled patch once.

```sh
python -m pip install latentport-vllm==0.1.0
vllm_site="$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm").origin).parent.parent)')"
patch="$(python -c 'from importlib.resources import files; print(files("latentport_vllm") / "patches" / "vllm-0.34.0.patch")')"
(cd "$vllm_site" && git apply --check "$patch" && git apply "$patch")
latentport-vllm serve Qwen/Qwen3.8-27B --lpcompact latentport/LPCompact-Qwen3.8-27B
```

## Model profiles

Use a base model and its selector from the same row.

| BF16 base checkpoint | Hugging Face selector | Default trigger |
| --- | --- | ---: |
| `Qwen/Qwen3.5-4B` | `latentport/LPCompact-Qwen3.5-4B` | 65,536 |
| `Qwen/Qwen3.5-9B` | `latentport/LPCompact-Qwen3.5-9B` | 65,536 |
| `Qwen/Qwen3-14B` | `latentport/LPCompact-Qwen3-14B` | 32,768 |
| `Qwen/Qwen3.6-35B-A3B` | `latentport/LPCompact-Qwen3.6-35B-A3B` | 65,536 |
| `Qwen/Qwen3.8-27B` | `latentport/LPCompact-Qwen3.8-27B` | 65,536 |
| `Qwen/Qwen3.8-Flash-Next` | `latentport/LPCompact-Qwen3.8-Flash-Next` | 65,536 |
| `google/gemma-4-31B-it` | `latentport/LPCompact-Gemma-4-31B-it` | 65,536 |

The launcher limits Qwen3-14B to its native 40,960 positions by default.
Gemma 4 compacts its global attention layers and keeps its sliding-window
cache on the logical token timeline.

The launcher selects eager, synchronous, text-only serving with hybrid cache
management enabled. Startup checks require BF16 cache, no automatic prefix
caching, and sufficient pages for the configured maximum sequence length,
concurrency, and one staged compaction. Lower `--max-num-seqs` or
`--max-model-len` if the configuration exceeds cache capacity. The trigger
and supported 4×/8× ratios follow each selector's manifest.

Each committed event logs the artifact and its revision, the logical token
count, attention rows before and after, elapsed time, and net pages returned
to the vLLM pool. The attention-row ratio describes the request's attention
cache. GPU allocation also includes weights, recurrent state, and the
remaining cache pool.

The shared scheduler transaction handles capacity and page ownership.
The worker selects the model-specific cache views from the pinned profile.
Flash-Next compacts whole QSA groups, moving the compressed index and K/V
together while leaving the raw-key ring and recurrent state in place.

Use `tests/gpu_smoke.py` against a local test server to check generated tokens,
prompt accounting, and allocator page reclamation. On a single RTX PRO 6000
Blackwell Server Edition with `--max-num-batched-tokens 2048`, the real-weight
8× smoke runs observed:

| Checkpoint | Attention rows | Allocator pages reclaimed | Generated after compaction |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B BF16 | 65,536 → 8,192 | 99 | 16 tokens |
| Qwen3.5-9B BF16 | 65,536 → 8,192 | 99 | 16 tokens |
| Qwen3-14B BF16 | 32,768 → 4,096 | 448 | 16 tokens |
| Qwen3.6-35B-A3B BF16 | 65,536 → 8,192 | 53 | 16 tokens |
| Qwen3.8-27B BF16 | 131,072 → 16,384 | 138 | 16 tokens |
| Flash-Next NVFP4 | 65,536 → 8,192 | 35 | 16 tokens |
| Gemma 4 31B IT BF16 | 65,536 → 8,192 | 448 | 16 tokens |

With the default prefill batch size on that GPU, Qwen3.6-35B-A3B also passed
at 8×. Gemma 4 served a 114,707-token prompt, compacted twice at 4×, returned
384 pages to the allocator each time, and generated 16 tokens.

The Flash-Next fixture also exercised repeated compaction and an incomplete
four-token QSA group with the real GPU cache kernels and dummy weights.

Two synthetic archive lookups per checkpoint used identical 65,552-token
prompts and greedy decoding in dense and compacted serving. The record
appeared near the beginning or the middle of the archive.

| Checkpoint | Exact answers dense / 8× | Median first-token latency dense / 8× |
| --- | ---: | ---: |
| Qwen3.8-27B BF16 | 2/2 / 2/2 | 12.24 s / 12.31 s |
| Flash-Next NVFP4 | 2/2 / 2/2 | 5.65 s / 5.49 s |

These two-prompt runs check continuation and report the observed latency for
that workload.

Profiles use the core model and selector catalog for checkpoint identity,
feature layers, KV head count, and positions. A new cache layout needs a worker
binding for its attention and auxiliary cache views.
