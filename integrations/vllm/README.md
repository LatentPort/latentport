# LPCompact for vLLM

[LPCompact for vLLM quickstart and GPU validation](https://latentport.com/integrations/vllm).

This integration serves the LPCompact text models on NVIDIA GPUs. It pins the
served checkpoint and downloads the selector directly from Hugging Face.
Serving uses BF16 compute and KV. For an offline deployment, prewarm the base checkpoint's
Hugging Face cache and LatentPort's selector cache under
`LATENTPORT_HOME/huggingface`, or supply `--lpcompact-weights-path` with the pinned
selector file.

## Weight formats

All seven [base model profiles](#model-profiles) support BF16, NF4, and GGUF
model weights. Each base model uses the same LPCompact selector across these
formats. Compute and the KV cache stay in BF16.

| Weight format | Launcher option | Setup |
| --- | --- | --- |
| BF16 | `--weight-format bf16` (default) | The stock image loads the pinned Hugging Face checkpoint. |
| NF4 | `--weight-format nf4` | Install the NF4 extra. The loader quantizes the pinned checkpoint on load with one GPU rank per replica. |
| GGUF | `--weight-format gguf` | Install the GGUF extra and pinned vLLM GGUF plugin. The loader uses the GGUF file specified for the base model in the profile table. |

NF4 and GGUF are experimental weight-loading modes with GPU qualification
pending. The [L40S quality and resident-session benchmarks](benchmarks/l40s/README.md)
used Qwen3.5-9B BF16 weights.

## Installation

### Docker quickstart

The NVIDIA image includes the pinned vLLM 0.34.0 installation and the LPCompact
patch. Run it with the NVIDIA Container Toolkit.

```sh
docker pull ghcr.io/latentport/latentport-vllm:0.2.0
docker run --gpus all --ipc=host -p 8000:8000 ghcr.io/latentport/latentport-vllm:0.2.0 \
  serve Qwen/Qwen3.8-27B \
  --lpcompact latentport/LPCompact-Qwen3.8-27B --lpcompact-ratio 8
```

To build from the repository root, run
`docker build -f integrations/vllm/Dockerfile -t latentport-vllm:dev .`.

For NF4 or GGUF in Docker, extend the stock image and install the corresponding
dependencies from the [quantized weight setup](#nf4-and-gguf-weights) into the
same Python environment as vLLM. The stock image already includes the
LPCompact patch.

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
installed, install the adapter from this checkout and apply its bundled patch once.

```sh
python -m pip install ./integrations/vllm
vllm_site="$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm").origin).parent.parent)')"
patch="$(python -c 'from importlib.resources import files; print(files("latentport_vllm") / "patches" / "vllm-0.34.0.patch")')"
(cd "$vllm_site" && git apply --check "$patch" && git apply "$patch")
latentport-vllm serve Qwen/Qwen3.8-27B --lpcompact latentport/LPCompact-Qwen3.8-27B
```

### NF4 and GGUF weights

Use `--weight-format bf16` for the default Hugging Face weights. Install the
NF4 extra in the same environment to quantize those pinned weights on load.
NF4 uses one GPU rank per model replica.

```sh
python -m pip install './integrations/vllm[nf4]'
latentport-vllm serve Qwen/Qwen3.8-27B --weight-format nf4 \
  --lpcompact latentport/LPCompact-Qwen3.8-27B
```

For GGUF, install the pinned vLLM GGUF plugin with a CUDA toolkit that matches
the vLLM environment. The launcher selects the pinned GGUF file for the base
model and uses the base model's pinned config and tokenizer. For a local copy,
pass `--gguf-path` with the original filename of the first part. Keep all parts
in the same directory.

```sh
python -m pip install './integrations/vllm[gguf]' 'setuptools>=77,<81' wheel ninja
python -m pip install --no-build-isolation --no-deps \
  'vllm-gguf-plugin @ git+https://github.com/vllm-project/vllm-gguf-plugin.git@497542d12327a67f806345438238ea702cadd824'
latentport-vllm serve Qwen/Qwen3.8-27B --weight-format gguf \
  --lpcompact latentport/LPCompact-Qwen3.8-27B
```

Flash-Next stages n-gram rows from disk in NF4 and GGUF mode.

## Model profiles

Use a base model and its selector from the same row.

| Base checkpoint | Hugging Face selector | Pinned GGUF format | Default trigger |
| --- | --- | --- | ---: |
| `Qwen/Qwen3.5-4B` | `latentport/LPCompact-Qwen3.5-4B` | Q4_K_M | 65,536 |
| `Qwen/Qwen3.5-9B` | `latentport/LPCompact-Qwen3.5-9B` | Q4_K_M | 65,536 |
| `Qwen/Qwen3-14B` | `latentport/LPCompact-Qwen3-14B` | Q4_K_M | 32,768 |
| `Qwen/Qwen3.6-35B-A3B` | `latentport/LPCompact-Qwen3.6-35B-A3B` | UD-Q4_K_M | 65,536 |
| `Qwen/Qwen3.8-27B` | `latentport/LPCompact-Qwen3.8-27B` | Q4_K_M | 65,536 |
| `Qwen/Qwen3.8-Flash-Next` | `latentport/LPCompact-Qwen3.8-Flash-Next` | Q2_K | 65,536 |
| `google/gemma-4-31B-it` | `latentport/LPCompact-Gemma-4-31B-it` | Q4_K_M | 65,536 |

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

## Resident sessions

Set `VLLM_API_KEY` or pass vLLM's `--api-key`, then add
`--lpcompact-max-sessions COUNT` to the launcher. Sessions use one Python API
server. Each idle session retains its attention pages and any recurrent
state on the GPU until deletion or idle expiry. The default TTL is 900
seconds and can be set with `--lpcompact-session-ttl`. A server restart clears
all sessions.

Use the same bearer key for each request in the session.

| Method and path | Purpose |
| --- | --- |
| `POST /v1/latentport/sessions` | Create a session and get its `id` and `next_turn`. |
| `POST /v1/latentport/sessions/{id}/turns` | Send `turn`, `token_ids` and optional `max_tokens`. Returns generated `text`, `token_ids`, `finish_reason` and the updated `logical_tokens`. |
| `GET /v1/latentport/sessions/{id}` | Read the next turn number, token count, busy status and most recent completed turn. |
| `DELETE /v1/latentport/sessions/{id}` | End the session and release its pages. |

Send the tokenized prompt on turn 1 and only the new suffix on later turns.
Use the pinned model's tokenizer and chat delimiters. The server includes
the prior reply's final sampled token in the next prefill. Retrieving the
latest turn with `GET` is safe after a client timeout. An old turn number or
an overlapping writer returns HTTP 409, and a full session quota returns 429.
Ordinary generation shares the same sequence slots, including the slots
occupied by idle sessions.

For sessions, startup reserves each slot's upper bound on physical attention
pages, plus the full budgets for recurrent and sliding-window state and one
staged copy. `--max-num-batched-tokens` controls how far occupancy can grow
between compaction checks. The configured logical context limit still applies
to the sum of all turns in a session.

See the [L40S benchmark](benchmarks/l40s/README.md) for paired answer scores,
timings and resident-session capacity.

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
