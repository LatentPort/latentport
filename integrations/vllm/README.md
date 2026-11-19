# LPCompact for vLLM

This integration serves text on NVIDIA GPUs with the published LPCompact
selectors for Qwen3.8-27B and Qwen3.8-Flash-Next. It pins the served checkpoint and
downloads the selector directly from Hugging Face. Both model families use BF16
compute and KV. For an offline deployment, prewarm the base checkpoint's
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

The launcher selects eager, synchronous, text-only serving with hybrid cache
management enabled. Startup checks require BF16 cache, no automatic prefix
caching, and sufficient pages for the configured maximum sequence length,
concurrency, and one staged compaction. Lower `--max-num-seqs` or
`--max-model-len` if the configuration exceeds cache capacity. The default
65,536-row trigger and 4×/8× ratios follow each selector's manifest.

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
Blackwell Server Edition, the real-weight 8× smoke runs observed:

| Checkpoint | Attention rows | Allocator pages reclaimed | Generated after compaction |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B BF16 | 131,072 → 16,384 | 138 | 16 tokens |
| Flash-Next NVFP4 | 65,536 → 8,192 | 35 | 16 tokens |

The Flash-Next fixture also exercised repeated compaction and an incomplete
four-token QSA group with the real GPU cache kernels and dummy weights.

For a new model with the
same cache layout, add its checkpoint and selector identity to the profile
table and qualify that pair. New cache layouts also need a worker binding for
their feature, attention, and auxiliary cache views.
