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

**LPCompact** is a learned state compaction model for **Qwen3.8-27B**,
available through LatentPort's `state.compact()` operation.

It selects useful attention-memory entries directly from a running model's
state. Recurrent state, convolution state and the logical continuation clock
are preserved. The operation uses no text summary or historical token replay.

## Model and runtime

This repository contains the trained compaction selector for Qwen3.8-27B:
one `model.safetensors` file, its configuration, model card and license.
Qwen base-model weights, including GGUF files, are obtained separately.

The shared Python runtime is maintained in
[LatentPort on GitHub](https://github.com/LatentPort/latentport).
Each trained state operation has its own Hugging Face repository, named by
operation and model compatibility. This model supplies compaction for
Qwen3.8-27B through that common runtime.

## One checkpoint across BF16 and GGUF

The same 393,473-parameter selector is used with Qwen3.8-27B BF16 and GGUF
weight quantizations. Base-model weight precision does not change the selector's
architecture or weights. LatentPort adapters supply the common BF16
attention-state layout.

Runtime qualification passed at 64K and 128K with BF16, Q4_K_M and Q8_0,
including continued generation, save/load, forks and repeated compaction.
See the [runtime validation record](https://github.com/LatentPort/latentport/blob/main/docs/compaction-validation.json).

## Use with LatentPort

Install [LatentPort from PyPI](https://pypi.org/project/latentport/) (0.2.1 or later)
and follow the [BF16 or GGUF setup instructions](https://github.com/LatentPort/latentport/blob/main/docs/compaction.md).

```python
import latentport as lp

with lp.load_model(backend="bf16") as model:
    state = model.prefill(history)
    state = state.compact()
    answer = model.session(state).ask("Continue the task.")
```

For GGUF, use `backend="gguf"` and supply `model_path` with the chosen
Qwen3.8-27B GGUF. Both formats automatically load the same selector.

The returned value is an ordinary immutable LatentPort State, supporting fork,
save/load, append and continued generation. The original snapshot remains valid.

## Compaction policy

The default policy retains one eighth of attention KV positions, with a
1,024-position floor. A 4x setting is also available. Normal activation starts
at 65,536 physical KV positions. At 128K, 8x selection reduces BF16 attention
tensor storage from 8 GiB to 1 GiB. Base weights and fixed recurrent state retain
their existing representation.

The selector reads three attention value layers and assigns a utility score to
each retained position. Its span policy preserves the first and last 512
positions and selects the remaining memory within the requested budget.

## Artifact and version

The compaction selector uses FP32 weights, a 3,072-feature input, a 128-unit hidden
layer with GELU, and one output score. `config.json` records its architecture,
policy and hashes. The `v0.1` tag identifies the original model checkpoint.
The model weights are unchanged by the repository rename.

Canonical repository: [latentport/LPCompact-Qwen3.8-27B](https://huggingface.co/latentport/LPCompact-Qwen3.8-27B).
