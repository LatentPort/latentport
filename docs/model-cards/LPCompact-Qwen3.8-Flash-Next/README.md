---
library_name: latentport
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- kv-cache
- qwen
- latentport
---

# LPCompact-Qwen3.8-Flash-Next

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

[GitHub](https://github.com/LatentPort/latentport) | [PyPI](https://pypi.org/project/latentport/) | [Installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) | [API](https://github.com/LatentPort/latentport/blob/main/docs/api.md)

## Overview

This 196,865-parameter selector compacts a single unpadded text history from Qwen3.8-Flash-Next at revision `812d96e10a0dd82b69d749e68693698cf6c36daa`. It scores the BF16 attention value vectors at layers 3, 15 and 31. The included Transformers helper gathers complete four-token sparse-attention blocks, their indexer keys, their original RoPE positions, and any partial final block. The first and last 512 positions are protected. Recurrent, convolution and n-gram history states are copied into the child.

## Compatibility

Use LatentPort 0.4.4 or later with Transformers 5.17.0 and the pinned BF16 Qwen/Qwen3.8-Flash-Next checkpoint. The standard model/state/session API supports a single unpadded text history. The BF16 loader can stream experts and n-gram embeddings from the pinned safetensors snapshot. GGUF and NF4 execution are not qualified for this integration.

## Usage

Install the runtime and backend dependencies using the installation guide above. Select the base model; compaction uses the same operation sequence as other supported models.

```python
import latentport as lp

history = "The project codename is Juniper."  # Replace with your long context.
with lp.load_model("Qwen/Qwen3.8-Flash-Next", backend="bf16") as model:
    state = model.prefill(history)
    compacted = state.compact()
    answer = model.session(compacted).ask("What is the project codename?")
    print(answer.text)
```

The runtime resolves the pinned selector automatically and preserves logical continuation positions. Ratios 4 and 8 are supported, starting at 65,536 physical positions; short histories are unchanged by default. Flash-Next retains whole sparse-attention blocks, indexer keys, original positions and recurrent, convolution and n-gram states.

## Evaluation

Both conditions used identical histories and questions. A success requires every canonical answer token and a valid generation stop token to be the greedy predictions given the correct preceding tokens. There are two held-out histories per task and length, with eight questions per history. Lost and gained count paired changes after compaction.

| Task | History tokens | Full memory | 8Ã— compact | Lost | Gained |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repository function lookup | 65,536 | 16/16 | 16/16 | 0 | 0 |
| Repository function lookup | 131,072 | 16/16 | 16/16 | 0 | 0 |
| Latest decision | 65,536 | 16/16 | 15/16 | 1 | 0 |
| Latest decision | 131,072 | 16/16 | 16/16 | 0 | 0 |
| Structured record lookup | 65,536 | 16/16 | 16/16 | 0 | 0 |
| Structured record lookup | 131,072 | 16/16 | 16/16 | 0 | 0 |
| Two-hop document lookup | 65,536 | 16/16 | 16/16 | 0 | 0 |
| Two-hop document lookup | 131,072 | 15/16 | 16/16 | 0 | 1 |

These measurements cover synthetic retrieval tasks and one training seed.

## Limitations

Measurements cover synthetic retrieval tasks and one training seed. The answer-token check uses correct preceding tokens and is not an unrestricted generation benchmark. Compaction is lossy: one latest-decision answer was lost at 64K. Qualification is limited to the pinned BF16 text cache; do not infer support for batching, padding, multimodal histories or other backends.

## Artifacts and provenance

The repository provides the 196,865-parameter selector and a standalone Transformers helper. Use the package selector configuration for the immutable repository revision and weight checksum. Base-model weights are separate.

The frozen BF16 teacher ran with exact expert weights offloaded to NVMe and a sharded n-gram lookup on one RTX PRO 6000. Record-label warm-up used 16 synthetic 16K histories and 256 AdamW updates. Utility fitting used 32 independent histories at 16K, 32K, 64K and 128K, 32 study questions per history and 1,024 updates. Utility targets estimate the effect of deleting a visible value from a sparse attention head.

## License

The selector and LatentPort code are licensed under Apache-2.0; see [LICENSE](LICENSE). Base-model weights and runtime dependencies retain their upstream licenses.
