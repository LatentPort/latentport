---
license: apache-2.0
base_model: google/gemma-4-31B-it
library_name: latentport
tags:
  - latentport
  - state-combination
---

# LPCombine-Gemma-4-31B-it

Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).

[LatentPort](https://latentport.com) · [Source](https://github.com/LatentPort/latentport) · [PyPI](https://pypi.org/project/latentport/)

## Overview

LPCombine combines two independently captured contexts into one resumable state.
It appends the right context to the left cache in that order. The cache-only method matches the first attention layer's values to the pinned model's token projections and verifies the match.

## Compatibility

Requires LatentPort 0.4.10 or later and the same model, weights, tokenizer, backend
and position policy for both context-ready input states.

| Backend | Retained tokens | Cache-only |
| --- | --- | --- |
| BF16 | Yes | Yes |
| NF4 | Yes | — |
| GGUF | Yes | — |

GPU validation used the pinned [Q4_K_M checkpoint](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/c1ac76e99d5513b141e8adde7288b85c3f9c32ec). For GGUF, set `backend="gguf"` and provide `model_path`. Keep all shards of a split checkpoint together.

## Usage

```sh
pip install "latentport[bf16]>=0.4.10"
```

```python
import latentport as lp

with lp.load_model("latentport/LPCombine-Gemma-4-31B-it", backend="bf16") as model:
    left = model.prefill("Nora belongs to team X.\n")
    right = model.prefill("Team X uses the passcode cedar.\n")
    merged = left.combine(right, model=model, method="cache")
    with model.session(merged) as session:
        print(session.ask("What is the passcode for Nora's team?").text)
```

Use `method="cache"` to request cache-only decoding on a compatible backend.
Use `model.live(retain_replay=True)` when recording resident states for
retained-token replay. `session.combine(right)` installs the result in the
current session. `latentport combine LEFT RIGHT OUTPUT` accepts saved states.

## Evaluation

- **BF16** matched the next greedy token from native append in all 2 trials and passed snapshot save/load.
- **NF4** matched native next-token logits in all 2 trials and after snapshot save/load.
- **GGUF** matched native next-token logits in all 2 trials and after snapshot save/load.

A separate sample split HumanEval and GSM8K prompts across the two states. The control prefills the complete chat template in one pass.

- **BF16** matched 2 of 2 native greedy completions. Task scores were 2/2 for `combine()` and 2/2 for native prefill.

In cache-only trials, the right token metadata was removed before combining.
[Runtime validation](runtime-validation.json) records the per-trial token counts.

## Limitations

The operation replays the right context. It checks recovered tokens before
updating a session. Cache-only replay requires every token to remain in the first sliding-attention layer. Use retained tokens once that window has dropped older positions.

## Artifacts and provenance

The cache-only codebook is derived from the pinned base checkpoint at runtime. [Profile and checkpoint pins](lpcombine.json) record the
[base revision](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475).

## License

LPCombine is released under Apache-2.0. Base-model weights are subject to
their upstream terms.
