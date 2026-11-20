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

[LPCombine overview](https://latentport.com/#combine) · [Source](https://github.com/LatentPort/latentport) · [PyPI](https://pypi.org/project/latentport/)

## Overview

LPCombine combines two independently captured contexts into one resumable state.
It appends the right context to the left cache in that order.

With LatentPort 0.4.11, new snapshots store input-embedding activations.
Cache-only combine matches those activations against the active model's
embedding table and replays the recovered right-hand context. Capture the
right state with `model.live()` to omit retained replay tokens.

## Compatibility

Requires LatentPort 0.4.11 or later and the same model, weights, tokenizer, backend
and position policy for both context-ready input states.

| Backend | Retained tokens | Cache-only |
| --- | --- | --- |
| BF16 | Yes | Yes |
| NF4 | Yes | Yes |
| GGUF | Yes | Yes |

GPU validation used the pinned [Q4_K_M checkpoint](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/c1ac76e99d5513b141e8adde7288b85c3f9c32ec). For GGUF, set `backend="gguf"` and provide `model_path`. Keep all shards of a split checkpoint together.

## Usage

Install the extra for your backend (`bf16`, `nf4`, or `gguf`).

```sh
pip install "latentport[bf16]>=0.4.11"
```

```python
import latentport as lp

with lp.load_model("latentport/LPCombine-Gemma-4-31B-it", backend="bf16") as model:
    left = model.prefill("Nora belongs to team X.\n")
    with model.live() as cache:
        cache.append("Team X uses the passcode cedar.\n")
        right = cache.snapshot()
    merged = left.combine(right, model=model, method="cache")
    with model.session(merged) as session:
        print(session.ask("What is the passcode for Nora's team?").text)
```

Use `method="cache"` to request cache-only decoding on any of the three backends.
Use `model.live(retain_replay=True)` when recording resident states for
retained-token replay. `session.combine(right)` installs the result in the
current session. `latentport combine LEFT RIGHT OUTPUT --method cache` accepts saved states.

## Evaluation

The 0.4.11 [cache-only validation](cache-only-validation.json) checks
exact recovery and native next-token logits after restoring a right snapshot
with its token and text metadata removed. Earlier profile results follow.

- **BF16** matched the next greedy token from native append in all 2 trials and passed snapshot save/load.
- **NF4** matched native next-token logits in all 2 trials and after snapshot save/load.
- **GGUF** matched native next-token logits in all 2 trials and after snapshot save/load.

A separate sample split HumanEval and GSM8K prompts across the two states. The control prefills the complete chat template in one pass.

- **BF16** matched 2 of 2 native greedy completions. Task scores were 2/2 for `combine()` and 2/2 for native prefill.

In cache-only trials, the right token metadata was removed before combining.
[Runtime validation](runtime-validation.json) records the per-trial token counts.

## Limitations

The operation replays the right context. Every stored activation must map
to a unique token in the active checkpoint. Recapture older snapshots to
get the embedding cache. Insert any desired separator before capture.

## Artifacts and provenance

The cache-only codebook is derived from the pinned base checkpoint at runtime. [Profile and checkpoint pins](config.json) record the
[base revision](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475).

## License

LPCombine is released under Apache-2.0. Base-model weights are subject to
their upstream terms.
