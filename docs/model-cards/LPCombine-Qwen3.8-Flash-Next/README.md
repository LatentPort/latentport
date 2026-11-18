---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
library_name: latentport
tags:
  - latentport
  - state-combination
---

# LPCombine-Qwen3.8-Flash-Next

Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).

[LatentPort](https://latentport.com) · [Source](https://github.com/LatentPort/latentport) · [PyPI](https://pypi.org/project/latentport/)

## Overview

LPCombine combines two independently captured contexts into one resumable state.
It appends the right context to the left cache in that order. The trained readout proposes tokens from the first full-attention layer's values and routing keys. Flash-Next verifies the candidates by evaluating its early layers.

## Compatibility

Requires LatentPort 0.4.9 or later and the same model, weights, tokenizer, backend
and position policy for both context-ready input states.

| Backend | Retained tokens | Cache-only |
| --- | --- | --- |
| BF16 | Yes | Yes |

## Usage

```sh
pip install "latentport[bf16]>=0.4.9"
```

```python
import latentport as lp

with lp.load_model("latentport/LPCombine-Qwen3.8-Flash-Next", backend="bf16") as model:
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

In cache-only trials, the right token metadata was removed before combining.
[Runtime validation](runtime-validation.json) records the per-trial token counts.

## Limitations

The operation replays the right context. It checks recovered tokens before
updating a session. The verifier re-evaluates short candidate prefixes. For long contexts, use retained tokens to avoid that additional work.

## Artifacts and provenance

The FP32 readout was fit on 24,576 synthetic tokens against the pinned BF16 model. The readout is stored in `model.safetensors`. [Profile and checkpoint pins](lpcombine.json) record the
[base revision](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/812d96e10a0dd82b69d749e68693698cf6c36daa).

## License

LPCombine is released under Apache-2.0. Base-model weights are subject to
their upstream terms.
