---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: latentport
tags:
  - latentport
  - state-combination
---

# LPCombine-Qwen3.5-4B

Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).

[LatentPort](https://latentport.com) · [Source](https://github.com/LatentPort/latentport) · [PyPI](https://pypi.org/project/latentport/)

## Overview

LPCombine combines two independently captured contexts into one resumable state.
It appends the right context to the left cache in that order. The trained readout proposes tokens from the first full-attention layer's keys and values. The model verifies the candidates against its early cache.

## Compatibility

Requires LatentPort 0.4.9 or later and the same model, weights, tokenizer, backend
and position policy for both context-ready input states.

| Backend | Retained tokens | Cache-only |
| --- | --- | --- |
| BF16 | Yes | Yes |
| NF4 | Yes | Yes |
| GGUF | Yes | — |

GGUF uses the native LatentPort runtime and a local `model_path`. The GPU test used the pinned [LM Studio Q4_K_M file](https://huggingface.co/lmstudio-community/Qwen3.5-4B-GGUF/blob/f9f88ac3e234be915e23811a6d28ea287bdb927e/Qwen3.5-4B-Q4_K_M.gguf).

## Usage

```sh
pip install "latentport[nf4]>=0.4.9"
```

```python
import latentport as lp

with lp.load_model("latentport/LPCombine-Qwen3.5-4B", backend="nf4") as model:
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

- **BF16** matched the next greedy token from native append in all 4 trials and passed snapshot save/load.
- **NF4** matched the next greedy token from native append in all 4 trials and passed snapshot save/load.
- **GGUF** matched the next greedy token from native append in all 4 trials and passed snapshot save/load.

A separate sample split HumanEval and GSM8K prompts across the two states. The control prefills the complete chat template in one pass.

- **NF4** matched 4 of 4 native greedy completions. Task scores were 1/4 for `combine()` and 1/4 for native prefill.
- **GGUF** matched 4 of 4 native greedy completions. Task scores were 1/4 for `combine()` and 1/4 for native prefill.

In cache-only trials, the right token metadata was removed before combining.
[Runtime validation](runtime-validation.json) records the per-trial token counts.

## Limitations

The operation replays the right context. It checks recovered tokens before
updating a session. The trained readout uses a dense right cache and native text positions.

## Artifacts and provenance

The FP32 readout was fit on 18,432 synthetic tokens with the pinned model frozen in NF4. The readout is stored in `model.safetensors`. [Profile and checkpoint pins](lpcombine.json) record the
[base revision](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a).

## License

LPCombine is released under Apache-2.0. Base-model weights are subject to
their upstream terms.
