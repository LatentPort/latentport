---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: latentport
tags:
  - latentport
  - state-combination
  - qwen3.8
  - gguf
---

# LPCombine-Qwen3.8-27b

Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).

[LatentPort](https://latentport.com) · [Source](https://github.com/LatentPort/latentport) · [PyPI](https://pypi.org/project/latentport/)

## Overview

LPCombine continues one Qwen3.8-27B cache with the context represented by
a second state and returns a new, resumable state. The order is left
followed by right. The operation keeps both input snapshots available.

The NF4 cache-only method uses a trained readout over layer 3 keys and
values to propose right-hand tokens. Qwen verifies the candidates against
the right-hand cache and appends the recovered tokens to the left state.
The retained-token method replays the recorded right-hand tokens through
the selected backend.

## Compatibility

Requires `latentport>=0.4.8`. Both inputs must be context-ready and
share the same model, weights, tokenizer, backend, state ABI, and position
policy. The Hub artifact contains the readout weights. The runtime loads
the pinned Qwen weights separately.

| Backend | Retained tokens | Trained cache-only readout |
| --- | --- | --- |
| BF16 | Yes | — |
| NF4 | Yes | Dense cache with native positions and the pinned checkpoint |
| GGUF | Yes | — |

GGUF uses LatentPort's native runtime and a compatible local model file.
The release smoke test used [Bartowski Q4_K_M](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/87410e7b9a5c21e6ca70c0c9cc5237de909fe621/Qwen3.8-27B-Q4_K_M.gguf).

## Usage

Install the extra for your backend.

```sh
pip install "latentport[nf4]>=0.4.8"
```

```python
import latentport as lp

with lp.load_model("latentport/LPCombine-Qwen3.8-27b", backend="nf4") as model:
    left = model.prefill("Nora belongs to team X.\n")
    right = model.prefill("Team X uses the passcode cedar.\n")
    combined = left.combine(right, model=model)
    with model.session(combined) as session:
        print(session.ask("What is the passcode for Nora's team?").text)
```

`model.combine(left, right)` returns a new state. `session.combine(right)`
installs the new state in the current session. Use `method="cache"` on NF4
to run the trained readout, including when the right snapshot has a
recorded token history. Use `model.live(retain_replay=True)` to record
token IDs when building a resident snapshot for BF16 or GGUF combination.

For BF16 use `latentport[bf16]` and `backend="bf16"`. For GGUF use
`latentport[gguf]`, install the native runtime with
`latentport install-native`, and pass `backend="gguf"` and
`model_path="Qwen3.8-27B-Q4_K_M.gguf"`. A CUDA PyTorch install is required
for the GPU runtime.

```sh
latentport combine left-state right-state combined-state --model-path /path/to/weights
```

For saved states, match the position and capacity settings used at capture.

## Evaluation

The NF4 research verifier recovered all 9,504 right-side tokens in the
held-out join suite. It matched exact append on 122 of 128 three-way
interventions and 72 of 72 two-turn recall trials. In a short coding and
math comparison, the research bridge recovered all 1,302 right-side
tokens and matched the native completion in 16 of 16 concise-output
trials. Both paths passed 6 of 8 HumanEval cases and answered 3 of 8 GSM8K
cases correctly. One HumanEval generation reached the output cap.

The research verifier used a wider search for rare tokens. The released
runtime uses one readout and rejects a token it cannot verify. See
[`runtime-validation.json`](runtime-validation.json) for release checks
and [`evaluation.json`](evaluation.json) for experiment details.

## Limitations

Combination requires replay of the right side. Latency grows with that
context, with additional verification time in cache-only mode. Insert
any desired separator before capture. Only text caches are supported by
the trained readout. A compacted right snapshot needs retained tokens
for replay.

## Artifacts and provenance

- `model.safetensors` contains the trained readout's FP32 weight and bias.
- `lpcombine.json` records the readout revision, SHA-256, and base model.
- `evaluation.json` and `runtime-validation.json` describe the evidence.
- Base checkpoint [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0).
- The readout was fit on 18,432 synthetic training tokens with Qwen frozen in NF4 on an NVIDIA L40S.

## License

The LPCombine readout and runtime are released under Apache-2.0. The base
Qwen checkpoint and external GGUF are subject to their upstream terms.
