---
library_name: latentport
license: apache-2.0
base_model:
- Qwen/Qwen3.5-4B
- Qwen/Qwen3.5-9B
tags:
- state-transfer
- safetensors
---

# LPBridge-Qwen3.5-4b-Qwen3.5-9b

[LPBridge overview](https://latentport.com/lpbridge)

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

[GitHub](https://github.com/LatentPort/latentport) | [PyPI](https://pypi.org/project/latentport/) | [Installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) | [API](https://github.com/LatentPort/latentport/blob/main/docs/api.md)

## Overview

A directional learned bridge from **Qwen/Qwen3.5-4B** to **Qwen/Qwen3.5-9B**. It translates attention keys and values, convolution history, recurrent memory and terminal hidden state so the target can continue without prefilling the historical context. Base-model weights are obtained separately.

## Compatibility

Use LatentPort 0.4.2 or later with Python 3.11+, CUDA PyTorch and the dependencies for `bf16`, `nf4` or `gguf`. Exact source and target revisions are recorded in `latentport.json`. GGUF requires a compatible LatentPort native library and qualified model files. Use `thinking=False`. Capture the context prefix on the source; the target processes the new question after translation. The package composite API handles this boundary automatically.

## Usage

Install the runtime and backend dependencies using the installation guide above. Select the bridge repository; the operation sequence is the same for every supported pair.

```python
import latentport as lp

with lp.load("latentport/LPBridge-Qwen3.5-4b-Qwen3.5-9b", backend="nf4", thinking=False) as port:
    state = port.source.capture([("user", "The project codename is Juniper.")])
    transferred = port.bridge.translate(state)
    session = port.target.session(transferred)
    answer = session.ask("What is the project codename?")
    print(answer.text)
```

For a managed conversation, use `port.composite.session()`. Backend selection changes setup requirements, not the `bridge.translate(state)` operation.

## Evaluation

Package smoke checks answered paired red/blue recall prompts correctly on BF16, NF4 and GGUF on an NVIDIA L40S. The task scores below used NF4 with thinking disabled.

Each skills row contains four simple thinking-off prompts. The native 9B baseline receives the full conversation. The question-only 9B baseline receives the final request. The calibrated parent is the writer before continuation training.

| Skill | Released checkpoint | Calibrated parent | Native 9B | Question-only 9B |
| --- | ---: | ---: | ---: | ---: |
| Coding | 4/4 | 4/4 | 4/4 | 0/4 |
| Reasoning | 4/4 | 4/4 | 4/4 | 0/4 |
| Document questions | 3/4 | 3/4 | 4/4 | 0/4 |
| Instruction following | 3/4 | 3/4 | 4/4 | 1/4 |
| **Four-category total** | **14/16** | **14/16** | **16/16** | **1/16** |

On a separate one-fact recall screen, the released checkpoint, the parent and native 9B each answered 12/12 correctly. Question-only 9B answered 6/12. Released checkpoint tied the parent on both screens. A harder, partial coding screen recorded zero successes in the first 15 comparisons for each arm. The planned full held-out evaluation was curtailed, so broader capability and improvement over the parent remain unverified.

## Limitations

This is a directional state transform, not a standalone language model or a reverse bridge. Compatibility is specific to the pinned model pair. Compacted and bounded-suffix resident source states are not supported for translation. Results from small task panels do not establish broad capability, and runtime smoke checks do not qualify answer quality across backends. The broader held-out evaluation was curtailed; improvement over the calibrated parent remains unverified.

## Artifacts and provenance

`latentport.json` records the model identities, artifact filenames and SHA-256 receipts; `bridge_config.json` describes the operation. The released writer is `step0064.safetensors`, SHA-256 `7a2ae07911662520dc38a59b11f1e49917942777563beb0d10f9809c5c1679cd`. The standalone reference runner remains available for reproducing its evaluation.

## License

LatentPort bridge weights and code are licensed under Apache-2.0; see [LICENSE](LICENSE). Base models and runtime dependencies retain their upstream licenses.
