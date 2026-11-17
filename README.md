# LatentPort

**LatentPort makes LLM internal states programmable.**

We're building a library of learned state transformations that enable models and
agents to transfer, compact, optimise and combine their internal states during
inference.

This allows developers to improve model performance, reduce inference costs and
build longer-running agents without repeatedly processing the same context.

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

LatentPort is an independent AI research project developing programmable internal state for language models.

The project began with [LatentPort: Beyond KV Cache](https://arxiv.org/abs/2609.25053), published in September 2026, demonstrating recurrent-state transfer between Qwen3.5 4B and 9B without target-prefix replay.

The project now includes a shared Python runtime, learned model-to-model transfers, state compaction, and state persistence and branching.

[Original paper](https://arxiv.org/abs/2609.25053) | [Research code](https://github.com/mmprotest/latentport-paper) | [Python package](https://pypi.org/project/latentport/)

<p align="center">
  <img src="https://raw.githubusercontent.com/LatentPort/latentport/main/docs/assets/latentport-ascii.gif" alt="LatentPort: programmable model state" width="640">
</p>

[Website](https://latentport.com) | [Get started](#get-started) | [Models](https://huggingface.co/latentport) | [Cookbook](https://github.com/LatentPort/latentport/blob/main/docs/cookbook.md) | [API](https://github.com/LatentPort/latentport/blob/main/docs/api.md)

![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)
[![PyPI](https://img.shields.io/pypi/v/latentport)](https://pypi.org/project/latentport/)

### Transfer a query to Qwen3.5-9B

Install CUDA PyTorch and `latentport[reference]==0.4.1` for NF4 bridge inference.
Select the 9B writer by its model repository:

```python
import latentport as lp

with lp.load("latentport/LPBridge-Qwen3.5-4b-Qwen3.5-9b", backend="nf4", thinking=False) as port:
    prefix = port.source.prefill([("user", "The ticket is red.")])
    request = port.target.codec.render(
        [("user", "The ticket is red."), ("user", "What color is the ticket?")],
        generation_prompt=True,
    )
    assert request.token_ids[:prefix.clock] == prefix.rendered_token_ids
    ready = port.handoff(prefix, target_suffix_ids=request.token_ids[prefix.clock:])
    print(port.target.session(ready).generate(max_new_tokens=16).text)
```

The checkpoint supports text-only, thinking-off handoff before the final question.
See [the runnable example](https://github.com/LatentPort/latentport/blob/main/examples/bridge_9b.py).

## A library of state operations

A model's internal state carries what it has already processed. LatentPort exposes
that state as an object developers can transform, persist, reuse and branch.
Learned operators work directly on state tensors, allowing inference to continue
from the transformed state.

| Operation | Purpose | Available today |
| --- | --- | --- |
| **Transfer** | Move working context into another model's state space. | Qwen3.5-4B to Qwen3.8-27B via `bridge.translate(state)`, and Qwen3.5-4B to Qwen3.5-9B via `port.handoff()`. |
| **Compact** | Reduce attention memory while retaining the model's recurrent and convolutional state. | LPCompact for Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B via `state.compact()`. |
| **Optimise** | Improve a state for the task or inference objective ahead. | Part of the roadmap. |
| **Combine** | Bring information from multiple states into a shared working state. | Part of the roadmap. |

The shared runtime also provides capture, append, save, load and fork operations.
Snapshots are immutable, so branches can share a starting context while advancing
independently. Each learned transformation is trained for its supported model
architecture or model pair.

## Get started

Install the lightweight Python package with Python 3.11 or later:

```shell
python -m pip install latentport
latentport doctor
```

The base package includes the API, diagnostics and demo interface. Inference
dependencies are optional: see [installation](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) for
`latentport[gguf]`, `latentport[bf16]` and CUDA PyTorch setup.

### Try the browser demo

On a qualified NVIDIA GPU host with Docker GPU access configured:

```shell
docker build -f docker/Dockerfile --target gguf -t latentport:local .
docker run --rm --gpus all -p 127.0.0.1:8768:8768 -v latentport-cache:/cache latentport:local
```

Open **http://localhost:8768**. The demo prepares the models, transfers context,
forks specialist sessions, and checks a generated repair with executable tests.
The initial transfer profile requires an SM120 GPU with at least 32 GB VRAM and
space for the bridge and the two pinned GGUF bases. The runtime fetches each from its publisher.
Windows Docker GPU execution requires the WSL 2 backend.

**Release status:** the Python package is available on [PyPI](https://pypi.org/project/latentport/).
The GHCR images are pending GPU qualification and publication; until then, use
the [local image build](https://github.com/LatentPort/latentport/blob/main/docs/installation.md).

Choose the setup for your operation:

- **Compaction:** [NF4, BF16 and GGUF setup](https://github.com/LatentPort/latentport/blob/main/docs/compaction.md), including static YaRN4 for continuous histories beyond the original physical window and the resident append API.
- **Model transfer:** [Windows GGUF setup](https://github.com/LatentPort/latentport/blob/main/docs/installation.md) and [conversation recipes](https://github.com/LatentPort/latentport/blob/main/docs/cookbook.md).

Set `LATENTPORT_HOME` to a directory with enough space for downloaded models and
caches. The runtime downloads transformation weights when needed; base models are
separate artifacts. Prebuilt GGUF runtimes are available for Windows x64 / CUDA 13
and Linux x64 / CUDA 12.8 on Blackwell SM120.

## Compact a running model's state

Each supported base model has a dedicated learned selector. The runtime chooses it from the state’s model identity.

```python
state = state.compact()
```

Given an agent's accumulated `history`, continue from its compacted state:

```python
import latentport as lp

with lp.load_model("Qwen/Qwen3.5-4B", backend="bf16") as model:
    state = model.prefill(history)
    state = state.compact()
    answer = model.session(state).ask("Continue the task.")
    print(answer.text)
```

The default policy retains one eighth of attention KV positions on histories with
at least 65,536 retained positions. A 4x setting is available with
`state.compact(ratio=4)`. Recurrent state, convolution state and the original
logical positions are preserved. Compaction uses no text summary or historical
token replay.

Pass `"Qwen/Qwen3.5-9B"` to use the 9B selector. The default base model is Qwen3.8-27B.
Each base model uses the same selector across its supported weight formats.
For GGUF, use `backend="gguf"` and supply `model_path` with your chosen base-model
file. See the [compaction guide](https://github.com/LatentPort/latentport/blob/main/docs/compaction.md) for context sizing, policy
details and the `state.compaction` receipt.

## Transfer context between models

The default L127 step-64 transfer bridge lets Qwen3.5-4B read a context and Qwen3.8-27B continue
from the translated state. It transfers attention K/V, recurrent memory,
convolution history and the terminal hidden state.

```python
import latentport as lp

with lp.load("latentport/LPBridge-Qwen3.5-4b-Qwen3.8-27b", backend="nf4") as port:
    port.warm()
    context = port.source.capture([("user", "Repository context: ...")])
    transferred = port.bridge.translate(context)
    answer = port.target.session(transferred).ask("Explain the repository.")
    print(answer.text)
```

Select `backend="bf16"`, `"nf4"` or `"gguf"`. Set `thinking=True` to use the thinking-on templates and correction bank. GGUF downloads the two base models from the publishers pinned in the bridge manifest. The same API accepts a different bridge repository and revision, or a verified local bundle directory.

For a complete conversation, `port.composite.session()` handles the initial
handoff and retains the target's state across turns. The default `cached_handoff`
policy translates once, then processes new input on the target. An explicit
`strict` policy keeps the source model responsible for new user prompts.

The [cookbook](https://github.com/LatentPort/latentport/blob/main/docs/cookbook.md) covers both policies, answer continuation,
streaming and MTP. Existing local GGUF files can be supplied through `source_path`
and `target_path`.

## Reuse, branch and resume

The same state abstraction supports persistence and independent continuations:

```python
review_state, planning_state = state.fork(2)
state.save("saved-state")
restored = lp.State.load("saved-state")
```

Prepare a repository once, let different agents work from the same snapshot, or
save a long-running session and resume it later. Restoring a state reuses the
captured context. Appending new tool results processes the new input.

Each snapshot records its exact model identity and runtime compatibility. Forks
share immutable storage; execution uses the selected backend's scheduling.

## Measured results

| Transformation | Measurement | Validation |
| --- | --- | --- |
| LPCompact Qwen3.5-4B and 9B, 128K history at 8× | Attention KV tensors fell from **4 GiB to 512 MiB** on each model. | BF16, NF4, GGUF Q4_K_M and GGUF Q8_0 passed continuation, fork, save/load and repeated compaction at 64K and 128K. [Results by task and model](docs/lpcompact-qwen35-validation.json) |
| LPCompact Qwen3.8-27B, 128K history at 8x | Attention KV tensors reduced from **8 GiB to 1 GiB**. | BF16, Q4_K_M and Q8_0 tested at 64K and 128K on RTX PRO 6000; continued generation, save/load, forks and repeated compaction passed. |
| Historical E119 transfer, approximately 16K prompts | **2.19x paired time-to-first-token speedup**; native median 24.663 s, composite median 11.087 s. | E120 resident NF4/BF16 research runtime on RTX 5090; paired 95% interval 2.15-2.23x. |

These measurements describe their stated workloads and configurations. Compaction
reduces attention-state storage; model weights and recurrent state retain their
existing representation. The transfer timing includes prefill, translation,
installation and the first emitted token, excluding cold model loading.

See [compaction validation](https://github.com/LatentPort/latentport/blob/main/docs/compaction-validation.json),
[transfer results](https://github.com/LatentPort/latentport/blob/main/docs/composite-results.json) and
[benchmark methodology](https://github.com/LatentPort/latentport/blob/main/docs/benchmarks.md) for the full records.

## A state operating layer for inference

The long-term direction is a state operating layer for AI inference:

- **Model-specific translators** connect the internal state spaces of compatible models.
- **Reusable operators** transfer, compact, optimise and combine state through a shared interface.
- **A policy engine** selects transformations automatically according to the application's capability, latency and memory requirements.

The current release provides explicit transfer and compaction operations, plus
the state lifecycle primitives. Learned optimisation, state combination and
automatic policy selection are the next layers of that vision.

## One runtime, separate transformation models

The shared Python runtime lives in this GitHub repository. Each trained state
transformation has its own Hugging Face repository, with model compatibility,
weights, versions and a model card.

- Compaction selectors for [Qwen3.5-4B](https://huggingface.co/latentport/LPCompact-Qwen3.5-4B), [Qwen3.5-9B](https://huggingface.co/latentport/LPCompact-Qwen3.5-9B) and [Qwen3.8-27B](https://huggingface.co/latentport/LPCompact-Qwen3.8-27B).
- The [directional 4B to 27B bridge](https://huggingface.co/latentport/LPBridge-Qwen3.5-4b-Qwen3.8-27b) contains the Safetensors and lossless GGUF bridge weights. Its manifest identifies the separate base checkpoints and supported model profiles.

Transformation repositories contain their own learned artifacts. Base-model
weights are obtained separately.

## Demos and documentation

- [Memory transplant demo](https://github.com/LatentPort/latentport/blob/main/examples/memory_transplant/README.md): live state transfer, specialist branches and a repair checked by executable tests.
- [State demos](https://github.com/LatentPort/latentport/blob/main/docs/state-demos.md): cached conversations, branching, save/resume and fresh context updates.
- [Time-to-first-token demo](https://github.com/LatentPort/latentport/blob/main/docs/ttft-demo.md): compare native 27B and composite inference on a fresh 16K prompt.
- [Compaction guide](https://github.com/LatentPort/latentport/blob/main/docs/compaction.md), [cookbook](https://github.com/LatentPort/latentport/blob/main/docs/cookbook.md) and [API contracts](https://github.com/LatentPort/latentport/blob/main/docs/api.md).
- [Native integration](https://github.com/LatentPort/latentport/blob/main/docs/native-abi.md) and [contributing](CONTRIBUTING.md).

Report reproducible issues at [GitHub Issues](https://github.com/LatentPort/latentport/issues).
Include the output of `python -m latentport doctor`, model identities and a minimal
example, with private prompts and credentials removed.

## License

The LatentPort runtime and transformation weights are Apache-2.0. Qwen base models
retain their upstream licenses, and llama.cpp retains its MIT license. See
[LICENSE](LICENSE) and the notices supplied with the model and native downloads.
