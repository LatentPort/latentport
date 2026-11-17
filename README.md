# LatentPort

**LatentPort makes LLM internal states programmable.**

**Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani).**

Capture a model's working context, transfer it to another model, compact its
attention memory, or save and branch it for later use. Learned transformations
operate on inference state so supported models can continue without replaying
the historical context.

The interface follows the operation. Select a supported bridge or model when
loading; use the same methods to work with its state.

```python
transferred = port.bridge.translate(state)
compacted = state.compact()
branches = state.fork(2)
state.save("saved-state")
```

Translation and compaction each have their own compatibility requirements. The
runtime checks them before proceeding; a shared interface does not make every
model pair or sequence of operations compatible.

<p align="center">
  <img src="https://raw.githubusercontent.com/LatentPort/latentport/main/docs/assets/latentport-ascii.gif" alt="LatentPort: programmable model state" width="640">
</p>

[Website](https://latentport.com) | [Get started](#get-started) | [Supported transformations](#supported-transformations) | [Cookbook](docs/cookbook.md) | [API](docs/api.md)

![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)
[![PyPI](https://img.shields.io/pypi/v/latentport)](https://pypi.org/project/latentport/)

## State operations

| Operation | API | Purpose |
| --- | --- | --- |
| Capture | `model.capture(context)` | Read context and retain its complete inference state. |
| Transfer | `bridge.translate(state)` | Transform a source snapshot into the supported target model's state. |
| Compact | `state.compact()` | Reduce attention KV while preserving recurrent memory, convolution state and logical positions. |
| Append | `model.append(state, text)` | Process new input from an existing state. |
| Fork | `state.fork(2)` | Create independent descendants of a shared immutable snapshot. |
| Save / load | `state.save(path)` / `lp.State.load(path)` | Persist and restore state without historical prompt prefill. |

Learned optimisation, multi-parent state combination and automatic operation
selection are on the roadmap. They are not implemented in this release.

## Get started

Install the lightweight package with Python 3.11 or later:

```shell
python -m pip install latentport
latentport doctor
```

Inference requires CUDA PyTorch and the appropriate backend dependencies.
See [installation](docs/installation.md) for NF4, BF16 and GGUF setup. GGUF also
requires a compatible LatentPort native library. Prebuilt libraries target
specific GPU and CUDA configurations; check the guide before downloading models.

Set `LATENTPORT_HOME` to a directory with enough space for model downloads and
caches. The runtime resolves pinned transformation artifacts and compatible base
models. Model weights and transformation weights are separate downloads.

## Transfer context

Choose `bridge_repo` from the [supported transformations](#supported-transformations)
below. This workflow is the same for every supported bridge:

```python
import latentport as lp

history = [("user", "The ticket is red.")]

with lp.load(bridge_repo, backend="nf4") as port:
    state = port.source.capture(history)
    transferred = port.bridge.translate(state)
    answer = port.target.session(transferred).ask("What color is the ticket?")
    print(answer.text)
```

The source reads the history, `translate()` creates target state, and the target
processes the new question. Changing the bridge repository selects the model
pair and trained transformation. It does not change the operation or require
manual token slicing. `port.handoff()` remains an advanced interface for callers
that already own an exact target-token suffix.

Use `backend="nf4"`, `"bf16"` or `"gguf"` where the selected bridge supports it.
Existing base files can be supplied with `source_path` and `target_path`.
Thinking mode, transfer boundaries and tokenizer compatibility are properties
of the selected checkpoint; see its model card and the [API contracts](docs/api.md).

### Conversations

The composite session manages transfer and continuation with the same calls:

```python
with lp.load(bridge_repo, backend="nf4") as port:
    chat = port.composite.session(history)
    answer = chat.ask("What color is the ticket?")
    print(answer.text, end="")
    while not answer.finished:
        answer = chat.generate()
        print(answer.text, end="")
    print()
    print(chat.ask("What evidence supports that answer?").text)
```

The default `cached_handoff` policy transfers the initial context once and keeps
the target's state for follow-ups. The runtime selects the checkpoint's supported
transfer boundary. Some bridges require the target to read the new question;
others can transfer its prepared answer state. Neither route replays historical
context on the target. A prefix-only bridge with no initial history runs the
new question directly on the target because there is no context to transfer.

`policy="strict"` synchronizes completed answers back to the source and transfers
again on subsequent turns. See the [cookbook](docs/cookbook.md) for streaming,
answer continuation, branching and MTP requirements.

The same command accepts any supported bridge repository:

```shell
python examples/chat.py --repo BRIDGE_REPOSITORY --backend nf4 --context "The ticket is red." --prompt "What color is the ticket?"
```

Replace `BRIDGE_REPOSITORY` with a repository from the table below. Use `--followup`
for another turn, or `--backend gguf` for a compatible native profile.

## Compact context

Choose a supported `model_id` from the table below. The runtime selects the
matching learned compaction weights from the state's identity:

```python
with lp.load_model(model_id, backend="nf4") as model:
    state = model.prefill(history)
    compacted = state.compact()
    answer = model.session(compacted).ask("Continue the task.")
    print(answer.text)
```

The default policy retains one eighth of attention KV at or above 65,536 retained
positions. Shorter histories return unchanged unless `force=True` is supplied.
Use `ratio=4` for a larger retained budget. Compaction preserves the logical clock
and original positions; it does not replace context with a text summary.

For a running session, `session.compact(reserve_tokens=...)` can make room for an
upcoming request within the configured physical capacity. Compaction reduces KV
storage, not model-weight memory. See the [compaction guide](docs/compaction.md)
for context sizing, extended positions, backend options and validation receipts.

## Reuse, branch and resume

```python
review_state, planning_state = state.fork(2)
state.save("saved-state")
restored = lp.State.load("saved-state")
```

Prepare context once for several agents, retain a checkpoint before a risky step,
or resume a saved session. Snapshots record exact model and runtime identities.
Forks share immutable storage; advancing a branch requires its own working state.
Snapshots contain prompt-derived tensors and conversation text, so protect them
like the original conversation. See [persistence and ownership](docs/api.md#memory-and-ownership).

## Supported transformations

One Python runtime serves the following trained artifacts. This is the current
support catalog, not a separate API for each checkpoint.

| Operation | Supported models | Selection |
| --- | --- | --- |
| Transfer | Qwen3.5-4B to Qwen3.5-9B | [`latentport/LPBridge-Qwen3.5-4b-Qwen3.5-9b`](https://huggingface.co/latentport/LPBridge-Qwen3.5-4b-Qwen3.5-9b) as `bridge_repo` |
| Transfer | Qwen3.5-4B to Qwen3.8-27B | [`latentport/LPBridge-Qwen3.5-4b-Qwen3.8-27b`](https://huggingface.co/latentport/LPBridge-Qwen3.5-4b-Qwen3.8-27b) as `bridge_repo` |
| Compact | Qwen3.5-4B | `Qwen/Qwen3.5-4B` as `model_id`; [selector](https://huggingface.co/latentport/LPCompact-Qwen3.5-4B) |
| Compact | Qwen3.5-9B | `Qwen/Qwen3.5-9B` as `model_id`; [selector](https://huggingface.co/latentport/LPCompact-Qwen3.5-9B) |
| Compact | Qwen3.8-27B | `Qwen/Qwen3.8-27B` as `model_id`; [selector](https://huggingface.co/latentport/LPCompact-Qwen3.8-27B) |
| Compact | Qwen3.8-Flash-Next | `Qwen/Qwen3.8-Flash-Next` as `model_id`; [selector](https://huggingface.co/latentport/LPCompact-Qwen3.8-Flash-Next) |
| Compact | Gemma 4 31B IT | `google/gemma-4-31B-it` as `model_id`; [selector](https://huggingface.co/latentport/LPCompact-Gemma-4-31B-it) |
| Compact | Qwen3-14B | `Qwen/Qwen3-14B` as `model_id`; [selector](https://huggingface.co/latentport/LPCompact-Qwen3-14B) |

Each bridge is directional and tied to exact checkpoints. Current bridges require
dense source state with complete alignment metadata; compacted and suffix-only
source states cannot be transferred. Supported quantizations and execution modes
are declared per artifact. An unsupported combination fails explicitly.

## Validation and performance

Published measurements describe specific checkpoints, hardware and workloads:

- At 128K context, 8x compaction reduced attention KV from 4 GiB to 512 MiB on
  Qwen3.5-4B/9B and from 8 GiB to 1 GiB on Qwen3.8-27B.
  See [compaction results](docs/lpcompact-qwen35-validation.json) and
  [backend validation](docs/compaction-validation.json) for exact configurations.
- Historical resident transfer measurements reported 2.19x paired initial
  time-to-first-token speedup at approximately 16K tokens. This is not a general
  throughput claim or a benchmark for every bridge. See [methodology](docs/benchmarks.md)
  and [transfer results](docs/composite-results.json).

State copies, model loading, backend scheduling and retained branches affect
memory and latency. GPU qualification and CPU contract tests cover different
parts of the runtime; consult the records for the profile you intend to use.

## Demos and documentation

- [Chat example](examples/chat.py): select a bridge and use the same conversation flags.
- [Browser demo setup](docs/installation.md): native runtime and local Docker instructions.
- [Memory transplant demo](examples/memory_transplant/README.md): transfer, specialist branches and executable repair checks.
- [State demos](docs/state-demos.md): cached conversations, branching, save/resume and fresh context.
- [Compaction guide](docs/compaction.md), [cookbook](docs/cookbook.md) and [API contracts](docs/api.md).
- [Native integration](docs/native-abi.md) and [contributing](CONTRIBUTING.md).

Report reproducible issues at [GitHub Issues](https://github.com/LatentPort/latentport/issues).
Include `python -m latentport doctor`, exact model identities and a minimal example
with private prompts and credentials removed.

## Project

Created by [Simon P. Villani](https://au.linkedin.com/in/simon-villani), LatentPort
is an independent AI research project. It began with
[LatentPort: Beyond KV Cache](https://arxiv.org/abs/2609.25053), published in
September 2026, and now includes learned transfer, compaction and state lifecycle
operations. Learned optimisation, combination and automatic policy selection
are the next stages of the project.

[Original research code](https://github.com/mmprotest/latentport-paper) |
[Transformation models](https://huggingface.co/latentport) |
[Python package](https://pypi.org/project/latentport/)

## License

The LatentPort runtime and transformation weights are Apache-2.0. Base models
retain their upstream licenses, and llama.cpp retains its MIT license. See
[LICENSE](LICENSE) and the notices supplied with model and native downloads.
