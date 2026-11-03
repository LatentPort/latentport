# LatentPort runtime

LatentPort makes a supported model's inference state a programmable object. The
first supported bridge transfers the complete Qwen3.5-4B hybrid state into the
Qwen3.8-27B state space using the selected LatentPort-4+27B research-preview
bridge. The source model and target model are separate downloads.

## Core API

```python
import latentport as lp

port = lp.load(
    "latentport/latentport-4-27b-research-preview",
    backend="reference",
    source_path="/models/Qwen3.5-4B",   # optional pinned local snapshot
    target_path="/models/Qwen3.8-27B",  # optional pinned local snapshot
)
print(port.warm())  # preloads both models and bridge; reports GPU residency

# The 4B reads the complete prompt. The 27B starts from translated H and
# complete K/V, convolution and recurrent state; it reads zero prompt tokens.
answer = port.composite.session().ask("Explain this code.", on_token=lambda token_id: print(token_id))
print(answer.text, answer.stats.request_ttft_s)

# Reuse a prepared context, then ask different questions in target branches.
context = port.source.capture([("user", "Repository context: ...")])
shared = port.bridge.translate(context)
left, right = shared.fork(2)
a = port.target.session(left).ask("Find security issues")
b = port.target.session(right).ask("Suggest performance improvements")
# These two branch questions are intentionally processed by the target.

# A literal prefix can also be captured and branched without chat messages.
raw = port.source.capture("Shared context text")
raw_target = port.bridge.translate(raw)
raw_answer = port.target.session(raw_target.fork()).ask("A different question")

# A live source answer can be handed off at a committed token boundary.
small = port.source.session()
small.prepare("Investigate this bug")
small.generate(max_new_tokens=32)
small.escalate(target=port.target, bridge=port.bridge)
larger_answer = small.generate(max_new_tokens=512)

# State is independent of loaded model objects and can be saved and forked.
context.save("/bulk/states/repository.lpstate")
restored = lp.State.load("/bulk/states/repository.lpstate")
extended = port.source.append(restored, "\nAdditional context")
```

`capture`, `translate`, `save`, `load`, `fork`, `append`, and `escalate` operate
on exact state identities. `ask()` begins a new chat turn; `generate()`
continues an answer. `append()` only adds tokens sequentially. Branches share a
logical immutable prefix; advancing a branch creates its own working cache.
Model revision, tokenizer template, quantization identity, backend and state
ABI are part of a state identity. Incompatible states fail explicitly.
The `on_token` callback receives token IDs as soon as they are selected; decode
the accumulated IDs with the target codec for text streaming.

The composite session synchronizes completed 27B answers back through the 4B
before a later question. It checks native token boundaries and reports the
synchronization time. Live multi-turn sessions append the next native turn
suffix because Qwen's whole-history template can rewrite earlier thinking;
previously consumed state is never replayed. A live escalation is one-way and
does not synchronize the source after handoff. Target-side branching with new
questions is a
different policy: the target processes those *new* question tokens, while the
historical context is transferred through the bridge.

## Backends and current status

The reference backend contains the selected dual-branch translator, pinned
NF4 model loader, native complete-state capture, target cache installation and
greedy decoding. It uses CUDA and the exact model revisions recorded in the
bridge bundle. Its runtime path is implemented but has not yet had a package
GPU integration run. Install with `pip install -e ".[reference]"` and put bulk
model files under a path set by `LATENTPORT_HOME` (the default on this Windows
machine is `F:/LatentPort/runtime`).

The GGUF backend now includes a compiled CUDA llama.cpp extension, typed
complete-state capture/installation, first-token projection from translated H,
and batched greedy MTP verification with full hybrid-state rollback. It loads
both GGUF models on the GPU and executes the bridge GGUF through Torch CUDA.
The native build, DLL loading, CPU contracts and bridge tensor bytes have been
checked. **Full model execution and performance qualification remain untested
until the GPU is available after E120.** Build and execution instructions are
in [docs/native-abi.md](docs/native-abi.md). Stock llama.cpp is not compatible
with the typed installer; missing MTP weights and unsupported configurations
fail explicitly.

The GGUF bridge file is a custom FP32 tensor container, separate from the
4B and 27B model GGUF files. It is not a standalone language model. Immutable
snapshots share storage; active branches use private working caches. The
single-GPU scheduler queues generation requests rather than exchanging full
states between each token. Paged KV sharing across active branches is not yet
implemented.

## State and diagnostics

`State.info` reports the clock, phase, representation, field count, tensor
bytes, storage sharing and target prompt count. `State.save()` writes `manifest.json` and
`tensors.safetensors` with a checksum and exact compatibility metadata. Large
snapshots and model downloads belong on bulk storage, outside the code repo.
Snapshots contain prompt-derived state and rendered text; protect them as
conversation data.

`Answer.stats` reports reader prefill and sync, translation, target prefill,
device transfer, target installation, first-token time, full request TTFT,
generation time, total time, target prompt tokens and optional peak allocated VRAM.
Native MTP also reports proposals, accepted drafts, verification batches and
ordinary warmup tokens. Requested MTP and actual speculation are separate.
Queue time is included in end-to-end TTFT. `port.warm()` reports residency,
native weight/cache/compute bytes and GPU free memory.
Native GGUF peak VRAM remains unavailable; the resident-memory report is a
snapshot, not a peak measurement. GGUF state installation and first-token head
computation have separate timings.
The no-replay rule applies to translated history and to the entire first-turn
composite prompt. Target-side branch questions are reported as target prompt
tokens.

## Development

CPU contract tests:

```shell
python -m unittest discover -s tests -v
```

GPU integration and performance checks should be run only when the separate
E120 experiment no longer owns the GPU. The package does not train or choose
the bridge, or run a model-quality evaluation campaign.

After E120, this exercises the implemented reference path:

```shell
python scripts/smoke_runtime.py --backend reference --source-path /models/Qwen3.5-4B --target-path /models/Qwen3.8-27B
```

It covers capture, save/load, append, translation, decoding, escalation and branching.
It writes its state and result under `LATENTPORT_HOME/smoke` and does not
substitute for a matched performance run.

After E120, run the native correctness checks with both GGUF files:

```shell
python scripts/smoke_gguf.py --gpu-is-free --source-path F:/LatentPort/models/reader.gguf --target-path F:/LatentPort/models/target-with-mtp.gguf
```

This adds native save/restore comparison, forced accepted/rejected MTP drafts,
committed-state restoration, zero-target-prefill counters and resident memory
reporting. It performs no quality benchmark or bridge selection.
