# API and execution contracts

| Object | Responsibility |
| --- | --- |
| `Port` | A pinned source model, target model and directional bridge; owns runtime lifetime. |
| `Model` | Capture/append context and create sessions for one exact model identity. |
| `State` | Immutable logical snapshot with owned tensor storage or a disk representation. |
| `Bridge` | Translate a source snapshot into the supported target state space. |
| `Session` | Append, converse, generate, transform, fork and save state; optionally keep a resident ingestion cache. |
| `CompositeSession` | Apply strict or cached-handoff execution policy. |
| `Answer` | Current call's text/tokens, resulting state, finished flag and timings. |

## Seven operations

| Operation | Example | Contract |
| --- | --- | --- |
| Capture | `port.source.capture(messages)` | Read with the source and capture complete H/K/V/C/R. |
| Translate | `port.bridge.translate(state)` | Create a compatible target snapshot without target history replay. |
| Save | `state.save(path)` | Persist owned state, exact identity and integrity data. |
| Load | `lp.State.load(path)` | Restore a snapshot; no prompt prefill. |
| Fork | `state.fork(2)` or `session.fork(2)` | Independent descendants sharing immutable storage. |
| Append | `port.source.append(state, text)` | Process only an appended token suffix. |
| Escalate | `session.escalate(target=port.target, bridge=port.bridge)` | Experimental one-way live handoff; source session changes only after successful initialization. |

The composite release and published composite results do not qualify live escalation as
product functionality. The primitive remains available for runtime experimentation.
Additional model pairs need separately trained and qualified bridges. There is no
reverse bridge, arbitrary state merge, automatic eviction or distributed execution
in this version of the Python API.

LPCompact serving and resident HTTP sessions are available through the
[vLLM integration](../integrations/vllm/README.md), with BF16, NF4, and GGUF
weight-loading options.

The qualified runtime accepts text and uses greedy decoding. Multimodal input, configurable
sampling, and automatic tool-call execution are outside this preview. Applications can
append textual tool results explicitly.

`port.composite.session()` defaults to `policy="cached_handoff"`. Use
`policy="strict"` explicitly for reader synchronization and full translation each turn.

All supported bridges expose the same transfer operation:

```python
with lp.load(bridge_repo, backend="nf4") as port:
    state = port.source.capture(history)
    transferred = port.bridge.translate(state)
    answer = port.target.session(transferred).ask(question)
```

Capture context before the question for the common workflow. The returned state
retains structured messages, so the target can append the question itself.
Manual token rendering and `port.handoff()` are not required for any model pair.

`port.composite.session(history).ask(question)` and `port.composite.run(messages)`
also dispatch through the selected bridge's supported boundary. A prefix-only
bridge transfers the history and lets the target ingest the final question;
an answer-boundary bridge can transfer the question's prepared state. Stats
include target question prefill when used, while historical target prefill
remains zero. With no history, a prefix-only bridge runs the new question on
the target directly; no source context exists to translate. `run()` accepts
structured messages ending in a user question, or a single question string.

For strict conversations, completed target answers are synchronized back to the
source before the next transfer. The same retry, stream invalidation, continuation
and fork contracts apply to both bridge types. This shared orchestration does not
extend a checkpoint's learned capabilities: prefix-only bridges still reject
answer-ready and mid-answer state passed directly to `translate()`.

## Conversation boundaries

- `ask(question)` begins a **new** user turn; accepts context-ready/completed-turn state.
- `generate()` continues an already prepared or partially generated assistant answer.
- `prepare(question)` processes a new turn but leaves generation to a later call.
- `finished` means the backend emitted an end token, not that the answer is factually correct.
- A composite next turn requires the native chat-turn EOS boundary.
- `Answer.text` and `token_ids` contain the current call's output; the state's
  `generated_token_ids` contains the committed prefix of the current answer.
- The API is synchronous. There is no `aask()` method. Applications can schedule calls
  in worker threads, but the single-GPU engine serializes generation transactions.

Failed `ask()` calls restore the preceding request boundary, including the reader
and synchronization policy in a composite session. Retry with the same question.
Failed `generate()` calls restore the preceding prepared/partial answer; retry
with `generate()`. Explicit `prepare()` remains a separate committed operation.

`on_token` runs only after the backend ingests a token (or verifies an MTP batch),
including EOS. Public state commits at the end of the request. If a later step,
callback, cancellation, snapshot or cleanup fails after any callback invocation,
`GenerationInterrupted` reports `token_ids` and `recovery_state`; its `__cause__`
holds the original failure. Discard all output from that failed call before
retrying. `token_ids` includes a callback that raised, since it may have delivered
output first. This is an explicit stream invalidation contract, not exactly-once
delivery or a per-token durable checkpoint. Callbacks must not reenter or mutate
the active session. Different sessions may be scheduled by the application.

`State.info` reports phase, clock, representation, tensor bytes, lineage, storage-sharing
status and cumulative target prompt count. A state must match the model revision,
backend, weights, template and ABI. No compatibility error triggers silent prompt replay.

## Memory and ownership

Forking is logically cheap because tensors are shared immutably. Advancing a branch
requires its own working state; retaining many answer snapshots consumes additional VRAM.
There is no paged prefix sharing across active native sequences. Release old references
when no longer needed. `port.close()` frees models and the bridge; independent states
remain owned by the caller. Both models normally remain resident; releasing the reader
snapshot in cached-handoff mode does not unload the reader's model weights.

Models obtained from a `Port` are non-owning views. Closing one view invalidates
that view and its sessions, while its sibling remains usable. Closing the `Port`
invalidates both views and releases the shared runtime once. A standalone model
owns its runtime. Close active sessions before closing their owner.

Translation passes private tensor copies to the translator, protecting parents
and forks even for inference-mode tensors or a translator that raises after an
in-place write. This adds source-state copy bandwidth and peak memory; GPU
measurement is required before claiming unchanged handoff performance.

New saves use `latentport.state.v2` with a required `manifest.sha256` receipt over
the complete manifest, which includes the tensor payload digest. These checks
detect accidental corruption; they do not authenticate a snapshot against someone
who can replace both files. Loading a trusted v1 snapshot requires
`State.load(path, allow_legacy=True)`; saving it again writes v2. Snapshots still
contain plaintext conversation data and need application-managed access controls.

Native states use BF16 K/V and FP32 H/C/R with explicit bridge coordinate conversion.
MTP draft caches are transient runtime state and are rebuilt from a newly generated
committed suffix after restore. See [native-abi.md](native-abi.md).

## Standalone extended-position execution

`lp.load_model(model_id, backend="bf16", max_logical_positions=n)` selects
a logical context length of `n` positions on all six pinned
LPCompact models. Alternatively, `position_policy="yarn4"` selects four times
the model's native range. Qwen3-14B retains a 40,960-position physical ceiling.
Set the length before ingesting history. See the [streaming example
and per-model position rules](compaction.md#continuous-histories-beyond-the-original-window).

`lp.load_model(backend="nf4" | "bf16" | "gguf", position_policy="yarn4")`
selects static YaRN factor 4 from the first token, with a logical limit of
1,048,576 positions. The default position policy retains the 262,144 logical
limit. `max_kv_positions` independently bounds physical attention occupancy
(at most 262,144); it does not change the rotary position range.

`state.compact(ratio=8)` is the shared learned compaction operation. Its result
retains original absolute positions and the same logical clock. Inspect the
actual attention sequence dimension through `state.physical_kv_length`.
`model.append(state, token_ids)` continues from that state using the same
position policy. All input, formatting and generated tokens advance the clock.

`model.session(state=None, resident=True)` selects resident ingestion within
the ordinary `Session` API. `model.live(state=None)` is a convenience alias
returning the same type. The backend must explicitly provide resident execution.
The shared session supports:

- `append(text_or_token_ids)` processes only fresh input and returns the logical clock.
- `clock` and `physical` expose logical and retained attention lengths separately.
- `compact(ratio=8, force=False, reserve_tokens=0)` returns an immutable result and can reserve capacity for the next request.
- `snapshot(release=False)` publishes an immutable `State`; `release=True` closes the live handle.
- `ask()` / `prepare()` / `generate()` use the existing conversation and decoding rules.
- `fork()` creates isolated sessions with the same residency preference.
- `save(path)` saves an immutable snapshot without closing the session.
- `escalate(target=..., bridge=...)` uses the same transformation transaction as compaction, subject to bridge compatibility.
- `close()` releases the handle while independently owned snapshots survive.

`append()` preserves a literal token suffix; it does not insert spaces or record
separators. When ingesting independent records, include the boundary explicitly,
for example `session.append("\n\n" + record)`. Otherwise an unfinished line can
join the next record and change how the model interprets it. For chunks cut from
one already tokenized stream, append the original token IDs so chunk boundaries
do not change tokenization.

Exact-token ingestion accepts nonnegative Python integers and rejects booleans,
floats, strings and empty sequences before invoking the backend.

Session compaction can trigger below 65,536 positions when the configured physical
limit is reached or the reserved budget would fill it. Include new prompt,
formatting and maximum generated tokens in `reserve_tokens`. Impossible physical
or logical budgets fail before installation. There is no automatic compaction on
append, and this token-capacity check is not a GPU allocation estimate.
`State.compact()` retains its fixed threshold unless `force=True` is supplied.
A no-op session transform preserves the resident handle and avoids reinstallation,
although inspecting a resident snapshot still materializes tensors.

All selectors require their trained base revision, including Qwen3.8-27B. A revision
match alone does not qualify arbitrary local model weights or GGUF quantizations;
exact weight-hash qualification remains separate from this compatibility check.

Consecutive resident appends reuse the working cache. Conversation preparation
and generation currently freeze that cache and use the existing decode path;
the next append reinstalls a resident cache. Accessing `session.state` or
`snapshot()` materializes an immutable snapshot; use `clock` and `physical`
for counters without copying state tensors. This is a unified execution API,
not a claim of zero-copy transitions between all operations.

Transformations freeze a source snapshot, validate the result against the
destination identity, and prepare installation before committing the model
change. If transformation or installation fails, the source model and exact
snapshot remain available. A subsequent append resumes from that snapshot
without text replay. Partial append failure closes the session; applications
must restore their last explicit checkpoint.

Live snapshots keep a bounded original-token suffix for exact stream boundaries.
They can be saved, loaded, forked and queried through `model.session(state)`
without replaying historical source. The suffix is boundary metadata, not text
injected as replacement memory. Position/quantization configuration is part of
state identity. See [compaction.md](compaction.md) for examples and native builds.


### Bridge alignment compatibility

The mixed-state bridge requires the complete source token stream and its exact
rendered bytes. Its numeric alignment maps each target token to a source
coordinate using byte spans and produces dense target positions. The bridge
explicitly rejects suffix-only streamed states and sparse/compacted states
before loading or invoking the learned translator. Destination capacity is
checked after token alignment and before translation. Ordinary dense handoffs
continue through the shared transaction.

A future streaming alignment contract must bind both tokenizer revisions and
template rules, preserve exact source/target token boundaries and clocks, define
how sparse retained positions map into the target cache, and survive save/load
and forks. It must also define behavior when the destination has insufficient
physical or positional capacity. Incrementally collected metadata could replace
whole-text reconstruction, but the current checkpoint has no established sparse
transfer contract. A bounded token suffix cannot supply that information.

Reverse bridges and optimisers are not supplied by this API change. The internal
snapshot-transform-install mechanism can install a different model identity,
but each learned operation must declare and validate its supported inputs.

### Explicit question-boundary transfer

Capture the prefix before the final request, then call
`port.handoff(prefix_state, target_suffix_ids=ids)`. Supply the target token IDs
for the withheld question and assistant opening. The returned answer-ready state
records the exact number of target prompt tokens processed. For a separate new
chat turn use `port.target.session(port.bridge.translate(prefix_state)).ask(question)`.
An arbitrary token suffix does not identify a canonical structured user message:
`handoff()` clears `messages` and sets `messages_complete=False`. The exact neural
and token stream remains authoritative. That flag survives generation, forks and
save/load. Appending unstructured tokens to a structured conversation also marks
its message metadata incomplete, preventing later turns from treating it as a
complete reconstruction of the stream.


### Exact weights for compaction

`State.compact()` and `Session.compact()` check the exact base-weight identity
against `configs/compaction-compatibility.json` before reading state tensors or
loading the selector. The catalog records immutable upstream safetensors hashes
and GGUF hashes tied to the trained base revision. It establishes checkpoint
provenance; it does not establish answer quality for every execution profile.

Unknown local weights or GGUF conversions require an explicit experimental opt-in:

```python
child = state.compact(allow_unqualified=True)
# The same option is available on session.compact(...).
```

The opt-in does not bypass model revision, backend or state-layout checks.
Existing GGUF validation reports without exact file hashes cannot populate the
catalog: notably, the pinned 27B revision currently has no cataloged GGUF hash.
Those files require the opt-in until an exact-file receipt is added. The legacy
`compact_flash_next_cache` helper receives no model identity, so callers using
raw Transformers caches must verify provenance themselves; prefer `State.compact()`.

`Answer.text` is a text delta across successive `generate()` calls for the same
answer, including after save/load or fork. Incomplete trailing UTF-8 characters
are withheld until more tokens arrive or EOS is committed. `Answer.token_ids`
continues to report every committed token in that call.


## Combine

[LPCombine overview](https://latentport.com/#combine).

`model.combine(left, right, method="auto")` returns a new context-ready
state from two context-ready states with identical model and execution
identities. The order is left followed by right. The original states
can still be used. The equivalent state call is
`left.combine(right, model=model)`. `session.combine(right)` updates the session
only after successful installation.

- `method="tokens"` uses the complete right-hand token stream captured by
  `prefill()` or a resident session opened with `retain_replay=True`.
  BF16, NF4 and GGUF run this through their native append paths.
- `method="cache"` uses cached model input embeddings to recover the right
  tokens on BF16, NF4 and GGUF. LatentPort 0.4.11 adds these activations to
  new states, including snapshots from `model.live()` with its default settings.
  Each token must map to a unique embedding row in the active checkpoint.
  For older snapshots without cached input embeddings, use the legacy
  cache-only backends in the [profile table](../README.md#supported-transformations)
  or recapture the right state to add the input-embedding cache.
- `method="auto"` selects recorded tokens when available and otherwise uses
  the profile's cache-only method.

`State.combination` identifies the method, both input state IDs and the
replayed token count. It round-trips via `State.save()`. The resulting
state can be queried, continued or forked. Compaction remains available on
backends that expose the attention tensors required by LPCompact.

All six profiles accept `backend="bf16"`, `"nf4"` or `"gguf"`.
Loading an LPCombine alias resolves the profile's Hub `config.json`. The
runtime checks its base checkpoint pin and can use the packaged profile
when the Hub is unreachable.
For Qwen3-14B, Gemma 4 31B and Flash-Next, GGUF snapshots store the complete
native llama.cpp sequence. Keep split checkpoint files together and provide
one shard as `model_path`. Use `NativeOptions(gpu_layers=...)` to set the
number of layers offloaded to the GPU for these three GGUF architectures.
Flash-Next automatically installs the pinned llama.cpp b12503 CPU runtime.
For CUDA on Flash-Next, supply the path to the b12503 CUDA library via
`NativeOptions(library=..., gpu_layers=...)`.

`latentport combine LEFT RIGHT OUTPUT` reads saved states and writes a
new snapshot. Use `--model-path` for local weights and match any position
and capacity flags used to capture the inputs.
