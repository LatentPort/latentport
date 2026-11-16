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

The composite release and published E120 results do not qualify live escalation as
product functionality. The primitive remains available for runtime experimentation.
Additional model pairs need separately trained and qualified bridges. There is no
reverse bridge, arbitrary state merge, automatic eviction, vLLM integration or distributed
execution in this version.

The qualified runtime accepts text and uses greedy decoding. Multimodal input, configurable
sampling, and automatic tool-call execution are outside this preview. Applications can
append textual tool results explicitly.

`port.composite.session()` defaults to `policy="cached_handoff"`. Use
`policy="strict"` explicitly for reader synchronization and full translation each turn.

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

Native states use BF16 K/V and FP32 H/C/R with explicit bridge coordinate conversion.
MTP draft caches are transient runtime state and are rebuilt from a newly generated
committed suffix after restore. See [native-abi.md](native-abi.md).

## Standalone extended-position execution

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
- `compact(ratio=8, force=False)` delegates to `State.compact()` and returns the immutable result.
- `snapshot(release=False)` publishes an immutable `State`; `release=True` closes the live handle.
- `ask()` / `prepare()` / `generate()` use the existing conversation and decoding rules.
- `fork()` creates isolated sessions with the same residency preference.
- `save(path)` saves an immutable snapshot without closing the session.
- `escalate(target=..., bridge=...)` uses the same transformation transaction as compaction, subject to bridge compatibility.
- `close()` releases the handle while independently owned snapshots survive.

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

The current E119 bridge requires the complete source token stream and its exact
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
