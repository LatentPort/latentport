# API and execution contracts

| Object | Responsibility |
| --- | --- |
| `Port` | A pinned source model, target model and directional bridge; owns runtime lifetime. |
| `Model` | Capture/append context and create sessions for one exact model identity. |
| `State` | Immutable logical snapshot with owned tensor storage or a disk representation. |
| `Bridge` | Translate a source snapshot into the supported target state space. |
| `Session` | Advance a model's conversation or generation state. |
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
