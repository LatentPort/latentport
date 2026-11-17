# Build with reusable model state

Select a supported bridge with `port = lp.load(bridge_repo, backend="gguf")`.
The recipes use the same operations for every bridge. Call `port.warm()` once,
and `port.close()` when done. Start with [installation](installation.md).

## Finish an answer, then ask another question

`Answer.text` is the text from the current call, not the whole answer so far. Accumulate
chunks, or decode `answer.state.generated_token_ids` for the complete current answer.
Use a generation budget in an application:

```python
def complete(session, question, *, mtp=False, chunks=8):
    answer = session.ask(question, max_new_tokens=256, mtp=mtp)
    parts = [answer.text]
    for _ in range(chunks - 1):
        if answer.finished:
            return "".join(parts)
        answer = session.generate(max_new_tokens=256, mtp=mtp)
        parts.append(answer.text)
    if not answer.finished:
        raise RuntimeError("Answer still in progress; increase the budget or resume later")
    return "".join(parts)

chat = port.composite.session([("user", "Document: ...")])
print(complete(chat, "Summarize the document."))
print(complete(chat, "What assumptions does the author make?"))
```

The source reads the context, the bridge transfers state, and the target generates.
The runtime handles whether the checkpoint transfers before or after the question.
Later questions append to the retained target state. No history is replayed on the
target. Choose `policy="strict"` to keep the reader synchronized with each turn;
this feeds back the last answer and retranslates before the next answer. For
prefix-only bridges the target also reads the new question. Policy stays fixed.

## Prepare once, branch into different tasks

```python
context = port.source.capture([("user", "Document collection: ...")])
shared = port.bridge.translate(context)
audit, summarize = port.target.session(shared).fork(2)
print(complete(audit, "Find contradictions across the documents."))
print(complete(summarize, "Extract dates, owners and deadlines."))
```

The context is translated once. Each target branch reads its own new question. Forks
share immutable snapshot storage; generation installs a private working cache. Requests
queue on one GPU. There is no claim of simultaneous agent execution or paged sharing
between active sequences. A native target can also cache a shared prefix: LatentPort's
additional benefit is preparing that initial context through the smaller model.

You can also branch a completed cached conversation:

```python
left, right = chat.fork(2)
print(complete(left, "Argue for the proposal."))
print(complete(right, "Argue against it."))
```

A fork made mid-answer must resume that answer with `generate()` before a new question.

## Save context and resume later

```python
from pathlib import Path
import latentport as lp

context = port.source.capture([("user", "Technical manual: ...")])
context.save(Path("F:/states/manual.lpstate"))
# In a later process with the same model/runtime identities:
restored = lp.State.load(Path("F:/states/manual.lpstate"))
shared = port.bridge.translate(restored)
print(complete(port.target.session(shared), "Which maintenance steps are required?"))
```

Saved state has a manifest, tensors and checksums. It is tied to the exact model revision,
quantization, backend, template and state ABI. It contains prompt-derived tensors **and
rendered text**: protect it like the original conversation. Save/load does not summarize
or shrink the context. Attention state grows with context length.

## Append new observations

```python
context = port.source.capture("Initial incident log: ...")
context = port.source.append(context, "\nNew observation: the worker restarted.")
target_state = port.bridge.translate(context)
print(complete(port.target.session(target_state), "What should we investigate first?"))
```

`append` adds tokens sequentially. It cannot remove a stale fact, edit an earlier document,
merge independent states or ingest an unlimited stream. Rebuild context when its earlier
contents must change. Translation remains a full-state operation, not an incremental
bridge update.

## Stream tokens

The callback receives backend-committed token IDs, including reasoning/control
tokens. Public session state commits when the request succeeds. A later failure
raises `lp.GenerationInterrupted`: discard this call's streamed output before
retrying from its restored request boundary. A UI should make streamed output
replaceable until success. A terminal cannot retract already printed text.
To avoid splitting UTF-8 text across individual tokens, decode the accumulated IDs:

```python
ids = []
shown = ""
def stream(token_id):
    global shown
    ids.append(token_id)
    text = port.target.codec.decode(ids, special=False)
    # Hold an incomplete byte sequence until the next token.
    if not text.endswith("\ufffd") and text.startswith(shown):
        print(text[len(shown):], end="", flush=True)
        shown = text

try:
    answer = chat.ask("Explain the tradeoffs.", on_token=stream)
except lp.GenerationInterrupted:
    ids.clear()
    shown = ""
    print("\n[Interrupted: discard the preceding partial answer before retrying.]")
    raise
# Pass the same callback to generate() when continuing a capped answer.
```

## Enable MTP

Use this option with a separately qualified target profile containing trained MTP heads. Default bridge generation uses ordinary greedy decoding.

```python
port = lp.load(bridge_repo, backend="gguf", native_options=lp.NativeOptions(mtp_drafts=4))
port.warm(mtp=True)
answer = port.composite.session().ask("Explain this code: ...", mtp=True)
print(answer.stats.mtp_proposed_tokens, answer.stats.mtp_accepted_tokens)
```

With a trained MTP head, the runtime can unroll multiple
draft steps and verify them greedily. Accepted/rejected drafts preserve committed hybrid
state. Speculation begins after the target has generated a native token; the bridge's
terminal hidden state is not treated as an MTP embedding. MTP is optional and separate
from the prefill/TTFT benefit. No general decode-speed claim follows from the smoke test.

## Inspect performance

```python
print(answer.stats.request_ttft_s)
print(answer.stats.reader_prefill_s, answer.stats.translation_s)
print(answer.stats.target_prefill_s, answer.stats.state_install_s)
print(answer.state.info)
```

`target_prompt_tokens` is cumulative for a state lineage. It stays zero for strict
composite input; cached-handoff follow-ups and target branch questions increase it.
Warm TTFT includes queueing and the work needed to produce the first token. It excludes
model downloads and loading. Preserve the first call's stats if an answer needs more
`generate()` calls. [Benchmark definitions](benchmarks.md).
