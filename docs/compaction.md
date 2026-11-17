# State compaction with LPCompact

LPCompact adds `state.compact()` to LatentPort's library of LLM state operations.
It uses a small learned selector to retain useful attention memory while
preserving the complete recurrent state, convolution state and continuation clock.

```python
import latentport as lp

with lp.load_model(backend="bf16") as model:
    state = model.prefill(history)
    state = state.compact()
    answer = model.session(state).ask("Continue the task.")
```

LatentPort 0.4.0 adds model-specific selectors for Qwen3.5-4B and Qwen3.5-9B.
Pass the base model ID to select its pinned revision and matching selector.

```python
with lp.load_model("Qwen/Qwen3.5-4B", backend="bf16") as model:
    compacted = model.prefill(history).compact()

with lp.load_model("Qwen/Qwen3.5-9B", backend="gguf", model_path=gguf_path) as model:
    compacted = model.prefill(history).compact()
```

Each selector reads the attention values produced by its own frozen base.
The runtime verifies the selected checkpoint's SHA-256 before scoring.
The default model is Qwen3.8-27B.

For Qwen3.5-4B and Qwen3.5-9B, the release evaluation compares exact answers
with full memory and 8× compaction on 128 paired questions per model. It covers
repository lookup, latest decisions, structured records and two-hop document
lookup at 64K and 128K. [Per-task scores and backend qualification](lpcompact-qwen35-validation.json)

Compaction produces an ordinary immutable `State`. The original state and its
forks remain valid. Reassigning `state` releases discarded KV tensors when their
last owner is released. The compacted state supports the same `fork()`, `save()`
and `State.load()` operations, and the same model's append and session APIs.

```python
compact = state.compact(ratio=8)
branches = compact.fork(2)
compact.save("saved-state")
restored = lp.State.load("saved-state")
answer = model.session(restored).ask("What should we do next?")
print(compact.compaction)
```

The selector operates directly on state tensors. No text summary or historical
token replay is needed for the operation.

## One selector for NF4, BF16 and GGUF

The [LPCompact-Qwen3.8-27B model](https://huggingface.co/latentport/LPCompact-Qwen3.8-27B) contains one
selector checkpoint. The same weights serve Qwen3.8-27B NF4, BF16 and GGUF weight
quantizations. Weight precision is independent of the selector: each adapter
exposes the same complete hybrid-state layout with BF16 attention keys and values.
Model identities still bind snapshots to the exact base weights and runtime that
created them.

Each trained state operation has its own Hugging Face repository, named for its
operation and compatible model. The shared Python runtime remains in the
[LatentPort GitHub repository](https://github.com/LatentPort/latentport).

Install LatentPort from PyPI (0.2.1 or later).
For BF16, use an environment with CUDA-enabled PyTorch and the CUDA compiler.
Install the extra against that PyTorch installation:

```shell
pip install --no-build-isolation "latentport[bf16]"
```

The BF16 backend uses the pinned Transformers runtime and CUDA recurrent kernels.

For GGUF, install the LatentPort native runtime with
`python -m latentport install-native`, or build the extension from source, and
select the base model file. Prebuilt runtimes target Windows x64 / CUDA 13 and
Linux x64 / CUDA 12.8 on Blackwell SM120:

```python
with lp.load_model(
    backend="gguf",
    model_path="Qwen3.8-27B-Q4_K_M.gguf",
    native_options=lp.NativeOptions(context_size=132096, mtp_drafts=0),
) as model:
    state = model.prefill(history)
    state = state.compact()
    answer = model.session(state).ask("Continue the task.")
```

GGUF quantizations use the same selector file. The native build retains absolute
attention positions, restores the recurrent endpoint, and resizes its working
cache when it activates a compacted snapshot. It grows that cache as new tokens
arrive. Existing translator loading through `lp.load()` is unchanged.

## Policy and inspection

The default ratio is 8; ratio 4 is also available. Below 65,536 physical KV
positions, `compact()` returns the original state unchanged. The policy protects
the first and last 512 retained positions and uses learned span selection for
the remaining budget. The minimum retained budget is 1,024 positions.

`compact.compaction` records physical occupancy, attention tensor bytes, elapsed
time, the number of learned positions, and reduction relative to accumulated
logical history. These are snapshot metrics; other snapshots and the native
working cache have their own allocation lifetimes. `force=True` enables smaller
diagnostic runs with the same budget rules.

The logical clock continues from the original history, including after repeated
compactions. `state.clock` counts every consumed token, including chat framing
and generated replies. `state.physical_kv_length` reports the actual attention
sequence dimension. Compaction changes physical occupancy while preserving the
original positions, recurrent state, convolution endpoints and continuation clock.

## Continuous histories beyond the original window

Standalone NF4, BF16 and GGUF support two static position policies:

| `position_policy` | Maximum logical clock | Maximum physical KV positions |
| --- | ---: | ---: |
| `"default"` | 262,144 | 262,144 |
| `"yarn4"` | 1,048,576 | 262,144 |

Choose the policy at model loading, before processing any input. YaRN4 uses the
pinned Qwen3.8-27B revision and its original rotary geometry, with static factor
4 throughout the lineage. Position rules and quantization are included in state
identity; incompatible snapshots are rejected. Existing default-policy BF16 and
GGUF identities remain compatible. The pretrained translator-pair API keeps its
original position rules.

```python
import latentport as lp

with lp.load_model(backend="nf4", position_policy="yarn4") as model:
    state = model.prefill(initial_context)
    for token_ids in new_input_chunks:
        if state.physical_kv_length + len(token_ids) > 258_048:
            state = state.compact(ratio=8)
        state = model.append(state, token_ids)
    state.save("million-token-state")
```

This is the ordinary immutable `State` flow. Each input chunk must fit in the
remaining physical and logical budgets. Compaction preserves the clock; the
final logical limit still applies even when substantial KV memory is free.
The 4,096-position headroom in this example leaves space for question framing
and generated tokens within the 262,144 physical bound.

For small incremental chunks, `model.session(resident=True)` keeps a resident working cache.
`model.live()` is an alias returning the same `Session` type.
It retains a bounded original-token tail for message boundaries and materializes
immutable states at snapshot, transformation and generation boundaries:

```python
with lp.load_model(backend="nf4", position_policy="yarn4") as model:
    with model.live() as live:
        for token_ids in new_input_chunks:  # e.g. 1,024 tokens per chunk
            if live.physical + len(token_ids) > 258_048:
                compacted = live.compact(ratio=8)
                # live.compact() freezes a State, calls State.compact(), then
                # installs a private working cache. Release unused snapshots.
                del compacted
            live.append(token_ids)
        state = live.snapshot(release=True)
    state.save("streamed-state")
    answer = model.session(state).ask("What did the first record say?")
```

`live.append()` tokenizes only new text or accepts exact token IDs. It does not
decode or replay historical input. Forks and published snapshots remain immutable.
A failed append closes its working handle; restore an exact snapshot to resume.
Failed compaction retains the preceding immutable state; the next append reinstalls it without replay. `live.compact()`
and `state.compact()` use the same frozen selector and policy.

Use `backend="bf16"` for BF16 weights, or `backend="gguf"` with `model_path` for
a GGUF file. `max_kv_positions` can lower the physical ceiling for a smaller
memory budget. Both capacities remain finite, and fitting a particular policy
depends on weights, working caches, snapshots and transient allocations.

## Installing these backends from source

Install CUDA PyTorch first, then the relevant extra from this checkout:

```shell
pip install --no-build-isolation ".[nf4]"
# Or for BF16:
pip install --no-build-isolation ".[bf16]"
```

NF4 uses double-quantized 4-bit weights, BF16 computation/KV/conv state and FP32
recurrent state, on CUDA:0, with CPU/disk weight offload disabled. MTP is disabled
for the standalone Transformers backends.

GGUF extended positions require a native library rebuilt from this checkout.
The additive `lp_open_extended` entry point carries the static position policy
and physical limit into C++; older native binaries continue serving default
positions and report an explicit rebuild error for extended positions. Existing
prebuilt archive hashes are unchanged.

On Windows, `scripts/build_native.ps1` builds the current native sources. Select
the resulting library explicitly:

```python
options = lp.NativeOptions(
    context_size=262_144, mtp_drafts=0, library="F:/my-build/bin/llama.dll"
)
model = lp.load_model(
    backend="gguf", model_path="Qwen3.8-27B-Q4_K_M.gguf",
    position_policy="yarn4", native_options=options,
)
```

The logical clock and retained positions are preserved across native cache
resizes. A smaller initial native context can grow to the physical ceiling
under the extended policy. Default-policy legacy capacity behavior is retained.

This supports bounded attention memory across an extended, finite logical
history. Retrieval quality and latency depend on the workload; the API does not
promise unlimited context or native-window accuracy across a million tokens.
