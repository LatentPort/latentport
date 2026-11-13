# LPCompact 0.1

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

## One selector for BF16 and GGUF

The [LPCompact model](https://huggingface.co/latentport/LPCompact) contains one
selector checkpoint. The same weights serve Qwen3.8-27B BF16 and GGUF weight
quantizations. Weight precision is independent of the selector: each adapter
exposes the same complete hybrid-state layout with BF16 attention keys and values.
Model identities still bind snapshots to the exact base weights and runtime that
created them.

Install the [LatentPort 0.1 release wheel](https://github.com/LatentPort/latentport/releases/tag/lpcompact-v0.1).
For BF16, use an environment with CUDA-enabled PyTorch and the CUDA compiler.
Install the extra against that PyTorch installation:

```shell
pip install --no-build-isolation "latentport[compact] @ https://github.com/LatentPort/latentport/releases/download/lpcompact-v0.1/latentport-0.1.0-py3-none-any.whl"
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
compactions. Use the model's existing native logical-position range.
