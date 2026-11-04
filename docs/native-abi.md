# Native GGUF runtime

`latentport._native` binds to a LatentPort-enabled llama.cpp DLL through a
versioned C ABI (`2`). Both language models execute GGUF weights through CUDA
llama.cpp. The custom F32 bridge GGUF executes through the same Torch CUDA
translator graph as the canonical bridge. It is not a llama.cpp language model
and is not a quantized GGML translation graph.

The implementation is pinned to llama.cpp
`b19cbe925be361d229f0fe03435affe4a2717f37`. `native/patch_llama.py` makes three
small changes in an isolated checkout: typed cache access, device capture nodes
in Qwen's graph, and selection of multiple appended Qwen MTP heads. It does not
modify the research checkout. Arbitrary stock llama.cpp versions are rejected.

## Install the supported native binary

For the Windows x64 / RTX 5090 preview, run `python -m latentport install-native`.
The package downloads a pinned archive, verifies its SHA-256, and installs matching
DLLs together under `LATENTPORT_HOME/native/build/bin`. The archive includes native
source, the llama.cpp source pin and dependency licenses. CUDA 13.0 runtime/cuBLAS
and the Microsoft VC++ redistributable must be installed separately.

Use `python -m latentport doctor` to check dependencies before loading models.
See [installation.md](installation.md) for the supported path. The source-build steps
below are for contributors; users of the qualified binary do not need a compiler.

## Build and load

Keep the isolated source, compiler outputs and model data on bulk storage.
For the current Windows workstation:

```powershell
# Source must be a disposable checkout of the exact commit above.
.\scripts\build_native.ps1 -Python python -Jobs 2
$env:LATENTPORT_NATIVE_LIB = 'F:\LatentPort\runtime\native\build\bin\llama.dll'
python scripts/check_native_abi.py
```

The build script uses MSVC 14.44, CUDA 13.0, CMake/Ninja and CUDA architecture
120. It compiles CUDA code and runs a CPU test; it does not execute models.
The `native/CMakeLists.txt` can also be configured directly with
`LATENTPORT_LLAMA_SOURCE`. Linux builds and other GPU architectures have not
been validated. Keep `llama.dll` and the matching `ggml*.dll` together. The CUDA
runtime and cuBLAS libraries must be available. Native code is included in the
source distribution; the Python wheel does not bundle these large binaries.

```python
import latentport as lp

port = lp.load(
    backend="gguf",
    source_path="F:/LatentPort/models/reader.gguf",
    target_path="F:/LatentPort/models/target-with-mtp.gguf",
    native_options=lp.NativeOptions(context_size=18432, batch_size=512,
                                    threads=8, mtp_drafts=4),
)
residency = port.warm(mtp=True)
answer = port.composite.session().ask("Explain this document ...", mtp=True)
print(answer.stats)
```

Use GGUFs of the exact model revisions in the bundle, converted with the
pinned Qwen layout. Arbitrary same-size fine-tunes are not interchangeable.
The target GGUF must contain its trained appended NextN/MTP weights to enable
MTP. Set `mtp_drafts=0` for a GGUF without them; requesting `mtp=True` then raises
an error. The runtime does not download or substitute another draft model.
Both models and the bridge remain resident. Allocation failure is explicit;
the engine does not silently spill weights or working caches to CPU.

## Complete state and ownership

There are 65 source fields and 129 target fields: terminal H, attention K/V,
convolution C and recurrent R. The typed installer checks every field before
mutating its private working cache. It installs target positions and sequence
metadata at the target tokenizer's clock. It never calls target prefill as a
fallback. The first target logits come from H through final RMS norm and the
quantized GGUF output projection.

The bridge's canonical coordinates differ from llama.cpp in three places:

* Attention is head-major in the bridge and token-major in native cache storage.
* Recurrent matrices are transposed, and Qwen's GGUF converter changes value
  heads from grouped order to tiled order. Both transformations are inverted.
* The canonical convolution snapshot has four positions. llama.cpp needs the
  last three for execution. Device graph taps retain the extra position and
  invert the value-channel ordering when exporting snapshots.

The native state ABI retains K/V in BF16 and H/C/R in FP32. Translated H/C are
accepted in BF16. The bridge receives BF16 H/C copies; this conversion never
changes the original native snapshot. Keeping native H/C in FP32 avoids
introducing a rounding step each time a session is saved or restored.

Captures own CUDA tensors independent of llama.cpp's ephemeral sequence
snapshots. `fork()` shares those immutable tensors in O(1). Installing an active
branch copies into a private native working cache once. **This implementation
does not provide paged attention-prefix sharing between concurrently active
native sequences.** Requests on the one-GPU engine queue for complete generation
transactions; they do not alternate full-cache copies on each token. Interleaved
inactive handles park to owned device snapshots. Save/load moves state to disk
only when requested.

Torch's allocator cannot see llama.cpp's memory needs. Under VRAM pressure,
the runtime releases unused Torch scratch blocks at translation, snapshot and
request boundaries. `port.warm()` reports `torch_cache_reclaims`. The 16K
two-turn check retained both answer states and finished with about 2 GB free
after this cleanup; that is an endpoint measurement, not a peak guarantee.
Additional retained states still consume their own tensor storage.

Graph taps are CUDA graph copy nodes, not per-layer Python callbacks. Complete
states stay on the GPU during capture, translation and installation. Native
MTP currently transfers small next-token hidden rows through llama.cpp's host
interface; it does not transfer the long attention cache through host memory.

## MTP commit protocol

Generation currently uses greedy target decoding. MTP verifies multiple tokens
in one target pass; it is not a loop labeled speculative while still evaluating
every accepted token separately.

1. Emit the first token from the installed H, then consume it through the target.
2. Obtain the target's actual `embeddings_nextn` representation. Translated H
   itself is not assumed to be this representation.
3. Draft from the generated suffix, starting with the committed hidden row.
   A single trained head may be unrolled. Multiple trained heads are selected
   by layer offset and build their own speculative prefix as in the pinned
   upstream MTP implementation. Historical prompt tokens are not replayed.
4. Verify the first selected token plus draft tokens in a batched target pass.
   Accept the matching greedy prefix, stopping at EOS or the request limit.
5. Remove rejected attention positions, materialize the selected recurrent
   rollback slot, restore four-position convolution history and terminal H.
   A published state never depends on a pending rollback to be applied later.
6. Rebuild only the draft's newly committed suffix from the correct target
   hidden rows. Rejected draft KV and hidden rows do not survive the transaction.

`Answer.stats` separates requested MTP from actual speculative batches, proposals,
accepted drafts and ordinary warmup tokens. A one-token request cannot claim an
MTP speedup. MTP draft state is an execution optimization, not part of the portable
129-field state. Restoring or switching a snapshot restarts MTP with an ordinary
generated-token warmup; it does not replay the prompt. Acceptance with a draft
that has no historical prompt cache must be measured on the real models.

The private `lp_test_force_drafts` hook changes proposals only. It permits the
GPU smoke test to exercise accepted and rejected branches deterministically;
target verification remains active. Normal inference never calls this hook.

## Verification boundary

Compilation, C++ CPU layout/acceptance tests, Python ownership/transaction tests,
and exact comparison of all 262 GGUF bridge tensor payloads can run without the
GPU. These checks do not establish numerical correctness, MTP throughput,
maximum context capacity, or TTFT speedup of the complete runtime.

When the GPU is available, run `scripts/smoke_gguf.py --gpu-is-free` with both
GGUF paths. It checks
native capture/restore, zero-target-prompt composite handoff, ordinary versus
MTP continuation, deterministic rejection/acceptance, persistence, branching,
live escalation, exact multi-turn reader synchronization and resident memory.
It writes its report to bulk storage and does not run a model-quality evaluation
campaign. Optional `--performance-contexts 512 16384` measures warm API TTFT
against native 27B through the same package.

The local Q4_K_M pair passed this bounded check on the RTX 5090; see
[runtime-validation.json](runtime-validation.json) for model hashes, timings,
memory accounting and limits. Its target has one trained MTP head, unrolled for
up to four proposals. Multiple distinct trained heads and other hardware remain
outside this check. MTP matched all 24 ordinary greedy tokens and passed forced
acceptance/rejection restoration. It accepted 14 of 33 proposals; generation
took 341 ms versus 367 ms ordinarily. That short sample is not a general MTP
throughput claim.

The selected bridge omits the frozen parent's unused K/V projections. All 129
final outputs matched the unpruned implementation exactly in the real-state
check. Live chat preserves committed reasoning and whitespace; it does not
re-render prior turns through a template that can remove them.

`scripts/check_long_chat.py` repeats the bounded 16K two-turn memory case. Pass
`--gpu-is-free`, both GGUF paths, and an `--output` JSON path on bulk storage.
The measured follow-up took 1.11 s versus 148 ms for a persistent native session
with the same committed history. Initial prompt acceleration therefore does
not establish an advantage for subsequent questions over an existing native
27B cache.
