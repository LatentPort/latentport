# Installation

The first preview targets **Windows x64 + RTX 5090 (32 GB)**. The native binary is built
for compute capability 12.0. Linux, macOS, AMD, CPU-only execution and other NVIDIA
architectures are not qualified by this release.

## Prerequisites

- Python **3.11 x64**.
- An NVIDIA driver that supports CUDA 13.0. The validation machine used driver 610.88.
- [CUDA Toolkit 13.0](https://developer.nvidia.com/cuda-13-0-0-download-archive), which supplies
  the CUDA 13 runtime and cuBLAS DLLs used by the native backend. PyTorch's CUDA 12.8
  wheel does **not** supply those CUDA 13 DLLs.
- [Microsoft Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
- Approximately **35 GB of free disk space** for runtime dependencies and downloaded
  artifacts. Use more space if saving long contexts or keeping extra environments.
- Enough free VRAM for both models, bridge and caches. Close other GPU workloads.

The supplied native runtime avoids a compiler installation. CUDA Toolkit runtime
libraries are still required. The qualified configuration holds both models on the GPU;
it does not silently fall back to CPU offload when allocations fail.

## Install from the repository

```powershell
git clone https://github.com/LatentPort/latentport.git
cd latentport
$env:LATENTPORT_HOME = 'F:\LatentPort\runtime'  # any suitable bulk-storage directory
$env:PIP_CACHE_DIR = "$env:LATENTPORT_HOME\pip-cache"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
python -m latentport install-native
python -m latentport doctor
```

`doctor` checks Python, CUDA availability, the GPU and native ABI without loading the
models. A successful report says `ready_for_model_load: true`; it is not a model-quality test.

The package is a development preview. Install this repository or the versioned preview
wheel; do not assume that an unrelated PyPI package named `latentport` is this release.
The maintainer must push this checkout before the clone command exposes these changes.

## Download once, run repeatedly

```powershell
python -m latentport download
python examples/chat.py --prompt "Explain how database indexes work."
```

The download command fetches the exact Q4_K_M reader and target, the FP32 bridge and
pinned tokenizer metadata. Approximately 22 GB is downloaded on the first run; later
runs reuse the cache. The target includes its trained MTP head. Hashes and revisions
are pinned in `latentport/configs/gguf-models.json` and `native-release.json`.

`lp.load(backend="gguf")` also downloads missing model assets automatically. It never
selects a different model pair or substitutes another quantization. `port.warm()` loads
the GPU allocations before requests; downloads, file validation and cold loading can
take substantially longer than warm TTFT. Keep one port alive for the conversation.

Set `LATENTPORT_HOME` in each shell, or persist it in your environment. Without it,
the default is `~/.cache/latentport`. Tokenizer metadata is downloaded from the pinned
upstream model revisions, even when language-model weights use GGUF.

## Use existing model files

```python
import latentport as lp

port = lp.load(
    backend="gguf",
    source_path="F:/models/Qwen3.5-4B-Q4_K_M.gguf",
    target_path="F:/models/Qwen3.8-27B-MTP-Q4_K_M.gguf",
    native_options=lp.NativeOptions(context_size=18432, mtp_drafts=4),
)
```

Use the exact model revisions in the bridge manifest. Same-size fine-tunes or arbitrary
GGUFs are not interchangeable. The published exact Q4_K_M pair is the qualified path.
`bridge_gguf_path=` can also supply the local bridge. It must match the selected bridge hash.
An explicit local canonical bundle can be passed as the first `lp.load(...)` argument.

For a target without MTP weights, use `mtp_drafts=0`; `mtp=True` will then fail explicitly.
Other quantizations require separate qualification. `context_size` includes the prompt,
all replies and follow-up turns; 18432 is the tested allocation for a ~16K prompt.
Context truncation or eviction is not automatic.

## Offline use

Download everything while online, then set `$env:HF_HUB_OFFLINE='1'`. Keep the same
`LATENTPORT_HOME`. Local GGUF paths alone do not supply bridge/config/tokenizer files.
A missing cached artifact produces an error rather than downloading an alternative.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `cuda_available: false` | Confirm the CUDA-enabled Torch wheel and NVIDIA driver; check `nvidia-smi`. |
| Native library missing | Run `python -m latentport install-native` in the same cache environment. |
| DLL load error / WinError 126 | Check CUDA 13.0 runtime/cuBLAS and the VC++ redistributable; keep all downloaded native DLLs together. |
| Stock llama.cpp / ABI mismatch | Use the pinned LatentPort native download, not a system llama.cpp library. |
| CUDA allocation failure | Close GPU workloads, release unused answer/state objects or reduce context capacity. |
| New question rejected | The previous answer is unfinished. Continue with `generate()` before another `ask()`. |
| Hub access error | Check network access and the configured HF login; the model releases are public. |
| GPU lost / driver timeout | Inspect system and driver logs. Recover the GPU before another model run; an application restart alone may not suffice. |

## Source builds and alternate backends

The prebuilt archive includes corresponding native source and upstream licenses.
See [native-abi.md](native-abi.md) to build the pinned extension yourself. A source build
on a new platform is not equivalent to qualification. The Transformers/NF4 reference
backend is available via `pip install ".[reference]"` and `backend="reference"`, but
its package integration has not received the GGUF release-path checks. There is no
vLLM backend in this preview.
