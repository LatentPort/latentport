# Installation

LatentPort 0.2.0 is available on PyPI as a lightweight package with optional
backend dependencies. The GHCR images are pending GPU qualification and
publication; source and local Docker build instructions are below.

## Try the demo with Docker

Prerequisites: an SM120 NVIDIA GPU with at least 32 GB VRAM, a compatible driver,
Docker with GPU access, and enough space for the image, 22.10 GB of model files,
and download/extraction scratch space. Start with at least 40 GB free; long saved
contexts and Docker build caches require additional space. Actual VRAM use depends
on context and other workloads. Linux needs NVIDIA Container Toolkit configured;
Windows needs Docker Desktop with its WSL 2 backend. No host CUDA toolkit is needed.

```shell
docker run --rm --gpus all -p 127.0.0.1:8768:8768 -v latentport-cache:/cache ghcr.io/latentport/latentport:0.2.0-gguf-cu128-sm120
```

Open http://localhost:8768. The UI reports setup stages before the interactive
transfer/branch/repair demonstration. Downloads are cached in the named volume.
Do not delete the volume to update the image. The first run downloads models;
first-run setup is not included in the displayed inference TTFT measurement.

For automated execution, append `demo --headless`. The exit code is nonzero on
failure and the cache contains the evidence under `demos/`. This image supports
Linux x86-64 / CUDA 12.8 / SM120; it does not enable CPU, Apple GPU, AMD, or other
NVIDIA architectures. The BF16 image uses tag `0.2.0-bf16-cu128-sm120`; model size
and context determine its larger VRAM requirement. Its default command is
`doctor --backend bf16`, not the GGUF transfer demo.

## Install the Python library

Use a virtual environment with Python 3.11 or later:

```shell
python -m pip install latentport
latentport --help
latentport doctor
```

The base package does not install Torch, Transformers, CUDA or model weights.
`doctor` emits JSON and exits with code 1 when inference prerequisites are missing;
that is not an installation failure. CLI help and the browser layout preview work
without inference dependencies:

```shell
latentport demo --ui-only
```

### GGUF inference

Install the CUDA PyTorch build for your platform first. The qualified recipe uses
Torch 2.11.0 with CUDA 12.8; a plain PyPI Torch install is not a portable way to
select the CUDA variant, particularly on Windows.

```shell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "latentport[gguf]"
latentport install-native
latentport doctor --backend gguf
latentport demo
```

`demo` installs the native runtime if missing, checks the GPU, downloads the exact
model profile, then runs the packaged browser app. It does not install Python
packages into your environment. On Windows `install-native` also downloads pinned
CUDA 13 runtime/cuBLAS DLLs and their notices (about 403 MB compressed), avoiding a
full CUDA toolkit install. The qualified Windows native binary uses CUDA 13 even
though Torch uses CUDA 12.8. A driver supporting both is required; the previously
qualified machine used driver 610.88. The Microsoft Visual C++ x64 runtime may be
required. `--no-cuda` uses your existing CUDA runtime libraries instead.

On Linux the native binary uses CUDA 12.8. Use the container for an assembled
runtime environment, or supply compatible CUDA runtime/cuBLAS libraries on the
loader search path. Native archives are checksum-verified and installed into
versioned directories. Activation happens after extraction and dependency setup.
`LATENTPORT_NATIVE_LIB` can explicitly select a custom compatible library.

### BF16 and reference inference

BF16 kernels require a CUDA build environment when installing from source. Prefer
the BF16 container if you do not want to build these locally. Install CUDA Torch
first, followed by the pinned backend extra:

```shell
python -m pip install --no-build-isolation "latentport[bf16]"
latentport doctor --backend bf16
```

The `compact` extra remains an alias for BF16 dependencies. The `reference` extra
adds the NF4/BF16 reference transfer dependencies, including bitsandbytes.
GGUF also requires Torch for bridge execution and GPU state tensors; it is not a
Torch-free backend. Diagnostics check prerequisites, not model quality or whether
all selected model weights are already cached.

## Cache and base-model repositories

Set `LATENTPORT_HOME` to a writable directory with sufficient free space. It defaults
to `~/.cache/latentport`; Docker uses `/cache`. Exact compatible files in the normal
Hugging Face cache are reused before downloading into LatentPort's cache. Set
`HF_HUB_OFFLINE=1` after preparing the cache to run without Hub access. Native setup
must already be complete for an offline run.

A base model does not need to be hosted in the LatentPort repository. The model
manifest records each file's repository, full commit revision, filename, byte size
and SHA-256. The bridge remains in its own repository. Until external replacements
pass qualification, the shipped defaults preserve the previously tested files.

To test an externally pinned model pair:

```shell
latentport download --model-manifest models.json
latentport demo --headless --model-manifest models.json
```

The same profile can be passed to `lp.load(backend="gguf", model_manifest="models.json")`.
Use `scripts/pin_gguf_profile.py` to generate candidate manifests without downloading
weights. Candidate files need compatible geometry, tokenizer streams and native
runtime metadata. Same model name/quantization does not establish compatibility.
The demo disables MTP; a target without MTP can run with `NativeOptions(mtp_drafts=0)`.
Other MTP configurations require separate qualification. Existing snapshots remain
bound to exact model weights and cannot silently switch to another quantization.

Existing local files can also be supplied as `source_path`, `target_path` and
`bridge_gguf_path` in `lp.load()`. These must satisfy the normal compatibility checks.

## Build from this checkout

```shell
python -m pip install -e ".[gguf]"
latentport demo
```

Install CUDA Torch first as above. For Docker:

```shell
docker build -f docker/Dockerfile --target gguf -t latentport:local .
docker run --rm --gpus all -p 127.0.0.1:8768:8768 -v latentport-cache:/cache latentport:local
```

Use `--target bf16` for the BF16 image. The compiler exists only in the GGUF build
stage; the BF16 target retains CUDA development tools for its runtime JIT kernels.
See [releasing](releasing.md) for publishing, qualification and migration gates.
