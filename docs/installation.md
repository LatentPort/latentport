# Installation

The runtime fetches the bridge and its pinned base models from their respective publishers. The Docker demo uses an NVIDIA SM120 GPU with at least 32 GB VRAM. Allow 40 GB of storage for downloads and extraction, plus Docker build storage. Build the image from this checkout using the commands below.

## Install the Python library

Use a virtual environment with Python 3.11 or later:

```shell
python -m pip install latentport
latentport --help
latentport doctor
```

To upgrade, run `python -m pip install --upgrade latentport`. Run `latentport install-native` when upgrading the GGUF backend.

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

The default bridge is `latentport/LPBridge-Qwen3.5-4b-Qwen3.8-27b`. BF16 and NF4 use the pinned upstream Qwen revisions. GGUF uses pinned Q4_K_M source and UD-Q4_K_M target files from Unsloth. The runtime downloads each on first use and verifies its SHA-256. To reuse matching local files, pass `source_path` and `target_path`, or the CLI options `--source-path` and `--target-path`.

An external Hub mirror of the exact model files can be configured with:

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

## Serve LPCompact with vLLM

Use the pinned vLLM 0.34.0 image or install the adapter and apply its bundled
patch in an existing vLLM environment. All seven LPCompact base model profiles
can load BF16, NF4, or GGUF weights. The stock image defaults to BF16. For NF4,
install the NF4 extra, which includes the bitsandbytes plugin. For GGUF, install
the GGUF extra and the pinned vLLM GGUF plugin. Set `--weight-format` to choose
the loader. [vLLM installation and model matrix](../integrations/vllm/README.md#installation)

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
