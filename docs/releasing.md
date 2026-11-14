# Releasing installation artifacts

The 0.2.0 release separates a lightweight Python package, optional inference
dependencies, GPU images, and pinned model assets. Model weights never enter a
wheel or Docker layer. Existing 0.1 model/native artifacts remain available.

## Maintainer setup

1. Verify the `latentport` project name in the maintainer PyPI account. A 404 or
   pending Trusted Publisher does not reserve a name.
2. Configure GitHub Trusted Publishers on PyPI and TestPyPI for
   `LatentPort/latentport`, workflow `release.yml`, and the matching `pypi` or
   `testpypi` environment. Protect these environments with maintainer reviewers.
3. Protect the `gpu-release` environment. Image publication requires recorded
   real-GPU validation; a successful CPU image build is insufficient.
4. Give the workflow package write access and make the GHCR package public.
   Verify `docker pull` succeeds without a registry login.

## Validation and publication

- Run the Package checks workflow, including installed-wheel checks on Windows,
  Linux and macOS, CPU contracts, and the Linux GGUF image build.
- Build both Docker targets locally. Record the base image digests, resolved
  `pip freeze`, native ABI, driver, GPU, and model identities with release evidence.
- On each advertised GPU/OS combination run `doctor`, the headless transfer demo,
  and state save/load/fork tests. Exercise first download, restart after interruption,
  cache reuse, and offline operation. BF16 requires a separate real inference check.
- Run Publish release with `destination=testpypi`. Install that artifact with
  `pip install --index-url https://test.pypi.org/simple/ --no-deps latentport==0.2.0`
  in a fresh environment and run `scripts/check_distribution.py`.
- Dispatch from the same reviewed commit with `destination=pypi`. The build job
  checks and uploads artifacts; the publishing job downloads those artifacts
  without rebuilding. Enable `publish_images` only after GPU qualification.
- Confirm `pip install latentport` from the normal PyPI index, anonymous image
  pull, and the exact README command. Remove the pending-release note only then.
- Tag the reviewed source commit and attach validation reports. Never replace
  an already published version or silently move a versioned image tag.

Base image digests are pinned in `docker/Dockerfile`. Direct inference dependencies are pinned in `docker/constraints.txt`. Record the
full resolved dependency list for each image; this file is not a transitive lock.
When changing the package version, update the versioned image tags and Docker
wheel filename at the same time. Public publish workflows are deliberately manual.

## External base-model migration

The bridge repository only needs the bridge, its license, model card, and
compatibility/verification metadata. Base models can live in other Hugging Face
repositories. `scripts/pin_gguf_profile.py` writes candidate manifests from exact
Hub revisions, filenames, byte counts and LFS SHA-256 values. This does not qualify
the files for inference.

`docs/upstream-gguf-candidate.json` contains a metadata-pinned candidate using
LM Studio's Qwen3.5-4B and Bartowski's Qwen3.8-27B Q4_K_M files. It is not the
default and has not been GPU-qualified. No base-model files need to be copied
into the bridge repository to test or subsequently adopt this profile.

Run `latentport demo --headless --model-manifest candidate.json` on the qualified
GPU, then check save/load, continuation, tokenizer parity and bridge behavior.
Validate MTP separately if the profile enables it. Different Q4_K_M conversions
can change both precision and metadata. Do not transfer benchmark claims from
the previous files without measurement.

Only after those checks replace `src/latentport/configs/gguf-models.json`. Keep
older pinned files for old releases/saved states. The initial 0.2.0 code retains
the current qualified defaults rather than silently selecting untested files.

## Native runtime maintenance

The Python package uses the existing ABI-2 native release artifacts. New builds
must pin the patched llama.cpp revision and record OS, CUDA, compute capability,
ABI, byte size and SHA-256 in `native-release.json`. Windows CUDA runtime archives
are independently pinned in `cuda-windows.json`; preserve their supplied notices.
Do not bundle NVIDIA drivers. Add GPU architectures only after executing tests
on those architectures, including minimum driver compatibility.
