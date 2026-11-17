# Releasing installation artifacts

LatentPort separates a lightweight Python package, optional inference
dependencies, GPU images, and pinned model assets. Model weights never enter a
wheel or Docker layer. LPCompact is a compaction feature of the shared library;
feature updates do not create separate GitHub product releases.

Python distributions are published on PyPI. Native binaries are distributed as versioned GitHub release assets with SHA-256 checksums. The bridge Hub contains inference parameters, bridge constants, and metadata.

Keep validation environments, downloaded SDKs, native builds and release archives
outside the source checkout, using a separate bulk directory such as
`F:/LatentPort/runtime` on Windows. Point `LATENTPORT_HOME`, `PIP_CACHE_DIR` and
build output directories there. `.validation/` is ignored, excluded from source
archives, and rejected by CI if tracked.

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
  `pip install --index-url https://test.pypi.org/simple/ --no-deps latentport==0.4.4`
  in a fresh environment and run `scripts/check_distribution.py`.
- Dispatch from the same reviewed commit with `destination=pypi`. The build job
  checks and uploads artifacts; the publishing job downloads those artifacts
  without rebuilding. Enable `publish_images` only after GPU qualification.
- Confirm `pip install latentport` from the normal PyPI index, anonymous image
  pull, and the exact README command. Remove the pending-release note only then.
- Record the reviewed source commit and validation reports. Never replace
  an already published version or silently move a versioned image tag.

Base image digests are pinned in `docker/Dockerfile`. Direct inference dependencies are pinned in `docker/constraints.txt`. Record the
full resolved dependency list for each image; this file is not a transitive lock.
When changing the package version, update the versioned image tags and Docker
wheel filename at the same time. Public publish workflows are deliberately manual.

## Add a bridge

Publish one directional pair per Hub repository. Its `latentport.json` uses schema `latentport.bridge-bundle.v2`. For the supported `qwen-hybrid-mixed-v1` architecture, the bundle provides the endpoint geometry, architecture settings, calibration, component weights, trained thinking-mode order and GGUF bridge. Record each artifact's byte size and SHA-256. A new 4B or 9B pair using this graph can be selected by repository and commit without changing the runtime default.

Pin each source and target tokenizer, template and model shard to its upstream repository and full commit. Use `scripts/pin_gguf_profile.py` to collect an external GGUF publisher's immutable file receipts. Populate `gguf_models` in the candidate bundle and test that local bundle before publishing. Compare both thinking modes, query-boundary handoff, native continuation and multi-turn chat on each backend. Each quantization requires qualification.

The runtime's Python API accepts `lp.load(repo, revision=commit, backend="nf4")`. Use `download --repo` or `demo --bridge-repo` and supply `--revision` to pin either command. After publishing, check the bridge file list and verify a fresh download. Change the default bridge and revision together when superseding an earlier release.

## Native runtime maintenance

Pin the patched llama.cpp revision, platform, CUDA version, compute capability, archive size and SHA-256 in `configs/native-release.json`. Publish native binaries as GitHub release assets and verify anonymous downloads. Windows CUDA redistributables have independent pins in `configs/cuda-windows.json`. Retain the supplied license notices.
