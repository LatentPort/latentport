"""Small installation and diagnostics CLI: python -m latentport --help."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import zipfile
import os
import tempfile
import shutil
import sys
import uuid
from urllib.request import urlopen

from .hub import bulk_home


def install_native(*, with_cuda=True):
    """Install a pinned, checksum-verified native runtime distribution."""
    system=platform.system()
    if system not in ("Windows", "Linux") or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("Prebuilt native downloads support Windows/Linux x64; see docs/installation.md")
    spec = json.loads((Path(__file__).parent / "configs/native-release.json").read_text())
    if system == "Linux":
        if "linux" not in spec:
            raise RuntimeError("This release has no pinned Linux native download")
        spec=spec["linux"]
    if "url" in spec:
        cache=bulk_home()/"downloads"
        cache.mkdir(parents=True,exist_ok=True)
        artifact=cache/(spec["sha256"]+".zip")
        if not artifact.is_file():
            fd,temporary=tempfile.mkstemp(dir=cache,suffix=".partial")
            try:
                with os.fdopen(fd,"wb") as out, urlopen(spec["url"],timeout=60) as response:
                    for block in iter(lambda:response.read(4*1024*1024),b""):
                        out.write(block)
                if hashlib.sha256(Path(temporary).read_bytes()).hexdigest()!=spec["sha256"]:
                    raise RuntimeError("Native archive checksum mismatch")
                Path(temporary).replace(artifact)
            finally:
                Path(temporary).unlink(missing_ok=True)
    else:
        # Preserve support for earlier pinned Hub-distributed native releases.
        from huggingface_hub import hf_hub_download
        artifact = Path(hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"],
                                      cache_dir=bulk_home()/"huggingface"))
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != spec["sha256"]:
        raise RuntimeError("Native archive checksum mismatch")
    parent = bulk_home()/"native"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent/spec["sha256"]
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(artifact) as archive:
            for name in archive.namelist():
                if Path(name).name != name or name in (".", "..") or ":" in name or "\\" in name:
                    raise RuntimeError("Unexpected native archive member")
            archive.extractall(staging)
        expected = "llama.dll" if system == "Windows" else "libllama.so"
        if not (staging/expected).is_file():
            raise RuntimeError("Native archive lacks its main library")
        if destination.exists():
            from ._downloads import digest
            if any(not (destination/p.name).is_file() or digest(p) != digest(destination/p.name)
                   for p in staging.iterdir()):
                raise RuntimeError(f"Native installation is corrupted: {destination}. Remove it and retry.")
        else:
            # Copy into an isolated staging directory, then activate only the complete installation.
            staged = parent/(spec["sha256"] + "-" + uuid.uuid4().hex)
            shutil.copytree(staging, staged)
            try:
                staged.rename(destination)
            except OSError:
                shutil.rmtree(staged)
                raise
    if with_cuda and system == "Windows" and spec.get("cuda") == "13.0":
        from ._downloads import install_windows_cuda
        install_windows_cuda()
    fd, receipt = tempfile.mkstemp(dir=parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as out:
            json.dump({"directory": destination.name, "sha256": spec["sha256"]}, out)
        Path(receipt).replace(parent/"active.json")
    finally:
        Path(receipt).unlink(missing_ok=True)
    name="llama.dll" if system=="Windows" else "libllama.so"
    print(f"Native runtime installed: {destination / name}")
    print("Next: python -m latentport doctor")


def doctor(backend="gguf"):
    from .diagnostics import diagnose
    result = diagnose(backend)
    print(json.dumps(result, indent=2))
    return 0 if result["ready_for_model_load"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="LatentPort installation, diagnostics and demo")
    commands = parser.add_subparsers(dest="command", required=True)
    native = commands.add_parser("install-native", help="Install verified prebuilt binaries")
    native.add_argument("--no-cuda", action="store_true", help="Use existing Windows CUDA runtime libraries")
    check = commands.add_parser("doctor", help="Report readiness without downloading models")
    check.add_argument("--backend", choices=("gguf", "bf16", "reference"), default="gguf")
    fetch = commands.add_parser("download", help="Download the pinned transfer profile")
    fetch.add_argument("--model-manifest", type=Path)
    demo = commands.add_parser("demo", help="Launch the packaged browser demo")
    from ._demo.server import add_arguments
    add_arguments(demo)
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        print(f"LatentPort: {exc}", file=sys.stderr)
        return 1


def dispatch(args):
    if args.command == "install-native":
        install_native(with_cuda=not args.no_cuda)
        return 0
    if args.command == "download":
        from ._dependencies import require_backend
        require_backend("gguf")
        from .hub import gguf_model_file, gguf_bridge_file, bundle_root, model_snapshot
        root = bundle_root()
        manifest = json.loads((root / "latentport.json").read_text())
        for side in ("source", "target"):
            print(f"{side}: {gguf_model_file(side, manifest_path=args.model_manifest)}", flush=True)
            model_snapshot(manifest["models"][side], metadata_only=True)
        print(f"bridge: {gguf_bridge_file()}")
        return 0
    if args.command == "demo":
        from ._demo.server import run
        return run(args)
    return doctor(args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
