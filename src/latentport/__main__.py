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
from urllib.request import urlopen

from .hub import bulk_home


def install_native():
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
    destination = bulk_home()/"native/build/bin"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            if Path(name).name != name or name in (".", "..") or ":" in name:
                raise RuntimeError("Unexpected native archive member")
        archive.extractall(destination)
    name="llama.dll" if system=="Windows" else "libllama.so"
    print(f"Native runtime installed: {destination / name}")
    print("Next: python -m latentport doctor")


def doctor():
    import torch
    from ._native.engine import Library, library_path
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "bulk_home": str(bulk_home()), "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if result["cuda_available"]:
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        result["gpu_free_bytes"], result["gpu_total_bytes"] = torch.cuda.mem_get_info()
    try:
        result["native_library"] = str(library_path())
        binding=Library(None)
        result["native_abi"] = binding.lib.lp_abi()
        result["native_state_compact"] = binding.sparse_state
    except Exception as exc:
        result["native_error"] = str(exc)
    result["ready_for_model_load"] = result["cuda_available"] and result.get("native_abi") == 2
    print(json.dumps(result, indent=2))
    return 0 if result["ready_for_model_load"] else 1


def main():
    parser = argparse.ArgumentParser(description="LatentPort preview setup and diagnostics")
    parser.add_argument("command", choices=("install-native", "download", "doctor"))
    args = parser.parse_args()
    if args.command == "install-native":
        install_native()
        return 0
    if args.command == "download":
        from .hub import gguf_model_file, gguf_bridge_file, bundle_root, model_snapshot
        root = bundle_root()
        manifest = json.loads((root / "latentport.json").read_text())
        for side in ("source", "target"):
            print(f"{side}: {gguf_model_file(side)}", flush=True)
            model_snapshot(manifest["models"][side], metadata_only=True)
        print(f"bridge: {gguf_bridge_file()}")
        return 0
    return doctor()


if __name__ == "__main__":
    raise SystemExit(main())
