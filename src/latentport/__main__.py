"""Small installation and diagnostics CLI: python -m latentport --help."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import zipfile

from .hub import bulk_home


def install_native():
    """Install the pinned, checksum-verified Windows native distribution."""
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("The preview native download supports Windows x64 only; see docs/installation.md")
    from huggingface_hub import hf_hub_download
    spec = json.loads((Path(__file__).parent / "configs/native-release.json").read_text())
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
    print(f"Native runtime installed: {destination / 'llama.dll'}")
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
        result["native_abi"] = Library(None).lib.lp_abi()
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
