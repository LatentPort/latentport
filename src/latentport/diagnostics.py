"""Read-only, backend-specific diagnostics. No model downloads or allocations."""
import json
import platform
from pathlib import Path

from ._dependencies import missing_dependencies
from .hub import bulk_home


def driver_cuda_version():
    """Read driver API compatibility after Torch has established GPU availability."""
    import ctypes
    library = ctypes.CDLL("nvcuda.dll" if platform.system() == "Windows" else "libcuda.so.1")
    version = ctypes.c_int()
    query = library.cuDriverGetVersion
    query.argtypes = [ctypes.POINTER(ctypes.c_int)]
    query.restype = ctypes.c_int
    if query(ctypes.byref(version)) != 0:
        raise RuntimeError("Cannot query CUDA driver compatibility")
    return version.value


def native_spec():
    system = platform.system()
    if system not in ("Windows", "Linux") or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("Prebuilt native runtimes support Windows/Linux x64 only")
    specs = json.loads((Path(__file__).parent / "configs/native-release.json").read_text())
    return specs["linux"] if system == "Linux" else specs


def diagnose(backend="gguf", *, check_native=True):
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "bulk_home": str(bulk_home()), "backend": backend,
              "package_installed": True, "missing_dependencies": missing_dependencies(backend),
              "cuda_available": False, "ready_for_model_load": False, "errors": []}
    if "torch" not in result["missing_dependencies"]:
        try:
            import torch
            result.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                          cuda_available=torch.cuda.is_available())
            if result["cuda_available"]:
                result["gpu"] = torch.cuda.get_device_name(0)
                result["gpu_capability"] = list(torch.cuda.get_device_capability(0))
                result["gpu_free_bytes"], result["gpu_total_bytes"] = torch.cuda.mem_get_info(0)
        except Exception as exc:
            result["errors"].append(f"PyTorch: {exc}")
    if result["missing_dependencies"]:
        result["errors"].append(
            f'Missing dependencies: {", ".join(result["missing_dependencies"])}. '
            f'Install CUDA PyTorch, then pip install "latentport[{backend}]".')
    if not result["cuda_available"]:
        result["errors"].append("CUDA is unavailable. Check the NVIDIA driver and CUDA-enabled PyTorch build.")
    if backend == "gguf":
        try:
            spec = native_spec()
            if result['cuda_available']:
                driver = driver_cuda_version()
                result['driver_cuda_version'] = driver
                # Major-version floor only: qualified hardware may require a newer driver.
                if driver < int(spec['cuda'].split('.')[0]) * 1000:
                    result['errors'].append(f"Native CUDA {spec['cuda']} requires a newer NVIDIA driver.")
            capability = ".".join(map(str, result.get("gpu_capability", [])))
            if capability and capability != spec["compute_capability"]:
                result["errors"].append(
                    f"This release is qualified for SM{spec['compute_capability']}, not SM{capability}.")
            if check_native:
                from ._native.engine import Library, library_path
                result["native_library"] = str(library_path())
                binding = Library(None)
                result["native_abi"] = binding.lib.lp_abi()
                result["native_state_compact"] = binding.sparse_state
        except Exception as exc:
            result["errors"].append(f"Native runtime: {exc}")
    result["ready_for_model_load"] = not result["errors"]
    result["model_weights_checked"] = False
    return result
