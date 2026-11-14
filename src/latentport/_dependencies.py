"""Backend requirements, checked before downloading model weights."""
from importlib.util import find_spec

from .errors import CapabilityError


def missing_dependencies(backend):
    modules = ["torch", "numpy", "transformers", "safetensors", "huggingface_hub"]
    if backend in ("bf16", "reference"):
        modules.append("accelerate")
    if backend == "bf16":
        modules.extend(["fla", "causal_conv1d"])
    if backend == "reference":
        modules.append("bitsandbytes")
    return [name for name in modules if find_spec(name) is None]


def require_backend(backend):
    missing = missing_dependencies(backend)
    if missing:
        raise CapabilityError(
            f"Missing {backend} dependencies: {', '.join(missing)}. "
            f'Install a compatible CUDA PyTorch first, then pip install "latentport[{backend}]". '
            "See docs/installation.md, or use the prebuilt Docker image for the demo."
        )
