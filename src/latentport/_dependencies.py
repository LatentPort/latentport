"""Backend requirements, checked before downloading model weights."""
from importlib.util import find_spec

from .errors import CapabilityError


def missing_dependencies(backend, *, transfer=False, transformer_only=False):
    modules = ["torch", "numpy", "transformers", "safetensors", "huggingface_hub"]
    if backend in ("bf16", "nf4", "reference"):
        modules.append("accelerate")
    if not transfer and not transformer_only and backend in ("bf16", "nf4"):
        modules.extend(["fla", "causal_conv1d"])
    if backend in ("nf4", "reference"):
        modules.append("bitsandbytes")
    return [name for name in modules if find_spec(name) is None]


def require_backend(backend, *, transfer=False, transformer_only=False):
    if transformer_only:
        missing = missing_dependencies(backend, transformer_only=True)
    else:
        missing = missing_dependencies(backend, transfer=True) if transfer else missing_dependencies(backend)
    if missing:
        raise CapabilityError(
            f"Missing {backend} dependencies: {', '.join(missing)}. "
            f'Install a compatible CUDA PyTorch first, then pip install "latentport[{backend}]". '
            "See docs/installation.md, or use the prebuilt Docker image for the demo."
        )
