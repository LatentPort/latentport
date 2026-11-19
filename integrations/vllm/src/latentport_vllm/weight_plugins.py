"""Register the opt-in quantization loaders in every vLLM process."""

from importlib.util import find_spec
from os import environ

from .settings import QUANTIZED_LOADERS


def require_plugin(load_format):
    allowlist = environ.get("VLLM_PLUGINS")
    if allowlist is not None and "latentport_weight_formats" not in allowlist.split(
        ","
    ):
        raise ValueError("Add latentport_weight_formats to VLLM_PLUGINS")
    if load_format == QUANTIZED_LOADERS["nf4"]:
        if find_spec("vllm_bnb_plugin") is None:
            raise ValueError("Install latentport-vllm[nf4] to serve NF4 weights")
    elif (
        find_spec("vllm_gguf_plugin") is None
        or find_spec("vllm_gguf_plugin.gguf_files") is None
    ):
        raise ValueError(
            "Install the GGUF plugin revision in the LPCompact setup guide"
        )
    elif allowlist is not None and "gguf" not in allowlist.split(","):
        raise ValueError("Add gguf to VLLM_PLUGINS")


def register_loaders():
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.model_loader import register_model_loader

    if find_spec("vllm_bnb_plugin") is not None:
        from .nf4 import NF4Config, NF4ModelLoader

        name = QUANTIZED_LOADERS["nf4"]
        register_quantization_config(name)(NF4Config)
        register_model_loader(name)(NF4ModelLoader)

    if find_spec("vllm_gguf_plugin") and find_spec("vllm_gguf_plugin.gguf_files"):
        from .gguf_loader import PinnedGGUFModelLoader

        register_model_loader(QUANTIZED_LOADERS["gguf"])(PinnedGGUFModelLoader)
