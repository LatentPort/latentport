"""NF4 loading for the pinned safetensors checkpoints."""

from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm_bnb_plugin.bitsandbytes_loader import BitsAndBytesModelLoader
from vllm_bnb_plugin.quantization.config import BitsAndBytesConfig

from latentport._models import FLASH_NEXT

from .settings import QUANTIZED_LOADERS


class NF4Config(BitsAndBytesConfig):
    def __init__(self):
        super().__init__(
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # Keep the QSA index projection separate from the packed Q/K/V
            # projection. PLE keeps its native disk-offloaded BF16 storage.
            llm_int8_skip_modules=["index_qk_proj", "ple"],
        )

    @classmethod
    def get_name(cls):
        return QUANTIZED_LOADERS["nf4"]

    @classmethod
    def from_config(cls, config):
        del config
        return cls()


def expert_weights(name, weight):
    suffix = name.removesuffix(".weight").rsplit(".", 1)[-1]
    if (
        weight.ndim != 3
        or ".mlp.experts." not in name
        or suffix
        not in {
            "gate_up_proj",
            "down_proj",
        }
    ):
        yield name, weight
        return

    prefix = name.rsplit(".mlp.experts.", 1)[0] + ".mlp.experts."
    for expert, matrix in enumerate(weight.unbind()):
        if suffix == "down_proj":
            yield f"{prefix}{expert}.down_proj.weight", matrix
        else:
            gate, up = matrix.chunk(2, dim=0)
            yield f"{prefix}{expert}.gate_proj.weight", gate
            yield f"{prefix}{expert}.up_proj.weight", up


class NF4ModelLoader(BitsAndBytesModelLoader):
    def _hf_weight_iter(self, hf_weights_files, use_safetensors):
        if not use_safetensors:
            raise ValueError("LPCompact NF4 requires the pinned safetensors checkpoint")
        weights = safetensors_weights_iterator(
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
            safetensors_load_strategy="lazy",
            readonly_ple=(
                self.load_config.model_loader_extra_config["base_model"] == FLASH_NEXT
            ),
        )
        for original_name, weight in weights:
            if original_name.startswith(("mtp.", "model.mtp.")):
                continue
            for name, tensor in expert_weights(original_name, weight):
                mapped_name = self.weight_mapper(name)
                if mapped_name is not None:
                    yield name, mapped_name, tensor
