"""Run pinned GGUF weights through vLLM's paged-cache models."""

import torch
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.loader import (
    GGUFLoadPlan,
    GGUFModelLoader,
    _get_unquantized_modules,
)
from vllm_gguf_plugin.quantization import recursive_replace_vocab_modules
from vllm_gguf_plugin.weight_utils import get_gguf_tensor_names
from vllm_gguf_plugin.weights_adapter import Qwen35GGUFAdapter, get_weights_adapter

from latentport._models import FLASH_NEXT

from .gguf_weights import gguf_files


class PinnedGGUFModelLoader(GGUFModelLoader):
    def __init__(self, load_config):
        BaseModelLoader.__init__(self, load_config)
        options = load_config.model_loader_extra_config
        self._base_model = options["base_model"]
        self._local_path = options["gguf_path"]
        self._model_files = None

    def _prepare_model_files(self, model_config):
        if self._model_files is None:
            self._model_files = GGUFModelFiles(
                backbone=gguf_files(
                    self._base_model, self._local_path, self.load_config.download_dir
                ),
                mm_proj=None,
            )
        return self._model_files

    def _prepare_adapter(self, model_config):
        files = self._prepare_model_files(model_config)
        config = model_config.hf_config
        names = get_gguf_tensor_names(files.backbone)
        if self._base_model == FLASH_NEXT:
            from .flash_gguf import FlashNextGGUFAdapter

            adapter = FlashNextGGUFAdapter()
        else:
            adapter = get_weights_adapter(config)

        # Retain the pinned config and tokenizer. No separate GGUF projector is
        # required because vLLM only instantiates the text tower.
        if (
            getattr(config, "vision_config", None) is not None
            and isinstance(adapter, Qwen35GGUFAdapter)
            and self._base_model != FLASH_NEXT
        ):
            name_map = {
                name: (
                    "model.language_model." + target.removeprefix("model.")
                    if target.startswith("model.")
                    else target
                )
                for name, target in adapter.build_name_map(files, model_config).items()
            }
        else:
            name_map = adapter.build_name_map(files, model_config)

        text_config = config.get_text_config()
        text_config.tie_word_embeddings = "output.weight" not in names
        return adapter, GGUFLoadPlan(
            files,
            name_map,
            _get_unquantized_modules(files, name_map)
            + tuple(adapter.extra_unquantized_modules),
            adapter.get_linear_layouts(files, model_config, name_map),
        )

    def load_model(self, vllm_config, model_config, prefix=""):
        if self._base_model != FLASH_NEXT:
            return super().load_model(vllm_config, model_config, prefix)

        from .flash_gguf import bind_ple_table

        adapter, plan = self._prepare_adapter(model_config)
        quantization = vllm_config.quant_config
        quantization.unquantized_modules.extend(plan.unquantized_modules)
        quantization.register_linear_layouts(plan.linear_layouts, prefix=prefix)

        device = torch.device(vllm_config.device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config, prefix=prefix
                )
                recursive_replace_vocab_modules(model, quantization, prefix=prefix)
            model.load_weights(self._iter_weights(adapter, plan, model_config))
            bind_ple_table(model, plan.files.backbone)
            process_weights_after_loading(model, model_config, device)
        return model.eval()
