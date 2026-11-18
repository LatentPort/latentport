"""Recover token IDs from cached input-embedding channels and the active weights."""
from threading import RLock

from .errors import CapabilityError, CompatibilityError, StateError


CHANNELS = 64


def _row_keys(rows):
    import numpy as np

    rows = np.ascontiguousarray(rows, dtype="<f2")
    return rows.view(np.dtype((np.void, CHANNELS * rows.itemsize))).reshape(-1)


class InputEmbeddingCache:
    def __init__(self, rows, vocabulary_size):
        self._rows = rows
        self._vocabulary_size = vocabulary_size
        self._index = None
        self._lock = RLock()

    def capture(self, ids):
        import numpy as np
        import torch

        return torch.from_numpy(self._rows(np.asarray(ids, dtype=np.int64)))

    def recover(self, activations, clock):
        import numpy as np
        import torch

        if activations.dtype != torch.float16 or activations.shape != (clock, CHANNELS):
            raise StateError("The right state's input-embedding cache is incomplete or has an invalid shape")
        with self._lock:
            if self._index is None:
                size = self._vocabulary_size()
                rows = np.empty((size, CHANNELS), dtype=np.float16)
                for start in range(0, size, 4096):
                    stop = min(start + 4096, size)
                    rows[start:stop] = self._rows(np.arange(start, stop))
                keys = _row_keys(rows)
                order = np.argsort(keys, kind="stable")
                self._index = (keys[order], order)
        keys, order = self._index
        wanted = _row_keys(activations.detach().cpu().numpy())
        low = np.searchsorted(keys, wanted, side="left")
        high = np.searchsorted(keys, wanted, side="right")
        invalid = np.flatnonzero(high - low != 1)
        if len(invalid):
            raise StateError(f"Could not uniquely identify right-state token {int(invalid[0])} from its cached embedding")
        return tuple(order[low].tolist())


class _GGUFEmbeddingRows:
    def __init__(self, files):
        self._files = files
        self._tensor = None
        self._lock = RLock()

    def __call__(self, ids):
        import numpy as np
        try:
            from gguf import GGML_QUANT_SIZES, GGUFReader, dequantize
        except ImportError as exc:
            raise CapabilityError("Install latentport[gguf] to cache GGUF input embeddings") from exc

        with self._lock:
            if self._tensor is None:
                for path in self._files:
                    self._tensor = next((tensor for tensor in GGUFReader(str(path)).tensors
                                         if tensor.name == "token_embd.weight"), None)
                    if self._tensor is not None:
                        break
                if self._tensor is None or self._tensor.data.ndim != 2 or self._tensor.shape[0] < CHANNELS:
                    raise CompatibilityError("The GGUF checkpoint has no supported input embedding tensor")
            tensor = self._tensor
        block, block_bytes = GGML_QUANT_SIZES[tensor.tensor_type]
        if tensor.data.dtype != np.uint8:
            block_bytes = block
        band_width = 8
        bands = np.linspace(0, int(tensor.shape[0]) - band_width, CHANNELS // band_width, dtype=int)
        rows = []
        try:
            # Distant bands distinguish rows with identical stretches after quantization.
            for start in bands:
                first_block, offset = divmod(start, block)
                end_block = (start + band_width + block - 1) // block
                data = tensor.data[ids, first_block * block_bytes:end_block * block_bytes]
                rows.append(dequantize(data, tensor.tensor_type)[:, offset:offset + band_width])
        except (NotImplementedError, ValueError) as exc:
            raise CompatibilityError(f"Cannot read input embeddings in GGUF type {tensor.tensor_type.name}") from exc
        return np.concatenate(rows, axis=1, dtype=np.float16)


def for_model(backend, identity, model_path):
    if identity.backend == "gguf":
        files = getattr(backend, "files", (model_path,))
        lookup = _GGUFEmbeddingRows(files)
    else:
        def lookup(ids):
            import torch

            with torch.inference_mode():
                decoder = backend._model(identity)
                text = getattr(decoder, "text", decoder)
                weight = text.get_input_embeddings().weight[:, :CHANNELS]
                rows = weight.index_select(0, torch.as_tensor(ids, device=weight.device))
                return rows.to(dtype=torch.float16, device="cpu").contiguous().numpy()

    return InputEmbeddingCache(lookup, lambda: backend.codec(identity).vocabulary_size)
