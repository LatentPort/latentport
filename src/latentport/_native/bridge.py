"""Execute the custom bridge GGUF with the identical resident Torch graph."""
import json
from pathlib import Path
from .._reference.bridge import ReferenceTranslator
from ..errors import CompatibilityError
from .gguf_reader import BridgeGGUF


class GGUFTranslator(ReferenceTranslator):
    def __init__(self, path, root):
        super().__init__(Path(root))
        self.path = Path(path)

    def _weights(self):
        import torch
        reader = BridgeGGUF(self.path)
        groups = {key: {} for key in self.config["weights"]}
        for name, array in reader.tensors.items():
            prefix, key = name.split(".", 1)
            if prefix not in groups:
                raise CompatibilityError(f"Unknown bridge tensor namespace: {prefix}")
            groups[prefix][key] = torch.from_numpy(array)
        if len(reader.tensors) != self.config["tensor_count"]:
            raise CompatibilityError("GGUF tensor count differs from bridge configuration")
        return groups
