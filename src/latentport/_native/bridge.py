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
        config = json.loads((self.root / "bridge_config.json").read_text(encoding="utf-8"))
        groups = {"parent":{}, "kv":{}, "h":{}, "population_means":{}}
        for name, array in reader.tensors.items():
            prefix, key = name.split(".", 1)
            if prefix not in groups:
                raise CompatibilityError(f"Unknown bridge tensor namespace: {prefix}")
            groups[prefix][key] = torch.from_numpy(array)
        if len(reader.tensors) != 262:
            raise CompatibilityError("Selected bridge tensor count changed")
        means = {k:v.to("cuda:0") for k,v in groups["population_means"].items()}
        return config, means, groups["parent"], groups["kv"], groups["h"]
