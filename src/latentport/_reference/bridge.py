"""Selected E119 final16 dual-branch translation, loaded from a pinned bundle."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from ..errors import CapabilityError, CompatibilityError
from .native import validate_components


class ReferenceTranslator:
    def __init__(self, root: Path):
        self.root = root
        self._loaded = None

    def _weights(self):
        from safetensors.torch import load_file
        root = self.root
        config = json.loads((root / "bridge_config.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "latentport.json").read_text(encoding="utf-8"))
        for name, receipt in manifest["artifacts"].items():
            path = root / name
            if not path.is_file() or path.stat().st_size != receipt["bytes"]:
                raise CompatibilityError("Bridge artifact missing or size changed: " + name)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != receipt["sha256"]:
                raise CompatibilityError("Bridge artifact hash changed: " + name)
        return (config,
                load_file(str(root / "constants/population_means.safetensors"), device="cuda"),
                load_file(str(root / "bridge/parent.safetensors"), device="cpu"),
                load_file(str(root / "bridge/kv.safetensors"), device="cpu"),
                load_file(str(root / "bridge/h.safetensors"), device="cpu"))

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        import torch
        from . import architecture as a
        if not torch.cuda.is_available():
            raise CapabilityError("The selected translator requires CUDA")
        config, means, original, updates, h_weights = self._weights()
        parent = a.HeadBasisModel(890520, means, 909521)
        parent.load_state_dict(original, strict=True)
        kv = a.HeadBasisModel(890520, means, 909521)
        # The frozen parts of both branches share resident storage. Override
        # parameters are replaced below; never copy into shared parent weights.
        kv.load_state_dict(parent.state_dict(), strict=True, assign=True)
        if set(updates) != set(config["kv_override_names"]):
            raise CompatibilityError("K/V override names differ from the selected checkpoint")
        parameters = dict(kv.named_parameters())
        with torch.no_grad():
            for name, value in updates.items():
                old = parameters[name]
                if value.shape != old.shape:
                    raise CompatibilityError("K/V parameter shape changed: " + name)
                module_name, _, leaf = name.rpartition(".")
                module = kv.get_submodule(module_name) if module_name else kv
                module.register_parameter(leaf, torch.nn.Parameter(
                    value.to(device=old.device, dtype=old.dtype), requires_grad=False))
        h = a.RMSLinearHResidual().cuda()
        h.load_state_dict(h_weights, strict=True)
        for model in (parent, kv, h):
            model.eval().requires_grad_(False)
        a.install(parent, consume_source=False)
        self._loaded = (a, parent, kv, h)
        return self._loaded

    def warm(self):
        self._load()

    def translate(self, components, source_clock: int, metadata):
        import torch
        validate_components(components, source_clock, "source")
        a, parent, kv, h = self._load()
        source = a.State([
            tuple(components[f"{field}.{i}"] for field in (("K", "V") if kind == "full_attention" else ("C", "R")))
            for i, kind in enumerate(a.GEO["source"]["types"])
        ], components["H"], source_clock)
        if source.terminal.device.type != "cuda":
            source = source.to("cuda")
        slots = a.slots_for(metadata, source.terminal.device).validate(source.length)
        with torch.inference_mode():
            inherited = parent(a.historical_source(source), slots, chunk=512)
            features = a.routed_features(kv, source, metadata)
            pieces = {}
            for start in range(0, slots.target_length, 512):
                row = a.Head.forward(kv.base, features[:, start:start + 512].contiguous(), start)
                for name, tensor in row.items():
                    pieces.setdefault(name, []).append(tensor)
            layers = list(inherited.layers)
            for i in kv.base.tkv:
                layers[i] = tuple(torch.cat(pieces[f"{field}.{i}"], dim=-2) for field in ("K", "V"))
            target = a.State(layers, h(inherited.terminal), slots.target_length)
        values = {"H": target.terminal}
        for i, kind in enumerate(a.GEO["target"]["types"]):
            x, y = ("K", "V") if kind == "full_attention" else ("C", "R")
            values[f"{x}.{i}"], values[f"{y}.{i}"] = target.layers[i]
        validate_components(values, target.length, "target", finite=True)
        return values
