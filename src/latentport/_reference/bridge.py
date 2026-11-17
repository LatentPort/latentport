"""Execute a manifest-selected mixed hybrid-state bridge."""
from __future__ import annotations

import json

from pathlib import Path

from ..errors import CapabilityError, CompatibilityError
from .native import validate_components


class ReferenceTranslator:
    def __init__(self, root: Path):
        self.root = root
        self.config = json.loads((root / "bridge_config.json").read_text())
        if self.config["architecture"] != "qwen-hybrid-mixed-v1":
            raise CompatibilityError("Unsupported translator architecture")
        self.geo = json.loads((root / "configs/geometry.json").read_text())
        self.arch = json.loads((root / "configs/architecture_v2.json").read_text())
        self.calibration = json.loads((root / "configs/calibration.json").read_text())
        self._loaded = None

    def _weights(self):
        from safetensors.torch import load_file
        return {group: load_file(str(self.root / path), device="cpu")
                for group, path in self.config["weights"].items()}

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        import torch
        from . import architecture as a
        if not torch.cuda.is_available():
            raise CapabilityError("The selected translator requires CUDA")
        groups = self._weights()
        means = {key: value.cuda() for key, value in groups["population_means"].items()}
        compact = groups["compact"]
        updates = {key[3:]: value for key, value in compact.items() if key.startswith("KV.")}
        cr = {key[5:]: value.cuda() for key, value in compact.items() if key.startswith("CR.p.")}
        target = self.geo["target"]
        stems = ("C_gain", "C_bias", "R_left_down", "R_left_up", "R_right_down", "R_right_up")
        expected = {f"{stem}_{i}" for i, kind in enumerate(target["types"])
                    if kind == "linear_attention" for stem in stems}
        if set(cr) != expected or len(compact) != len(updates) + len(cr):
            raise CompatibilityError("Persistent correction tensor inventory mismatch")
        args = (890520, means, 909521, self.geo, self.arch, self.calibration)
        parent = a.HeadBasisModel(*args)
        parent.load_state_dict(groups["parent"], strict=True)
        kv = a.HeadBasisModel(*args)
        # The frozen parts of both branches share resident storage. Override
        # parameters are replaced below; never copy into shared parent weights.
        kv.load_state_dict(parent.state_dict(), strict=True, assign=True)
        if set(updates) != set(self.config["kv_override_names"]):
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
        h = torch.nn.ModuleDict({"banks": torch.nn.ModuleList(
            [a.RMSLinearHResidual(width=target["hidden"]) for _ in self.config["thinking_modes"]])}).cuda()
        h.load_state_dict(groups["h"], strict=True)
        for model in (parent, kv, h):
            model.eval().requires_grad_(False)
        a.install(parent, consume_source=False)
        # This bundle uses the parent's H/C/R and the trained branch's K/V.
        # Parent K/V projections are unused and can occupy a full extra cache.
        parent.base._emit_kv = False
        self._loaded = (a, parent, kv, h["banks"], cr)
        return self._loaded

    def warm(self):
        self._load()

    def translate(self, components, source_clock: int, metadata):
        import torch
        validate_components(components, source_clock, "source", geometry=self.geo["source"])
        a, parent, kv, h, cr = self._load()
        mode = self.config['thinking_modes'].index(metadata['thinking_mode'])
        source = a.State([
            tuple(components[f"{field}.{i}"] for field in (("K", "V") if kind == "full_attention" else ("C", "R")))
            for i, kind in enumerate(self.geo["source"]["types"])
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
            for i in parent.base.tlin:
                c, r = layers[i]
                cf, rf = c.float(), r.float()
                rms = cf.square().mean((-2, -1), keepdim=True).add(1e-12).sqrt()
                conv = cf * (1 + cr[f"C_gain_{i}"][mode]) + rms * cr[f"C_bias_{i}"][mode]
                left = cr[f"R_left_up_{i}"][mode] @ (cr[f"R_left_down_{i}"][mode] @ rf)
                right = (rf @ cr[f"R_right_down_{i}"][mode]) @ cr[f"R_right_up_{i}"][mode]
                layers[i] = (conv.to(c.dtype), (rf + left + right).to(r.dtype))
            target = a.State(layers, h[mode](inherited.terminal), slots.target_length)
        values = {"H": target.terminal}
        for i, kind in enumerate(self.geo["target"]["types"]):
            x, y = ("K", "V") if kind == "full_attention" else ("C", "R")
            values[f"{x}.{i}"], values[f"{y}.{i}"] = target.layers[i]
        validate_components(values, target.length, "target", finite=True, geometry=self.geo["target"])
        return values
