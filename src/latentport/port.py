"""A selected source, target and bridge with explicit execution policies."""
from __future__ import annotations

import json
from pathlib import Path

from .bridge import Bridge
from .errors import CapabilityError
from .hub import DEFAULT_REPO, DEFAULT_REVISION, bundle_root, model_snapshot
from .identity import ModelIdentity
from .model import Model


class Port:
    def __init__(self, source: Model, target: Model, bridge: Bridge):
        self.source, self.target, self.bridge = source, target, bridge

    def close(self):
        """Release resident model/bridge storage; independent States stay owned."""
        runtime = self.source.backend
        if hasattr(runtime, "close"):
            runtime.close()
        elif runtime.name == "reference":
            runtime._models.clear()
        self.bridge._translator = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def composite(self):
        from .composite import Composite
        return Composite(self.source, self.target, self.bridge)

    def warm(self, *, mtp: bool = False) -> dict:
        """Preload both models and bridge, and report measured residency."""
        runtime = self.source.backend
        if mtp and not runtime.supports_mtp:
            raise CapabilityError("MTP is unavailable in this runtime")
        if runtime.name == "reference":
            import torch
            runtime._model(self.source.identity)
            runtime._model(self.target.identity)
            self.bridge.prepare()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            return {"device": runtime.device, "gpu_total_bytes": int(total),
                    "gpu_free_bytes": int(free),
                    "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
                    "source_resident": True, "target_resident": True,
                    "bridge_resident": True, "mtp_resident": False}
        if runtime.name == "gguf":
            self.bridge.prepare()
            report = runtime._native().residency()
            if not all(report.get(key) for key in ("source_resident", "target_resident", "bridge_resident")):
                raise CapabilityError("Native runtime has not kept both models and bridge resident")
            if mtp and not report.get("mtp_resident"):
                raise CapabilityError("MTP state is not resident")
            return report
        raise CapabilityError("Unknown backend residency policy")


def load(repo: str | Path = DEFAULT_REPO, *, revision: str = DEFAULT_REVISION,
         backend: str = "reference", source_path: str | Path | None = None,
         target_path: str | Path | None = None, device: str = "cuda:0",
         bridge_gguf_path: str | Path | None = None, native_options=None) -> Port:
    root = bundle_root(repo, revision=revision)
    manifest = json.loads((root / "latentport.json").read_text(encoding="utf-8"))
    if backend == "reference":
        from ._reference.native import ReferenceBackend
        specs = {side: {**manifest["models"][side],
                        "template_kwargs": manifest["runtime"]["template_kwargs"][side]}
                 for side in ("source", "target")}
        runtime = ReferenceBackend({"source": source_path, "target": target_path}, specs,
                                   device=device,
                                   resolver=lambda side, metadata_only=False:
                                   model_snapshot(specs[side], metadata_only=metadata_only))
        weights = "nf4-double-bf16"
    elif backend == "gguf":
        from .gguf import GGUFBackend
        from .hub import gguf_model_file
        if source_path is None:
            source_path = gguf_model_file("source")
        if target_path is None:
            target_path = gguf_model_file("target")
        runtime = GGUFBackend(source_path, target_path, root, manifest,
                              bridge_gguf_path=bridge_gguf_path, options=native_options)
        runtime.require_extension()
        weights = None
    else:
        raise ValueError("backend must be 'reference' or 'gguf'")
    models = {}
    for side in ("source", "target"):
        spec = manifest["models"][side]
        identity = ModelIdentity(repo_id=spec["repo_id"], revision=spec["revision"],
                                 backend=runtime.name,
                                 weights=weights if weights is not None else runtime.weights(side),
                                 tokenizer_revision=spec["revision"],
                                 template_sha256=spec["metadata_files"]["chat_template.jinja"]["sha256"],
                                 state_abi=runtime.state_abi)
        models[side] = Model(identity, runtime, role=side)
    bridge = Bridge(bundle_root=root, source=models["source"], target=models["target"])
    return Port(models["source"], models["target"], bridge)
