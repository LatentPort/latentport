"""A selected source, target and bridge with explicit execution policies."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .bridge import Bridge
from .errors import CapabilityError
from .hub import DEFAULT_REPO, bundle_root, model_snapshot
from .identity import ModelIdentity
from .model import Model
from ._tokens import token_ids


class Port:
    def __init__(self, source: Model, target: Model, bridge: Bridge):
        if source.backend is not target.backend:
            raise ValueError("A Port requires one shared backend")
        if source._owner is not None or target._owner is not None:
            raise ValueError("Model views already belong to a Port")
        self.source, self.target, self.bridge = source, target, bridge
        self.closed = False
        source._owner = target._owner = self

    def handoff(self, prefix, *, target_suffix_ids):
        """Translate a saved prefix and consume the caller's exact target-side question tokens."""
        from .errors import StateError
        if prefix.phase != "context_ready":
            raise StateError("Query handoff requires a snapshot taken before the final request")
        ids = token_ids(target_suffix_ids)
        translated = self.bridge.translate(prefix)
        # Arbitrary exact tokens do not identify a canonical user message.
        # Keep the neural stream, but do not publish a misleading partial chat.
        return replace(self.target.append(translated, ids, phase="answer_ready", messages=()),
                       messages_complete=False, generated_token_ids=())

    def close(self):
        """Release resident model/bridge storage; independent States stay owned."""
        if self.closed:
            return
        self.closed = True
        runtime = self.source.backend
        if hasattr(runtime, "close"):
            runtime.close()
        elif runtime.name in ("nf4", "bf16", "reference"):
            runtime._models.clear()
        self.bridge._translator = None

    def __enter__(self):
        self.source._check_open()
        self.target._check_open()
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def composite(self):
        from .composite import Composite
        return Composite(self.source, self.target, self.bridge)

    def warm(self, *, mtp: bool = False) -> dict:
        """Preload both models and bridge, and report measured residency."""
        self.source._check_open()
        self.target._check_open()
        runtime = self.source.backend
        if mtp and not runtime.supports_mtp:
            raise CapabilityError("MTP is unavailable in this runtime")
        if runtime.name in ("nf4", "bf16", "reference"):
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


def load(repo: str | Path = DEFAULT_REPO, *, revision: str | None = None,
         backend: str = "nf4", thinking: bool = False, source_path: str | Path | None = None,
         target_path: str | Path | None = None, device: str = "cuda:0",
         bridge_gguf_path: str | Path | None = None, native_options=None,
         model_manifest: str | Path | None = None) -> Port:
    if backend == "reference":
        backend = "nf4"
    if backend not in ("nf4", "bf16", "gguf") or type(thinking) is not bool:
        raise ValueError("Use backend='nf4', 'bf16' or 'gguf' and a boolean thinking mode")
    from ._dependencies import require_backend
    from .hub import verify_model_weights
    require_backend(backend, transfer=True)
    if backend == "gguf":
        from ._native.engine import library_path
        library_path(native_options.library if native_options else None)
    root = bundle_root(repo, revision=revision, full=True, backend=backend)
    manifest = json.loads((root / "latentport.json").read_text(encoding="utf-8"))
    if backend not in manifest["runtime"]["backends"]:
        raise CapabilityError(f"This bridge does not declare {backend} execution")
    mode = "thinking_on" if thinking else "thinking_off"
    if mode not in manifest["runtime"].get("thinking_modes", ("thinking_off", "thinking_on")):
        raise CapabilityError(f"This bridge does not support {mode}")
    geo = json.loads((root / "configs/geometry.json").read_text())
    for side in ("source", "target"):
        manifest["runtime"]["template_kwargs"][side]["enable_thinking"] = thinking
    specs = {side: {**manifest["models"][side], "geometry": geo[side],
                    "template_kwargs": manifest["runtime"]["template_kwargs"][side]}
             for side in ("source", "target")}
    if backend == "gguf":
        from .gguf import GGUFBackend
        from ._native.engine import NativeOptions
        from .hub import gguf_model_file
        locations = [gguf_model_file(side, manifest_path=model_manifest, root=root, local_path=path)
                     for side, path in (("source", source_path), ("target", target_path))]
        runtime = GGUFBackend(*locations, root, manifest, bridge_gguf_path=bridge_gguf_path,
                              options=native_options or NativeOptions(mtp_drafts=0))
        runtime.require_extension()
    else:
        from ._reference.native import ReferenceBackend
        locations = {}
        for side, path in (("source", source_path), ("target", target_path)):
            path = Path(path) if path else model_snapshot(specs[side], metadata_only=False)
            verify_model_weights(path, specs[side])
            locations[side] = path
        runtime = ReferenceBackend(locations, specs, device=device, precision=backend)
    models = {}
    for side in ("source", "target"):
        spec = manifest["models"][side]
        identity = ModelIdentity(repo_id=spec["repo_id"], revision=spec["revision"],
                                 backend=runtime.name,
                                 weights=(runtime.weights(side) if backend == "gguf" else backend + ":" + spec["revision"]),
                                 tokenizer_revision=spec["revision"],
                                 template_sha256=spec["metadata_files"]["chat_template.jinja"]["sha256"],
                                 state_abi=runtime.state_abi,
                                 template_options_sha256=hashlib.sha256(json.dumps(
                                     specs[side]["template_kwargs"], sort_keys=True).encode()).hexdigest())
        models[side] = Model(identity, runtime, role=side)
    bridge = Bridge(bundle_root=root, source=models["source"], target=models["target"])
    bridge.thinking_mode = "thinking_on" if thinking else "thinking_off"
    return Port(models["source"], models["target"], bridge)
