"""Load a model independently of any translator, using the existing Model API."""
import hashlib
from pathlib import Path

from .errors import CompatibilityError
from .identity import ModelIdentity
from .model import Model

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"


def _weight_fingerprint(path):
    """Bind BF16 snapshots to actual tensor files, including local checkpoints."""
    files=sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise CompatibilityError("BF16 loading requires safetensors model weights")
    combined=hashlib.sha256()
    for file in files:
        digest=hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b""):
                digest.update(block)
        combined.update(file.name.encode()+b"\0"+digest.digest())
    return combined.hexdigest()


def load_model(repo: str = MODEL, *, revision: str = REVISION, backend: str = "bf16",
               model_path: str | Path | None = None, native_options=None) -> Model:
    """Load standalone Qwen3.8-27B; translator loading remains lp.load()."""
    if backend not in ("bf16", "gguf"):
        raise ValueError("backend must be 'bf16' or 'gguf'")
    from ._dependencies import require_backend
    require_backend(backend)
    from .hub import model_snapshot
    if repo != MODEL:
        raise CompatibilityError("Standalone execution currently supports Qwen3.8-27B")
    spec = {"repo_id": repo, "revision": revision, "metadata_files": {},
            "template_kwargs": {"enable_thinking": False}}
    # Resolve pinned tokenizer metadata even when the weights are a local GGUF.
    metadata = model_snapshot(spec, metadata_only=True)
    resolved=metadata.name
    if len(resolved)==40 and all(c in "0123456789abcdef" for c in resolved):
        revision=spec["revision"]=resolved
    for name in ("config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        path = metadata / name
        if path.exists():
            spec["metadata_files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                            "bytes": path.stat().st_size}
    if backend == "bf16":
        from ._reference.bf16 import BF16Backend
        if model_path is None:
            model_path=model_snapshot(spec,metadata_only=False)
        runtime = BF16Backend({"target": model_path}, {"target": spec},
                              resolver=lambda side, metadata_only=False: model_snapshot(spec, metadata_only=metadata_only))
        weights = "bf16-sha256:"+_weight_fingerprint(model_path)
    elif backend == "gguf":
        from .gguf import GGUFBackend
        from ._native.engine import Engine, NativeOptions, library_path
        if model_path is None:
            raise ValueError("GGUF execution requires model_path")
        options = native_options or NativeOptions(context_size=66560, mtp_drafts=0)
        library_path(options.library)
        class SingleGGUFBackend(GGUFBackend):
            def _get_native(self):
                if self._engine is None:
                    self._engine = Engine(None, self.paths["target"], None, None, self.options)
                return self._engine
        manifest = {"models": {"target": spec}, "runtime": {"template_kwargs": {"target": spec["template_kwargs"]}}}
        runtime = SingleGGUFBackend(model_path, model_path, metadata, manifest, options=options)
        weights = runtime.weights("target")
    else:
        raise ValueError("backend must be 'bf16' or 'gguf'")
    identity = ModelIdentity(repo, revision, runtime.name, weights, revision,
                             spec["metadata_files"]["chat_template.jinja"]["sha256"], runtime.state_abi)
    return Model(identity, runtime, role="target")
