"""Load a model independently of any translator, using the existing Model API."""
import hashlib
from dataclasses import replace
from pathlib import Path

from .errors import CompatibilityError
from .identity import ModelIdentity
from .model import Model
from .context import ContextOptions, PINNED_REVISION, execution_weights
from ._models import DEFAULT_MODEL, FLASH_NEXT, MODEL_REVISIONS, model_geometry

MODEL = DEFAULT_MODEL
REVISION = PINNED_REVISION


def _weight_fingerprint(path):
    """Bind Transformers snapshots to actual tensor files, including local checkpoints."""
    files=sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise CompatibilityError("Transformers loading requires safetensors model weights")
    # Identity verification can read tens of GB on a cold disk. Use the same
    # progress controls as Hub downloads rather than leaving startup silent.
    from huggingface_hub.utils import tqdm
    combined=hashlib.sha256()
    for file in tqdm(files, desc="Verifying model weight shards", unit="shard"):
        digest=hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b""):
                digest.update(block)
        combined.update(file.name.encode()+b"\0"+digest.digest())
    return combined.hexdigest()


def load_model(repo: str = MODEL, *, revision: str | None = None, backend: str = "bf16",
               model_path: str | Path | None = None, native_options=None,
               position_policy: str | None = None, max_kv_positions: int | None = None) -> Model:
    """Load a standalone Qwen checkpoint using its pinned revision by default.

    ``max_kv_positions`` bounds physical attention memory. ``yarn4`` extends
    logical positions to 1,048,576 without increasing that physical capacity.
    ``State.compact`` retains the same position policy throughout a lineage.
    """
    if backend not in ("bf16", "nf4", "gguf"):
        raise ValueError("backend must be 'bf16', 'nf4' or 'gguf'")
    if native_options is not None and backend != "gguf":
        raise ValueError("native_options requires backend='gguf'")
    if backend == "gguf":
        from ._native.engine import NativeOptions
        if native_options is not None and not isinstance(native_options, NativeOptions):
            raise TypeError("native_options must be NativeOptions")
    context = ContextOptions(position_policy if position_policy is not None else
                             getattr(native_options, "position_policy", "default"),
                             max_kv_positions if max_kv_positions is not None else
                             getattr(native_options, "physical_limit", 262144))
    if repo not in MODEL_REVISIONS:
        raise CompatibilityError(f"Unsupported standalone model: {repo}")
    if repo == FLASH_NEXT and (backend != "bf16" or context.position_policy != "default"):
        raise CompatibilityError("Flash-Next supports BF16 with its default position policy")
    if revision is None:
        revision = MODEL_REVISIONS[repo]
    if (repo != MODEL or backend == "nf4" or context.position_policy != "default") and revision != MODEL_REVISIONS[repo]:
        raise CompatibilityError("NF4 and extended positions require the exact pinned model revision")
    from ._dependencies import require_backend
    require_backend(backend)
    from .hub import model_snapshot
    geometry = model_geometry(repo)
    spec = {"repo_id": repo, "revision": revision, "metadata_files": {}, "geometry": geometry,
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
    if backend in ("bf16", "nf4"):
        from ._reference.bf16 import BF16Backend
        runtime_type = BF16Backend
        if repo == FLASH_NEXT:
            from ._reference.flash_next import FlashNextBackend
            runtime_type = FlashNextBackend
        quantization = None
        if backend == "nf4":
            from ._reference.nf4 import NF4Backend, NF4_CONFIG
            runtime_type, quantization = NF4Backend, NF4_CONFIG
        if model_path is None:
            model_path=model_snapshot(spec,metadata_only=False)
        runtime = runtime_type({"target": model_path}, {"target": spec}, options=context,
                              resolver=lambda side, metadata_only=False: model_snapshot(spec, metadata_only=metadata_only))
        weights = execution_weights(backend+"-sha256:"+_weight_fingerprint(model_path), context, quantization=quantization)
    elif backend == "gguf":
        from .gguf import GGUFBackend
        from ._native.engine import Engine, NativeOptions, library_path
        if model_path is None:
            raise ValueError("GGUF execution requires model_path")
        options = (replace(native_options, position_policy=context.position_policy, physical_limit=context.physical_limit)
                   if native_options is not None else NativeOptions(
                       context_size=context.physical_limit if context.position_policy=="yarn4" else min(66560,context.physical_limit), mtp_drafts=0,
                       position_policy=context.position_policy, physical_limit=context.physical_limit))
        library_path(options.library)
        class SingleGGUFBackend(GGUFBackend):
            def _get_native(self):
                if self._engine is None:
                    self._engine = Engine(None, self.paths["target"], None, None, self.options)
                    self._engine.geometry["target"] = geometry
                return self._engine
        manifest = {"models": {"target": spec}, "runtime": {"template_kwargs": {"target": spec["template_kwargs"]}}}
        runtime = SingleGGUFBackend(model_path, model_path, metadata, manifest, options=options)
        runtime.geometry["target"] = geometry
        weights = runtime.weights("target")
    identity = ModelIdentity(repo, revision, runtime.name, weights, revision,
                             spec["metadata_files"]["chat_template.jinja"]["sha256"], runtime.state_abi)
    return Model(identity, runtime, role="target")
