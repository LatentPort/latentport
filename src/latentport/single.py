"""Load a model independently of any translator, using the existing Model API."""
import hashlib
from dataclasses import replace
from pathlib import Path

from .errors import CompatibilityError
from .identity import ModelIdentity
from .model import Model
from .context import ContextOptions, PINNED_REVISION, execution_weights

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"


def _weight_fingerprint(path):
    """Bind Transformers snapshots to actual tensor files, including local checkpoints."""
    files=sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise CompatibilityError("Transformers loading requires safetensors model weights")
    combined=hashlib.sha256()
    for file in files:
        digest=hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b""):
                digest.update(block)
        combined.update(file.name.encode()+b"\0"+digest.digest())
    return combined.hexdigest()


def load_model(repo: str = MODEL, *, revision: str = REVISION, backend: str = "bf16",
               model_path: str | Path | None = None, native_options=None,
               position_policy: str | None = None, max_kv_positions: int | None = None) -> Model:
    """Load standalone Qwen3.8-27B with static default or YaRN4 positions.

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
    if (backend == "nf4" or context.position_policy != "default") and revision != PINNED_REVISION:
        raise CompatibilityError("NF4 and extended positions require the exact pinned model revision")
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
    if backend in ("bf16", "nf4"):
        from ._reference.bf16 import BF16Backend
        runtime_type = BF16Backend
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
                return self._engine
        manifest = {"models": {"target": spec}, "runtime": {"template_kwargs": {"target": spec["template_kwargs"]}}}
        runtime = SingleGGUFBackend(model_path, model_path, metadata, manifest, options=options)
        weights = runtime.weights("target")
    identity = ModelIdentity(repo, revision, runtime.name, weights, revision,
                             spec["metadata_files"]["chat_template.jinja"]["sha256"], runtime.state_abi)
    return Model(identity, runtime, role="target")
