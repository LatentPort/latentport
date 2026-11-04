"""Strict adapter contract for a LatentPort-enabled llama.cpp native extension.

Stock llama.cpp state serialization cannot install arbitrary translated Qwen
hybrid tensors. This adapter refuses to run if the required extension is absent
or if it does not report the complete-state and MTP capabilities explicitly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock
from typing import Mapping

from .errors import CapabilityError, CompatibilityError, StateError
from .hub import model_snapshot
from ._reference.native import validate_components


STATE_ABI = "llama.cpp.b19cbe9.qwen3.5.complete-hybrid.bf16.v2"
EXTENSION_ABI = 2


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GGUFBackend:
    name = "gguf"
    state_abi = STATE_ABI
    device = "cuda:0"
    translation_device = "cuda:0"
    copy_on_write = True

    def __init__(self, source_path, target_path, bundle_root, manifest, *, bridge_gguf_path=None, engine=None, options=None):
        self.paths = {"source": Path(source_path).resolve(), "target": Path(target_path).resolve()}
        self.bundle_root = Path(bundle_root)
        self.manifest = manifest
        self.bridge_gguf_path = Path(bridge_gguf_path).resolve() if bridge_gguf_path is not None else None
        self._engine = engine
        self._engine_lock = RLock()
        self.options = options
        self._hashes = {}
        self._codecs = {}
        for side, path in self.paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{side} GGUF file missing: {path}")
            # Hub snapshots resolve to extensionless, content-addressed blobs.
            # Validate the format rather than the resolved filename suffix.
            with path.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise CompatibilityError(f"{side} file is not a GGUF container: {path}")

    def _bridge_file(self):
        from .hub import gguf_bridge_file
        path = gguf_bridge_file(self.bridge_gguf_path)
        if not path.is_file() or path.stat().st_size != 2578356288:
            raise CompatibilityError("Selected GGUF bridge file missing or size changed")
        if _hash_file(path) != "53cbd8104120b88dc2dcbc1d54f98133bde16c7f95ee33dc7c3d73b1c2292486":
            raise CompatibilityError("Selected GGUF bridge hash changed")
        self.bridge_gguf_path = path
        return path

    def weights(self, side):
        if side not in self._hashes:
            self._hashes[side] = "gguf-sha256:" + _hash_file(self.paths[side])
        return self._hashes[side]

    def require_extension(self):
        if self._engine is None:
            from ._native.engine import library_path
            library_path(self.options.library if self.options else None)

    def _side(self, identity):
        for side, spec in self.manifest["models"].items():
            if (identity.repo_id, identity.revision, identity.weights) == (
                spec["repo_id"], spec["revision"], self.weights(side)):
                return side
        raise CompatibilityError("GGUF state is bound to a different exact base file")

    def _native(self):
        with self._engine_lock:
            return self._get_native()

    def _get_native(self):
        if self._engine is None:
            from ._native.engine import Engine
            self._engine = Engine(
                str(self.paths["source"]), str(self.paths["target"]),
                str(self._bridge_file()), self.bundle_root, self.options)
        capabilities = self._engine.capabilities()
        if (capabilities.get("abi") != EXTENSION_ABI or
            not capabilities.get("full_hybrid_state_install") or
            not capabilities.get("device_resident_state") or
            not capabilities.get("immutable_snapshot_sharing")):
            raise CapabilityError("Native extension lacks the required complete hybrid-state ABI")
        return self._engine

    @property
    def supports_mtp(self):
        return bool(self._native().capabilities().get("mtp_commit_and_rollback"))

    def codec(self, identity):
        side = self._side(identity)
        if side not in self._codecs:
            from transformers import AutoTokenizer
            from .codec import Codec
            spec = self.manifest["models"][side]
            path = model_snapshot(spec, metadata_only=True)
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self._codecs[side] = Codec(tokenizer, self.manifest["runtime"]["template_kwargs"][side])
            # The native GGUF tokenizer must produce these exact IDs before
            # bridge alignment can be trusted.
            for sample in ("LatentPort tokenizer probe", "<|im_start|>user\nprobe<|im_end|>"):
                if tuple(self._native().tokenize(side, sample)) != self._codecs[side].encode(sample):
                    raise CompatibilityError(f"{side} GGUF tokenizer differs from pinned HF tokenizer")
        return self._codecs[side]

    def capture(self, identity, token_ids):
        side = self._side(identity)
        values = self._native().capture(side, tuple(token_ids))
        validate_components(values, len(token_ids), side, native_precision=True)
        return values

    def advance(self, identity, components, clock, token_ids):
        side = self._side(identity)
        validate_components(components, clock, side, native_precision=True)
        values = self._native().advance(side, components, clock, tuple(token_ids))
        validate_components(values, clock + len(token_ids), side, native_precision=True)
        return values

    def first_logits(self, identity, components, clock):
        side = self._side(identity)
        validate_components(components, clock, side, native_precision=True)
        return self._native().first_logits(side, components["H"])

    def eos_ids(self, identity):
        return frozenset(int(x) for x in self._native().eos_ids(self._side(identity)))

    def install(self, identity, components, clock):
        side = self._side(identity)
        validate_components(components, clock, side, native_precision=True)
        self._native().check_install(side, components, clock)

    def begin_decode(self, identity, components, clock, *, mtp=False):
        side = self._side(identity)
        validate_components(components, clock, side, native_precision=True)
        if mtp and not self.supports_mtp:
            raise CapabilityError("Native extension cannot commit and roll back MTP draft state")
        handle = self._native().begin_decode(side, components, clock, mtp=mtp)
        return _NativeDecodeHandle(handle, side, clock, mtp)

    def translator(self):
        return _NativeTranslator(self)

    def generation_guard(self):
        # Serialize complete requests on the supported single-GPU runtime.
        # Do not swap full branch states between individual decoding tokens.
        return self._native()._lock

    def finish_request(self):
        # The generation frame has released its temporary input snapshot now.
        self._native()._trim_torch_cache()

    def close(self):
        if self._engine is not None:
            self._engine.close()


class _NativeTranslator:
    def __init__(self, backend):
        self.backend = backend

    def warm(self):
        engine = self.backend._native()
        with engine._lock:
            engine.bridge.warm()

    def translate(self, components, source_clock, metadata):
        validate_components(components, source_clock, "source", native_precision=True)
        values = self.backend._native().translate(components, metadata)
        validate_components(values, metadata["target_length"], "target")
        return values


class _NativeDecodeHandle:
    def __init__(self, handle, side, clock, mtp):
        self._handle, self.side, self.clock, self.mtp = handle, side, clock, mtp
        self.logits = handle.logits()

    def step(self, token_id):
        # Native MTP may speculate internally, but the public state advances
        # by exactly one accepted token. Rejected drafts must not persist.
        receipt = self._handle.commit(int(token_id))
        if receipt["committed_clock"] != self.clock + 1 or not receipt["rejected_drafts_rolled_back"]:
            raise StateError("Native decoder committed an invalid MTP state")
        self.clock += 1
        self.logits = self._handle.logits()

    def snapshot(self):
        values = self._handle.snapshot()
        validate_components(values, self.clock, self.side, native_precision=True)
        return values

    def greedy_batch(self, limit, eos):
        tokens = self._handle.greedy_batch(limit, eos)
        if not tokens or len(tokens) > limit or self._handle.clock != self.clock + len(tokens):
            raise StateError("Native MTP batch has an invalid committed clock")
        self.clock = self._handle.clock
        return tokens

    def stats(self):
        return self._handle.stats()

    def close(self):
        self._handle.close()

    def enable_mtp(self):
        self._handle.enable_mtp()
        self.mtp = True
