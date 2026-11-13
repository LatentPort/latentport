"""Immutable logical snapshots with exact compatibility and optional disk storage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from .errors import CompatibilityError, StateError
from .identity import CompactionInfo, ModelIdentity, StateInfo


STATE_FORMAT = "latentport.state.v1"
PHASES = frozenset({"context_ready", "answer_ready", "in_answer", "turn_ready"})


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _tensor_description(tensor: Any) -> tuple[int, tuple[int, ...], str]:
    shape = tuple(int(x) for x in tensor.shape)
    if hasattr(tensor, "element_size"):
        size = int(tensor.numel()) * int(tensor.element_size())
    else:
        size = int(tensor.nbytes)
    return size, shape, str(tensor.dtype)


@dataclass(frozen=True, slots=True)
class State:
    identity: ModelIdentity
    clock: int
    phase: str
    origin: str
    target_prompt_tokens: int = 0
    # These values describe source-side prompt rendering. They can be absent for
    # translated states, whose historical target tokens were never processed.
    rendered_text: str | None = None
    rendered_token_ids: tuple[int, ...] | None = None
    messages: tuple[tuple[str, str], ...] = ()
    generated_token_ids: tuple[int, ...] = ()
    state_id: str = field(default_factory=lambda: uuid4().hex)
    parent_id: str | None = None
    shares_storage: bool = False
    _components: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    _snapshot: Path | None = field(default=None, repr=False, compare=False)
    _disk_index: Mapping[str, Mapping[str, Any]] | None = field(default=None, repr=False, compare=False)
    compaction: CompactionInfo | None = None

    def __post_init__(self):
        if self.clock <= 0 or self.phase not in PHASES or self.target_prompt_tokens < 0:
            raise StateError("Invalid state clock, phase or target prompt count")
        if (self._components is None) == (self._snapshot is None):
            raise StateError("A state needs exactly one physical representation")
        if self._components is not None:
            if not self._components or not all(isinstance(k, str) and k for k in self._components):
                raise StateError("State components must have stable names")
            object.__setattr__(self, "_components", MappingProxyType(dict(self._components)))
        if self.rendered_token_ids is not None and len(self.rendered_token_ids) > self.clock:
            raise StateError("Rendered source tokens exceed the state clock")
        if self.messages and any(len(row) != 2 for row in self.messages):
            raise StateError("Messages must be (role, content) pairs")

    @property
    def info(self) -> StateInfo:
        if self._components is not None:
            sizes = [_tensor_description(x)[0] for x in self._components.values()]
            count, used = len(sizes), sum(sizes)
            representation = "device" if any(getattr(x, "device", "cpu").type == "cuda"
                                              for x in self._components.values()
                                              if hasattr(getattr(x, "device", None), "type")) else "memory"
        else:
            count = len(self._disk_index or {})
            used = sum(int(x["bytes"]) for x in (self._disk_index or {}).values())
            representation = "disk"
        return StateInfo(self.identity, self.clock, self.phase, self.origin,
                         self.target_prompt_tokens, representation, count, used,
                         self.shares_storage, self.state_id, self.parent_id)

    def fork(self, count: int = 1) -> "State | tuple[State, ...]":
        """Logical forks share their immutable prefix until a branch is advanced."""
        if count < 1:
            raise ValueError("fork count must be positive")
        copies = tuple(replace(self, state_id=uuid4().hex, parent_id=self.state_id,
                               shares_storage=True)
                       for _ in range(count))
        return copies[0] if count == 1 else copies

    def compact(self, *, ratio: int = 8, force: bool = False) -> "State":
        """Return an LPCompact snapshot with the same logical clock and model.

        Recurrent, convolutional and endpoint state are preserved. Histories
        below 65,536 retained KV positions return this state unchanged unless
        force=True. The parent and its forks remain valid.
        """
        from ._compact import compact
        return compact(self, ratio=ratio, force=force)

    def require_model(self, identity: ModelIdentity) -> None:
        self.identity.require_same(identity)

    def _tensors(self, *, device: str | None = None) -> dict[str, Any]:
        """Materialize a private working copy before a backend may mutate it."""
        import torch
        if self._components is not None:
            return {name: torch.as_tensor(value).detach().to(device or getattr(value, "device", "cpu"), copy=True)
                    for name, value in self._components.items()}
        from safetensors.torch import load_file
        if self._snapshot is None:
            raise StateError("No snapshot payload")
        return load_file(str(self._snapshot / "tensors.safetensors"), device=device or "cpu")

    def _read_tensors(self, *, device: str | None = None) -> dict[str, Any]:
        """Private read-only view for translation; translators must not mutate it."""
        if self._components is None:
            return self._tensors(device=device)
        if device is None:
            return dict(self._components)
        return {name: value if str(value.device) == device or
                (device == "cuda" and value.device.type == "cuda") else value.to(device)
                for name, value in self._components.items()}

    def save(self, path: str | Path) -> Path:
        """Save a full state without requiring either model to stay loaded."""
        from safetensors.torch import save_file
        root = Path(path).expanduser().resolve()
        if root.exists():
            raise FileExistsError(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = root.with_name(root.name + ".partial-" + uuid4().hex)
        staging.mkdir()
        try:
            if self._snapshot is not None:
                shutil.copyfile(self._snapshot / "tensors.safetensors", staging / "tensors.safetensors")
            else:
                payload = {name: value.detach().to("cpu").contiguous() if hasattr(value, "detach")
                           else __import__("torch").as_tensor(value).contiguous()
                           for name, value in self._components.items()}
                save_file(payload, str(staging / "tensors.safetensors"))
            from safetensors import safe_open
            index = {}
            with safe_open(str(staging / "tensors.safetensors"), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    view = handle.get_slice(name)
                    shape = list(view.get_shape())
                    index[name] = {"shape": shape, "dtype": str(view.get_dtype()),
                                   "bytes": _dtype_bytes(str(view.get_dtype())) * _product(shape)}
            record = {"format": STATE_FORMAT, "identity": asdict(self.identity),
                      "clock": self.clock, "phase": self.phase, "origin": self.origin,
                      "target_prompt_tokens": self.target_prompt_tokens,
                      "rendered_text": self.rendered_text,
                      "rendered_token_ids": self.rendered_token_ids, "messages": self.messages,
                      "generated_token_ids": self.generated_token_ids, "state_id": self.state_id,
                      "parent_id": self.parent_id, "shares_storage": False, "tensor_index": index,
                      "compaction": asdict(self.compaction) if self.compaction else None,
                      "tensor_sha256": _digest(staging / "tensors.safetensors"),
                      "tensor_file_bytes": (staging / "tensors.safetensors").stat().st_size}
            (staging / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            staging.replace(root)
            return root
        except BaseException:
            if staging.resolve().is_relative_to(root.parent.resolve()):
                shutil.rmtree(staging, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> "State":
        """Inspect and fork a snapshot before loading model weights or using GPU."""
        root = Path(path).expanduser().resolve()
        record = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if record.get("format") != STATE_FORMAT:
            raise StateError("Unsupported snapshot format")
        tensor_path = root / "tensors.safetensors"
        if tensor_path.stat().st_size != record["tensor_file_bytes"] or _digest(tensor_path) != record["tensor_sha256"]:
            raise StateError("Snapshot tensor checksum mismatch")
        from safetensors import safe_open
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(record["tensor_index"]):
                raise StateError("Snapshot tensor names changed")
            for name in handle.keys():
                view = handle.get_slice(name)
                expected = record["tensor_index"][name]
                if list(view.get_shape()) != expected["shape"] or str(view.get_dtype()) != expected["dtype"]:
                    raise StateError("Snapshot tensor schema changed: " + name)
        return cls(identity=ModelIdentity(**record["identity"]), clock=int(record["clock"]),
                   phase=record["phase"], origin=record["origin"],
                   target_prompt_tokens=int(record["target_prompt_tokens"]),
                   rendered_text=record.get("rendered_text"),
                   rendered_token_ids=tuple(record["rendered_token_ids"]) if record.get("rendered_token_ids") is not None else None,
                   messages=tuple(tuple(m) for m in record.get("messages", ())),
                   generated_token_ids=tuple(record.get("generated_token_ids", ())),
                   state_id=record["state_id"], parent_id=record.get("parent_id"),
                   shares_storage=False,
                   compaction=CompactionInfo(**record["compaction"]) if record.get("compaction") else None,
                   _snapshot=root, _disk_index=record["tensor_index"])


def _product(values):
    result = 1
    for x in values:
        result *= x
    return result


def _dtype_bytes(name):
    sizes = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I8": 1, "U8": 1,
             "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}
    if name not in sizes:
        raise StateError("Unsupported snapshot tensor dtype " + name)
    return sizes[name]
