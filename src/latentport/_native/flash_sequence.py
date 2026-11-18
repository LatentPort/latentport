"""Isolated llama.cpp worker for the Flash-Next GGUF runtime."""
from dataclasses import replace
import pickle
import struct
import subprocess
import sys
from threading import RLock

from .flash_native import flash_library_path
from .sequence import FLASH_STATE_ABI, Sequence, SequenceBackend, choose_gpu_layers, model_files
from .._models import model_geometry
from ..errors import CapabilityError, CompatibilityError, StateError
from ..state import State


_HEADER = struct.Struct("<Q")
_ERRORS = {cls.__name__: cls for cls in (CapabilityError, CompatibilityError, StateError, ValueError)}


def _send(stream, value):
    payload = pickle.dumps(value, protocol=5)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _receive(stream):
    header = stream.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise EOFError("Flash-Next runtime closed its pipe")
    size, = _HEADER.unpack(header)
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("Flash-Next runtime returned an incomplete response")
    return pickle.loads(payload)


def _arrays(values):
    return None if values is None else {name: tensor.detach().cpu().numpy() for name, tensor in values.items()}


def _tensors(values):
    import torch

    return {name: torch.from_numpy(value) for name, value in values.items()}


class FlashSequenceBackend(SequenceBackend):
    state_abi = FLASH_STATE_ABI

    def __init__(self, path, metadata, spec, options, context):
        self.files = model_files(path)
        self.metadata, self.spec, self.options, self.context = metadata, spec, options, context
        self.lock = RLock()
        self.process = self.tokenizer = self.eos = None
        self.closed = False

    def _rpc(self, operation, *arguments):
        with self.lock:
            if self.closed:
                raise StateError("The Flash-Next GGUF model is closed")
            if self.process is None:
                layers = self.options.gpu_layers
                if layers is None:
                    layers = 0 if self.options.library is None else choose_gpu_layers(
                        len(model_geometry(self.spec["repo_id"])["types"]), self.files)
                options = replace(self.options, gpu_layers=layers, library=str(flash_library_path(self.options)))
                self.process = subprocess.Popen([sys.executable, "-u", "-m", __name__],
                                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                self._exchange("open", self.files[0], self.metadata, self.spec, options, self.context)
            return self._exchange(operation, *arguments)

    def _exchange(self, operation, *arguments):
        try:
            _send(self.process.stdin, (operation, arguments))
            error, payload = _receive(self.process.stdout)
        except (EOFError, OSError) as exc:
            raise StateError("Flash-Next native runtime exited during " + operation) from exc
        if error:
            raise _ERRORS.get(error, StateError)(payload)
        return payload

    def codec(self, identity):
        from transformers import AutoTokenizer
        from ..codec import Codec

        with self.lock:
            if self.tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(self.metadata, local_files_only=True)
                codec = Codec(tokenizer, self.spec.get("template_kwargs"))
                chat = codec.render((("user", "LatentPort probe"),), generation_prompt=True)
                samples = ("LatentPort tokenizer probe", chat.rendered_text)
                probes = tuple((sample, codec.encode(sample)) for sample in samples)
                self.eos = self._rpc("validate_codec", probes, tokenizer.all_special_ids)
                self.tokenizer = codec
            return self.tokenizer

    def capture(self, identity, ids):
        return _tensors(self._rpc("capture", identity, ids))

    def advance(self, identity, values, clock, ids):
        return _tensors(self._rpc("advance", identity, _arrays(values), clock, ids))

    def live(self, identity, state=None):
        if state is None:
            return FlashSequence(self, identity)
        state.require_model(identity)
        return FlashSequence(self, identity, state._read_tensors(), state.clock, state.state_id)

    def begin_decode(self, identity, values, clock, *, mtp=False):
        if mtp:
            raise CapabilityError("This GGUF sequence runtime uses native greedy decoding")
        return FlashSequence(self, identity, values, clock, decoding=True)

    def first_logits(self, identity, values, clock):
        import torch

        return torch.from_numpy(self._rpc("first_logits", identity, _arrays(values), clock))

    def install(self, identity, values, clock):
        return self._rpc("install", identity, _arrays(values), clock)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            process = self.process
            if process is None:
                return
            try:
                if process.poll() is None:
                    self._exchange("shutdown")
            except (StateError, OSError):
                pass
            finally:
                process.stdin.close()
                process.stdout.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)


class FlashSequence:
    def __init__(self, backend, identity, values=None, clock=0, parent=None, *, decoding=False):
        self.backend, self.identity = backend, identity
        self.clock, self.parent_id, self.decoding = clock, parent, decoding
        self.logits = values["logits"].clone() if values is not None else None
        self.handle = backend._rpc("sequence", identity, _arrays(values), clock)

    @property
    def physical(self):
        return self.clock

    def append(self, ids):
        import torch

        if self.handle is None:
            raise StateError("The Flash-Next sequence is closed")
        self.clock, scores = self.backend._rpc("append", self.handle, tuple(ids))
        self.logits = torch.from_numpy(scores)
        return self.logits

    def step(self, token):
        self.append((token,))

    def snapshot(self, *, release=False):
        if self.handle is None:
            raise StateError("The Flash-Next sequence is closed")
        values = _tensors(self.backend._rpc("values", self.handle))
        if self.decoding:
            return values
        state = State(self.identity, self.clock, "context_ready", "gguf_live_boundary",
                      target_prompt_tokens=self.clock, parent_id=self.parent_id, _components=values)
        self.parent_id = state.state_id
        if release:
            self.close()
        return state

    def close(self):
        if self.handle is not None:
            if not self.backend.closed and self.backend.process.poll() is None:
                self.backend._rpc("close_sequence", self.handle)
            self.handle = None


def _serve():
    stream, output = sys.stdin.buffer, sys.stdout.buffer
    backend, handles, next_handle = None, {}, 0
    try:
        while True:
            try:
                operation, arguments = _receive(stream)
            except EOFError:
                break
            try:
                if operation == "open":
                    backend = SequenceBackend(*arguments)
                    result = None
                elif operation == "shutdown":
                    backend.close()
                    _send(output, (None, None))
                    break
                elif operation == "sequence":
                    result = next_handle
                    handles[next_handle] = Sequence(backend, *arguments)
                    next_handle += 1
                elif operation == "append":
                    handle, ids = arguments
                    sequence = handles[handle]
                    scores = sequence.append(ids)
                    result = (sequence.clock, scores)
                elif operation == "values":
                    result = handles[arguments[0]].values()
                elif operation == "close_sequence":
                    handles.pop(arguments[0]).close()
                    result = None
                elif operation in ("capture", "advance", "validate_codec", "first_logits", "install"):
                    result = getattr(backend, operation)(*arguments)
                else:
                    raise StateError("Unknown native runtime operation")
                _send(output, (None, result))
            except Exception as exc:
                _send(output, (type(exc).__name__, str(exc)))
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    _serve()
