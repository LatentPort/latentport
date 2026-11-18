"""Native llama.cpp sequences for standalone architectures outside the tensor ABI."""
import ctypes as ct
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
from threading import RLock
from types import SimpleNamespace

from .engine import Library, library_path
from .._models import FLASH_NEXT, GEMMA, QWEN3, model_geometry
from .._tokens import token_ids
from ..context import execution_weights
from ..errors import CapabilityError, CompatibilityError, StateError
from ..state import State


STATE_ABI = "llama.cpp.b19cbe9.sequence.v1"
FLASH_STATE_ABI = "llama.cpp.b12503.sequence.v1"
ARCHITECTURES = {FLASH_NEXT: b"qwen4exp", GEMMA: b"gemma4", QWEN3: b"qwen3"}
SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf")

# The by-value layouts come from include/llama.h at the revision pinned in
# native-release.json. The native library also keeps the LatentPort ABI check.
P, I, U, F, B = ct.c_void_p, ct.c_int32, ct.c_uint32, ct.c_float, ct.c_bool


class ModelParams(ct.Structure):
    _fields_ = [
        ("devices", P), ("tensor_buft_overrides", P), ("n_gpu_layers", I),
        ("split_mode", I), ("load_mode", I), ("tensor_read_lazy", I), ("main_gpu", I),
        ("tensor_split", P), ("progress_callback", P), ("progress_callback_user_data", P),
        ("kv_overrides", P), ("vocab_only", B), ("check_tensors", B), ("use_extra_bufts", B),
        ("no_host", B), ("no_alloc", B), ("load_mtp", B),
    ]


class ContextParams(ct.Structure):
    _fields_ = [
        ("n_ctx", U), ("n_batch", U), ("n_ubatch", U), ("n_seq_max", U), ("n_rs_seq", U),
        ("n_outputs_max", U), ("n_outputs_max_per_seq", U), ("n_threads", I), ("n_threads_batch", I),
        ("ctx_type", I), ("rope_scaling_type", I), ("pooling_type", I), ("attention_type", I),
        ("flash_attn_type", I), ("rope_freq_base", F), ("rope_freq_scale", F), ("yarn_ext_factor", F),
        ("yarn_attn_factor", F), ("yarn_beta_fast", F), ("yarn_beta_slow", F), ("yarn_orig_ctx", U),
        ("defrag_thold", F), ("cb_eval", P), ("cb_eval_user_data", P), ("type_k", I), ("type_v", I),
        ("abort_callback", P), ("abort_callback_data", P), ("embeddings", B), ("offload_kqv", B),
        ("no_perf", B), ("op_offload", B), ("swa_full", B), ("kv_unified", B),
        ("samplers", P), ("n_samplers", ct.c_size_t), ("ctx_other", P),
    ]


class FlashContextParams(ct.Structure):
    _fields_ = [field for field in ContextParams._fields_ if field[0] != "defrag_thold"]
    _position = next(i for i, field in enumerate(_fields_) if field[0] == "abort_callback")
    _fields_[_position:_position] = [("moe_cache_size", ct.c_size_t), ("moe_cache_cpu_share", F)]


def _stock_library(path):
    dirs = [os.add_dll_directory(str(path.parent))] if os.name == "nt" else []
    base_name = "ggml-base.dll" if os.name == "nt" else "libggml-base.so"
    base = ct.CDLL(str(path.with_name(base_name)))
    base.ggml_commit.restype = ct.c_char_p
    if not (base.ggml_commit() or b"").startswith(b"4577932fa"):
        raise CapabilityError("Flash-Next GGUF requires a llama.cpp b12503 library")
    lib = ct.CDLL(str(path))
    name = "ggml.dll" if os.name == "nt" else "libggml.so"
    ggml = ct.CDLL(str(path.with_name(name)))
    ggml.ggml_backend_load_all_from_path.argtypes = [ct.c_char_p]
    ggml.ggml_backend_load_all_from_path.restype = None
    ggml.ggml_backend_load_all_from_path(os.fsencode(path.parent))
    return SimpleNamespace(lib=lib, dirs=dirs, ggml=ggml, base=base)


class Batch(ct.Structure):
    _fields_ = [("n_tokens", I), ("token", P), ("embd", P), ("pos", P), ("n_seq_id", P),
                ("seq_id", P), ("logits", P)]


@lru_cache(maxsize=4)
def _native(path, flash=False):
    binding = _stock_library(path) if flash else Library(path)
    lib = binding.lib
    size = ct.c_size_t
    context_params = FlashContextParams if flash else ContextParams
    signatures = {
        "llama_backend_init": ([], None),
        "llama_supports_gpu_offload": ([], B),
        "llama_model_default_params": ([], ModelParams),
        "llama_context_default_params": ([], context_params),
        "llama_model_load_from_file": ([ct.c_char_p, ModelParams], P),
        "llama_model_free": ([P], None),
        "llama_model_meta_val_str": ([P, ct.c_char_p, P, size], I),
        "llama_model_n_embd": ([P], I),
        "llama_model_n_layer": ([P], I),
        "llama_model_get_vocab": ([P], P),
        "llama_vocab_n_tokens": ([P], I),
        "llama_vocab_is_eog": ([P, I], B),
        "llama_tokenize": ([P, ct.c_char_p, I, P, I, B, B], I),
        "llama_init_from_model": ([P, context_params], P),
        "llama_free": ([P], None),
        "llama_get_memory": ([P], P),
        "llama_memory_seq_pos_max": ([P, I], I),
        "llama_state_seq_get_size": ([P, I], size),
        "llama_state_seq_get_data": ([P, P, size, I], size),
        "llama_state_seq_set_data": ([P, P, size, I], size),
        "llama_decode": ([P, Batch], I),
        "llama_get_logits_ith": ([P, I], ct.POINTER(F)),
    }
    for name, (args, result) in signatures.items():
        try:
            function = getattr(lib, name)
        except AttributeError as exc:
            raise CapabilityError("The installed native library has an older llama.cpp sequence API") from exc
        function.argtypes, function.restype = args, result
    lib.llama_backend_init()
    return binding


def model_files(path):
    path = Path(path).expanduser().absolute()
    match = SHARD.fullmatch(path.name)
    if match:
        prefix, number, count = match.groups()
        if not 1 <= int(number) <= int(count):
            raise CompatibilityError("Invalid GGUF shard numbering")
        files = tuple(path.with_name(f"{prefix}-{part:05d}-of-{count}.gguf")
                      for part in range(1, int(count) + 1))
    else:
        files = (path,)
    for file in files:
        with file.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise CompatibilityError(f"Invalid GGUF container: {file}")
    return files


def choose_gpu_layers(layers, files):
    import torch

    free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    available = max(0, free - 4 * 1024**3)
    weight_bytes = sum(file.stat().st_size for file in files)
    return -1 if available > weight_bytes else layers * available // weight_bytes


class SequenceBackend:
    name = "gguf"
    state_abi = STATE_ABI
    device = "cpu"
    supports_mtp = False
    copy_on_write = True

    def __init__(self, path, metadata, spec, options, context):
        self.files = model_files(path)
        self.metadata, self.spec = metadata, spec
        self.options, self.context = options, context
        self.array_only = spec["repo_id"] == FLASH_NEXT
        if self.array_only:
            self.state_abi = FLASH_STATE_ABI
            path = Path(options.library).resolve()
            self.binding = _native(path, flash=True)
        else:
            self.binding = _native(library_path(options.library))
        self.lib = self.binding.lib
        self.lock = RLock()
        self.active = set()
        self.model = self.vocab = self.tokenizer = self.eos = None
        self.closed = False

    def weights(self):
        from ..gguf import _hash_file

        hashes = [_hash_file(path) for path in self.files]
        digest = hashes[0] if len(hashes) == 1 else hashlib.sha256(bytes.fromhex("".join(hashes))).hexdigest()
        return execution_weights("gguf-sha256:" + digest, self.context)

    def check_capacity(self, clock, physical, count=0):
        self.context.check(clock, physical, count)
        if physical + count > self.options.context_size:
            raise StateError("GGUF sequence exceeds the configured native context size")

    def _open(self):
        with self.lock:
            if self.closed:
                raise StateError("The GGUF model is closed")
            if self.model is not None:
                return
            params = self.lib.llama_model_default_params()
            params.load_mtp = False
            params.split_mode = 0
            repo = self.spec["repo_id"]
            geometry = model_geometry(repo)
            layers = len(geometry["types"])
            gpu_layers = self.options.gpu_layers
            if gpu_layers is None:
                gpu_layers = 0
                if self.lib.llama_supports_gpu_offload() and not self.array_only:
                    gpu_layers = choose_gpu_layers(layers, self.files)
            if gpu_layers and not self.lib.llama_supports_gpu_offload():
                raise CapabilityError("The selected llama.cpp library has no GPU backend")
            params.n_gpu_layers = gpu_layers
            model = self.lib.llama_model_load_from_file(os.fsencode(self.files[0]), params)
            if not model:
                raise CompatibilityError(f"llama.cpp could not load {self.files[0].name}")
            arch = ct.create_string_buffer(32)
            self.lib.llama_model_meta_val_str(model, b"general.architecture", arch, len(arch))
            if (arch.value != ARCHITECTURES[repo] or self.lib.llama_model_n_embd(model) != geometry["hidden"]
                    or self.lib.llama_model_n_layer(model) != layers):
                self.lib.llama_model_free(model)
                raise CompatibilityError("GGUF architecture or model geometry differs from the selected profile")
            self.model = model
            self.vocab = self.lib.llama_model_get_vocab(model)
            self.vocabulary = self.lib.llama_vocab_n_tokens(self.vocab)

    def _tokens(self, text):
        text = text.encode("utf-8")
        size = max(32, len(text) + 16)
        array = (I * size)()
        count = self.lib.llama_tokenize(self.vocab, text, len(text), array, size, False, True)
        if count < 0:
            array = (I * -count)()
            count = self.lib.llama_tokenize(self.vocab, text, len(text), array, -count, False, True)
        if count < 0:
            raise CompatibilityError("Native GGUF tokenizer could not encode a probe")
        return tuple(array[:count])

    def codec(self, identity):
        from transformers import AutoTokenizer
        from ..codec import Codec

        with self.lock:
            if self.tokenizer is not None:
                return self.tokenizer
            self._open()
            kind = Codec
            if identity.repo_id == GEMMA:
                from .._reference.gemma import GemmaCodec
                kind = GemmaCodec
            tokenizer = AutoTokenizer.from_pretrained(self.metadata, local_files_only=True)
            codec = kind(tokenizer, self.spec.get("template_kwargs"))
            chat = codec.render((("user", "LatentPort probe"),), generation_prompt=True)
            samples = ("LatentPort tokenizer probe", chat.rendered_text)
            self.eos = self.validate_codec(tuple((sample, codec.encode(sample)) for sample in samples),
                                           tokenizer.all_special_ids)
            self.tokenizer = codec
            return codec

    def validate_codec(self, probes, special_ids):
        with self.lock:
            self._open()
            if any(self._tokens(text) != expected for text, expected in probes):
                raise CompatibilityError("GGUF tokenizer differs from the pinned model tokenizer")
            return frozenset(i for i in special_ids
                             if i < self.vocabulary and self.lib.llama_vocab_is_eog(self.vocab, i))

    def _validate(self, values, clock):
        if self.array_only:
            import numpy as data
        else:
            import torch as data
        self.check_capacity(clock, clock)
        if (values.keys() != {"sequence", "logits"} or values["sequence"].dtype != data.uint8
                or values["sequence"].ndim != 1 or not values["sequence"].shape[0]
                or values["logits"].dtype != data.float32 or values["logits"].shape != (1, self.vocabulary)):
            raise StateError("Invalid GGUF sequence snapshot")

    def capture(self, identity, ids):
        self.check_capacity(0, 0, len(ids))
        with Sequence(self, identity) as handle:
            handle.append(ids)
            return handle.values()

    def advance(self, identity, values, clock, ids):
        self.check_capacity(clock, clock, len(ids))
        with Sequence(self, identity, values, clock) as handle:
            handle.append(ids)
            return handle.values()

    def live(self, identity, state=None):
        if state is None:
            return Sequence(self, identity)
        state.require_model(identity)
        return Sequence(self, identity, state._read_tensors(), state.clock, parent=state.state_id)

    def begin_decode(self, identity, values, clock, *, mtp=False):
        if mtp:
            raise CapabilityError("This GGUF sequence runtime uses native greedy decoding")
        return SequenceDecode(Sequence(self, identity, values, clock))

    def first_logits(self, identity, values, clock):
        self._open()
        self._validate(values, clock)
        return values["logits"].copy() if self.array_only else values["logits"].clone()

    def eos_ids(self, identity):
        self.codec(identity)
        return self.eos

    def install(self, identity, values, clock):
        with Sequence(self, identity, values, clock):
            pass

    def close(self):
        with self.lock:
            for handle in tuple(self.active):
                handle.close()
            if self.model is not None:
                self.lib.llama_model_free(self.model)
                self.model = None
            self.closed = True


class Sequence:
    def __init__(self, backend, identity, values=None, clock=0, *, parent=None):
        self.backend, self.identity = backend, identity
        self.clock, self.parent_id = clock, parent
        self.logits, self.context = None, None
        self.failed = False
        lib = backend.lib
        with backend.lock:
            backend._open()
            if values is not None:
                backend._validate(values, clock)
            params = lib.llama_context_default_params()
            options = backend.options
            params.n_ctx = options.context_size
            params.n_batch = min(options.batch_size, options.context_size)
            params.n_ubatch = min(params.n_batch, 128)
            params.n_seq_max = 1
            params.n_threads = params.n_threads_batch = options.threads
            self.context = lib.llama_init_from_model(backend.model, params)
            if not self.context:
                raise CapabilityError("Could not allocate the native GGUF context")
            backend.active.add(self)
            try:
                if values is not None:
                    if backend.array_only:
                        import numpy as np
                        raw = np.ascontiguousarray(values["sequence"])
                        address, size = raw.ctypes.data, raw.size
                    else:
                        raw = values["sequence"].cpu().contiguous()
                        address, size = raw.data_ptr(), raw.numel()
                    if lib.llama_state_seq_set_data(self.context, address, size, 0) != size:
                        raise StateError("llama.cpp rejected the saved sequence")
                    position = lib.llama_memory_seq_pos_max(lib.llama_get_memory(self.context), 0)
                    if position != clock - 1:
                        raise StateError("Restored GGUF sequence clock differs from its metadata")
                    logits = values["logits"]
                    self.logits = logits.copy() if backend.array_only else logits.cpu().clone()
            except BaseException:
                self.close()
                raise

    @property
    def physical(self):
        return self.clock

    @property
    def closed(self):
        return self.context is None

    def append(self, ids):
        import numpy as np

        backend = self.backend
        ids = token_ids(ids, vocabulary_size=backend.vocabulary)
        lib = backend.lib
        size = min(backend.options.batch_size, backend.options.context_size)
        with backend.lock:
            if self.closed or self.failed:
                raise StateError("The GGUF sequence is closed or failed")
            backend.check_capacity(self.clock, self.clock, len(ids))
            try:
                for start in range(0, len(ids), size):
                    group = ids[start : start + size]
                    tokens = (I * len(group))(*group)
                    positions = (I * len(group))(*range(self.clock, self.clock + len(group)))
                    last = (ct.c_int8 * len(group))()
                    last[-1] = 1
                    batch = Batch(len(group), ct.addressof(tokens), None, ct.addressof(positions),
                                  None, None, ct.addressof(last))
                    status = lib.llama_decode(self.context, batch)
                    if status:
                        raise StateError(f"Native GGUF decode failed with status {status}")
                    self.clock += len(group)
                scores = lib.llama_get_logits_ith(self.context, -1)
                if not scores:
                    raise StateError("Native GGUF decode returned no next-token logits")
                array = np.ctypeslib.as_array(scores, shape=(backend.vocabulary,)).copy()
                if backend.array_only:
                    self.logits = array[None, :]
                else:
                    import torch
                    self.logits = torch.from_numpy(array).unsqueeze(0)
                return self.logits
            except BaseException:
                self.failed = True
                raise

    def values(self):
        lib = self.backend.lib
        with self.backend.lock:
            if self.closed or self.failed or self.logits is None:
                raise StateError("There is no complete GGUF sequence to snapshot")
            size = lib.llama_state_seq_get_size(self.context, 0)
            if self.backend.array_only:
                import numpy as np
                data = np.empty(size, dtype=np.uint8)
                address, logits = data.ctypes.data, self.logits.copy()
            else:
                import torch
                data = torch.empty(size, dtype=torch.uint8, device="cpu")
                address, logits = data.data_ptr(), self.logits.clone()
            if not size or lib.llama_state_seq_get_data(self.context, address, size, 0) != size:
                raise StateError("llama.cpp could not serialize the full sequence")
            return {"sequence": data, "logits": logits}

    def snapshot(self, *, release=False):
        state = State(self.identity, self.clock, "context_ready", "gguf_live_boundary",
                      target_prompt_tokens=self.clock, parent_id=self.parent_id, _components=self.values())
        self.parent_id = state.state_id
        if release:
            self.close()
        return state

    def close(self):
        with self.backend.lock:
            if self.context is not None:
                self.backend.lib.llama_free(self.context)
                self.context = None
                self.backend.active.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SequenceDecode:
    def __init__(self, sequence):
        self.sequence = sequence

    @property
    def logits(self):
        return self.sequence.logits

    @property
    def clock(self):
        return self.sequence.clock

    def step(self, token):
        self.sequence.append((token,))

    def snapshot(self):
        return self.sequence.values()

    def close(self):
        self.sequence.close()
