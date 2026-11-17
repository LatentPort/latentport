"""ctypes binding to the pinned CUDA llama.cpp extension.

Model weights and native working caches stay resident. Immutable Python states
own their tensors; a native working cache is installed once per activation.
Interleaved sessions park to device snapshots, never to host serialized blobs.
"""
from __future__ import annotations
import ctypes as ct
from dataclasses import dataclass
import os
from pathlib import Path
from threading import RLock
from time import perf_counter

from ..errors import CapabilityError, CompatibilityError, StateError
from .._reference.native import _geometry, validate_components
from .._positions import physical_length
from ..context import ContextOptions


@dataclass(frozen=True)
class NativeOptions:
    context_size: int = 18432
    batch_size: int = 512
    threads: int = 8
    mtp_drafts: int = 4
    library: str | None = None
    position_policy: str = "default"
    physical_limit: int = 262144

    def __post_init__(self):
        context = self.context_options
        if type(self.context_size) is not int or not 32 <= self.context_size <= context.physical_limit:
            raise ValueError("context_size must be 32..physical_limit")
        if not 16 <= self.batch_size <= 1024 or not 1 <= self.threads <= 128:
            raise ValueError("invalid batch size or thread count")
        if not 0 <= self.mtp_drafts <= 8:
            raise ValueError("mtp_drafts must be 0..8; zero disables MTP allocation")

    @property
    def context_options(self):
        return ContextOptions(self.position_policy, self.physical_limit)

    @property
    def logical_limit(self):
        return self.context_options.logical_limit


def library_path(explicit=None):
    path = explicit or os.environ.get("LATENTPORT_NATIVE_LIB")
    if path:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise CapabilityError(f"LatentPort native library is missing: {candidate}")
        return candidate
    from ..hub import bulk_home
    name = "llama.dll" if os.name == "nt" else "libllama.so"
    active = bulk_home()/"native/active.json"
    candidates = []
    if active.is_file():
        import json
        version = json.loads(active.read_text())["directory"]
        if Path(version).name != version or ":" in version or version in (".", ".."):
            raise CapabilityError("Invalid native runtime installation receipt")
        candidates.append(bulk_home()/"native"/version/name)
    candidates += [bulk_home()/"native/build/bin"/name,
                  Path(__file__).parent/"lib"/name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise CapabilityError("Native runtime missing. Run: python -m latentport install-native")


class Field(ct.Structure):
    _fields_ = [("kind",ct.c_int32),("layer",ct.c_int32),("data",ct.c_void_p),("bytes",ct.c_uint64)]


class Info(ct.Structure):
    _fields_ = [(name,ct.c_int32) for name in (
        "abi","layers","hidden","kv_heads","head_dim","conv_channels","recurrent_heads",
        "mtp_heads","max_drafts","context_size","vocabulary")]+[("auxiliary_bytes",ct.c_uint64)]


class Receipt(ct.Structure):
    _fields_ = [(name,ct.c_int32) for name in ("count","proposed","accepted","warmup","clock","rolled_back")]


class Trace(ct.Structure):
    _fields_ = [(name,ct.c_uint64) for name in ("prefill_tokens","committed_tokens","verified_tokens",
                "head_calls","installs","exports","model_bytes","context_bytes","compute_bytes","auxiliary_bytes")]
    _fields_ += [("last_install_s",ct.c_double),("last_head_s",ct.c_double)]


class Library:
    def __init__(self, path):
        path = library_path(path)
        self._dll_dirs = []
        if os.name == "nt":
            cuda = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"))
            from ..hub import bulk_home
            for directory in (path.parent, bulk_home()/"native/cuda13", cuda/"bin", cuda/"bin/x64"):
                if directory.is_dir():
                    self._dll_dirs.append(os.add_dll_directory(str(directory)))
        try:
            self.lib = ct.CDLL(str(path))
        except OSError as exc:
            raise CapabilityError(f"Cannot load native runtime and its CUDA dependencies: {path}: {exc}") from exc
        p,i = ct.c_void_p,ct.c_int
        signatures = {
            "lp_abi":([],i), "lp_error":([],ct.c_char_p),
            "lp_open":([ct.c_char_p,i,i,i,i,i],p), "lp_close":([p],None),
            "lp_get_info":([p,ct.POINTER(Info)],i),
            "lp_tokenize":([p,ct.c_char_p,i,ct.POINTER(ct.c_int32),i],i),
            "lp_reset":([p],i), "lp_install":([p,ct.POINTER(Field),i,i],i),
            "lp_export":([p,ct.POINTER(Field),i],i), "lp_prefill":([p,ct.POINTER(ct.c_int32),i],i),
            "lp_logits":([p,ct.POINTER(ct.c_float),i],i), "lp_commit":([p,i],i),
            "lp_head":([p,p,ct.c_uint64],i),
            "lp_greedy":([p,i,ct.POINTER(ct.c_int32),i,ct.POINTER(ct.c_int32),ct.POINTER(Receipt)],i),
            "lp_clock":([p],i), "lp_sync":([p],i),
            "lp_get_trace":([p,ct.POINTER(Trace)],i),
        }
        try:
            for name,(args,result) in signatures.items():
                fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=result
        except AttributeError as exc:
            raise CapabilityError("This is stock llama.cpp, not a LatentPort-enabled build") from exc
        if self.lib.lp_abi()!=2:
            raise CapabilityError("Native ABI differs; rebuild the pinned runtime")
        self.extended_context = hasattr(self.lib, "lp_open_extended")
        if self.extended_context:
            self.lib.lp_open_extended.argtypes = [ct.c_char_p, i, i, i, i, i, i, i]
            self.lib.lp_open_extended.restype = p
        self.sparse_state = all(hasattr(self.lib, name) for name in
                                ("lp_install_sparse", "lp_kv_positions", "lp_resize_context"))
        if self.sparse_state:
            extra = {"lp_install_sparse": ([p,ct.POINTER(Field),i,i,ct.POINTER(ct.c_int32),i],i),
                     "lp_kv_positions": ([p,ct.POINTER(ct.c_int32),i],i),
                     "lp_resize_context": ([p,i],i)}
            for name,(args,result) in extra.items():
                fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=result

    def check(self, result):
        if result < 0:
            raise StateError(self.lib.lp_error().decode("utf-8",errors="replace"))
        return result


class Engine:
    def __init__(self, source, target, bridge, bundle_root, options=None):
        self.options=options or NativeOptions()
        self.paths={side:str(path) for side,path in (("source",source),("target",target)) if path is not None}
        self.binding=Library(self.options.library)
        self.lib=self.binding.lib
        self._models={}; self._active={}; self._lock=RLock(); self._closed=False
        self._cache_reclaims=0
        self.geometry = {side: _geometry(side) for side in self.paths}
        self.bridge=None
        if bridge is not None:
            if self.options.position_policy != "default":
                raise CapabilityError("Extended positions apply to standalone models, not pretrained translator pairs")
            from .bridge import GGUFTranslator
            self.bridge=GGUFTranslator(bridge,bundle_root)
            self.geometry = self.bridge.geo

    def capabilities(self):
        return {"abi":2,"full_hybrid_state_install":True,"device_resident_state":True,
                "immutable_snapshot_sharing":True,"paged_attention_sharing":False,
                "extended_positions":getattr(self.binding,"extended_context",False),
                "logical_position_limit":self.options.logical_limit,"physical_kv_limit":self.options.physical_limit,
                "mtp_commit_and_rollback":self.options.mtp_drafts>0}

    def _model(self, side):
        if side not in self.paths:
            raise CapabilityError(f"No {side} model is configured in this runtime")
        if self._closed:
            raise StateError("Native engine is closed")
        if side not in self._models:
            opts=self.options
            args=(os.fsencode(self.paths[side]),int(side=="target"),opts.context_size,
                  opts.batch_size,opts.threads,opts.mtp_drafts if side=="target" else 0)
            if not opts.context_options.legacy:
                if not getattr(self.binding,"extended_context",False):
                    raise CapabilityError("Native library lacks extended-position support; rebuild with the current native sources")
                handle=self.lib.lp_open_extended(*args,int(opts.position_policy=="yarn4"),opts.physical_limit)
            else:
                handle=self.lib.lp_open(*args)
            if not handle:
                raise CapabilityError(self.lib.lp_error().decode("utf-8",errors="replace"))
            info=Info()
            self.binding.check(self.lib.lp_get_info(handle,ct.byref(info)))
            geometry = self.geometry[side]
            fields = ("hidden", "kv_heads", "head_dim", "conv_channels", "recurrent_heads")
            if info.layers != len(geometry["types"]) or any(getattr(info, key) != geometry[key] for key in fields):
                self.lib.lp_close(handle)
                raise CompatibilityError("Native model geometry differs from the selected bridge")
            self._models[side]=(handle,info)
        return self._models[side]

    def _park(self, side):
        active=self._active.pop(side,None)
        if active is not None and active.dirty:
            active.values=self._snapshot(side)
            active.dirty=False

    @staticmethod
    def _fields(values):
        import torch
        keep=[]; fields=[]
        kinds={"H":0,"K":1,"V":2,"C":3,"R":4}
        for name,value in values.items():
            if name == "P":
                continue # Sparse positions are host metadata, not a CUDA field.
            # Readonly input state is never modified. Packing on the GPU is
            # needed only for noncontiguous translator outputs or disk loads.
            value=value.to(device="cuda:0").contiguous()
            keep.append(value)
            field,_,layer=name.partition(".")
            fields.append(Field(kinds[field],int(layer) if layer else -1,value.data_ptr(),value.numel()*value.element_size()))
        torch.cuda.synchronize(0) # Torch producer -> independent GGML streams
        return (Field*len(fields))(*fields),keep

    def _trim_torch_cache(self):
        # Torch cannot see llama.cpp's allocations. Its idle scratch blocks
        # otherwise consume the last VRAM on long, retained conversations.
        # Reclaim only unused blocks under pressure; live models/states stay
        # on CUDA and short requests keep the allocator's fast reuse path.
        import torch
        free,_=torch.cuda.mem_get_info(0)
        idle=torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0)
        if free < 1024**3 and idle >= 256*1024**2:
            torch.cuda.empty_cache()
            self._cache_reclaims+=1

    def _install(self,side,values,clock):
        validate_components(values,clock,side,native_precision=True,geometry=self.geometry[side])
        physical=physical_length(values,clock)
        self.options.context_options.check(clock,physical)
        if "P" in values:
            if not self.binding.sparse_state:
                raise CapabilityError("Rebuild the native runtime to enable state.compact() continuations")
            self._resize(side,min(self.options.physical_limit,((physical+1024+255)//256)*256))
        else:
            _,info=self._model(side)
            if clock>info.context_size:
                if self.options.context_options.legacy and clock>self.options.context_size:
                    raise StateError("Dense state exceeds configured context capacity")
                self._resize(side,max(clock,self.options.context_size))
        handle,_=self._model(side)
        started=perf_counter()
        fields,keep=self._fields(values)
        transferred=perf_counter()
        if "P" in values:
            raw=values["P"].cpu().tolist()
            positions=(ct.c_int32*len(raw))(*raw)
            self.binding.check(self.lib.lp_install_sparse(handle,fields,len(fields),clock,positions,len(raw)))
        else:
            self.binding.check(self.lib.lp_install(handle,fields,len(fields),clock))
        trace=self.trace(side)
        return {"device_transfer_s":transferred-started,"state_install_s":trace["last_install_s"],
                "target_first_token_compute_s":trace["last_head_s"]}

    def _resize(self,side,capacity):
        if not 32 <= capacity <= self.options.physical_limit:
            raise StateError("Working cache exceeds the physical attention limit")
        handle,info=self._model(side)
        if capacity==info.context_size:
            return
        if not self.binding.sparse_state:
            raise CapabilityError("Native working-cache resize requires an updated build")
        self.binding.check(self.lib.lp_resize_context(handle,capacity))
        self.binding.check(self.lib.lp_get_info(handle,ct.byref(info)))

    def _ensure_room(self,side,count,*,partial=False):
        handle,info=self._model(side)
        clock=self.binding.check(self.lib.lp_clock(handle))
        if partial:
            count=min(count,self.options.logical_limit-clock)
        n=self.binding.check(self.lib.lp_kv_positions(handle,None,0)) if self.binding.sparse_state else 0
        physical=n or clock
        if partial:
            count=min(count,self.options.physical_limit-physical)
            if count <= 0:
                raise StateError("No remaining logical or physical attention capacity")
        self.options.context_options.check(clock,physical,count)
        if physical+count<=info.context_size:
            return
        if not n and self.options.context_options.legacy and physical+count>self.options.context_size:
            if partial:
                return # Original lp_greedy commits only the available suffix.
            raise StateError("Dense continuation exceeds configured context capacity")
        values=self._snapshot(side)
        capacity=(min(self.options.physical_limit,max(physical+count,info.context_size*2))
                  if n or not self.options.context_options.legacy else self.options.context_size)
        self._resize(side,capacity)
        fields,keep=self._fields(values)
        if n:
            positions=(ct.c_int32*n)(*values["P"].cpu().tolist())
            self.binding.check(self.lib.lp_install_sparse(handle,fields,len(fields),clock,positions,n))
        else:
            self.binding.check(self.lib.lp_install(handle,fields,len(fields),clock))

    def _snapshot(self,side):
        import torch
        handle,_=self._model(side)
        clock=self.binding.check(self.lib.lp_clock(handle))
        n=self.binding.check(self.lib.lp_kv_positions(handle,None,0)) if self.binding.sparse_state else 0
        physical=n or clock
        g=self.geometry[side]
        shapes={"H":(1,1,g["hidden"]),"K":(1,g["kv_heads"],physical,g["head_dim"]),
                "V":(1,g["kv_heads"],physical,g["head_dim"]),"C":(1,g["conv_channels"],g["conv_width"]),
                "R":(1,g["recurrent_heads"],g["recurrent_k"],g["recurrent_v"])}
        names=["H"]
        for i,kind in enumerate(g["types"]):
            names.extend(f"{f}.{i}" for f in (("K","V") if kind=="full_attention" else ("C","R")))
        values={name:torch.empty(shapes[name[0]],dtype=torch.float32 if name[0] in "HCR" else torch.bfloat16,
                                device="cuda:0") for name in names}
        fields,keep=self._fields(values)
        self.binding.check(self.lib.lp_export(handle,fields,len(fields)))
        if n:
            positions=(ct.c_int32*n)()
            self.binding.check(self.lib.lp_kv_positions(handle,positions,n))
            values["P"]=torch.tensor(list(positions),device="cuda:0",dtype=torch.long)
        self._trim_torch_cache()
        return values

    def _logits(self,side):
        import numpy as np
        handle,info=self._model(side)
        values=np.empty((1,info.vocabulary),dtype=np.float32)
        self.binding.check(self.lib.lp_logits(handle,values.ctypes.data_as(ct.POINTER(ct.c_float)),info.vocabulary))
        return values

    def tokenize(self,side,text):
        with self._lock:
            handle,_=self._model(side)
            data=text.encode("utf-8")
            size=self.lib.lp_tokenize(handle,data,len(data),None,0)
            if size>=0:
                return ()
            out=(ct.c_int32*(-size))()
            n=self.binding.check(self.lib.lp_tokenize(handle,data,len(data),out,len(out)))
            return tuple(out[:n])

    def capture(self,side,ids):
        if not ids:
            raise StateError("Prefill requires one or more tokens")
        self.options.context_options.check(0,0,len(ids))
        with self._lock:
            self._park(side)
            handle,_=self._model(side)
            _,info=self._model(side)
            if len(ids)>info.context_size:
                if self.options.context_options.legacy and len(ids)>self.options.context_size:
                    raise StateError("Prefill exceeds configured context capacity")
                self._resize(side,max(len(ids),self.options.context_size))
            self.binding.check(self.lib.lp_reset(handle))
            tokens=(ct.c_int32*len(ids))(*ids)
            self.binding.check(self.lib.lp_prefill(handle,tokens,len(ids)))
            return self._snapshot(side)

    def advance(self,side,values,clock,ids):
        with self._lock:
            self._park(side);self._install(side,values,clock)
            self._ensure_room(side,len(ids))
            handle,_=self._model(side)
            tokens=(ct.c_int32*len(ids))(*ids)
            self.binding.check(self.lib.lp_prefill(handle,tokens,len(ids)))
            return self._snapshot(side)

    def check_install(self,side,values,clock):
        with self._lock:
            self._park(side);self._install(side,values,clock)

    def first_logits(self,side,h):
        import torch
        with self._lock:
            self._park(side)
            handle,_=self._model(side)
            h=h.to("cuda:0").contiguous()
            if h.dtype not in (torch.bfloat16,torch.float32) or tuple(h.shape)!=(1,1,self.geometry[side]["hidden"]):
                raise StateError("Invalid terminal H")
            torch.cuda.synchronize(0)
            self.binding.check(self.lib.lp_head(handle,h.data_ptr(),h.numel()*h.element_size()))
            return self._logits(side)

    def begin_decode(self,side,values,clock,mtp=False):
        with self._lock:
            if mtp and (side!="target" or not self.options.mtp_drafts):
                raise CapabilityError("MTP needs the target GGUF's trained heads and mtp_drafts > 0")
            handle=DecodeHandle(self,side,values,clock,mtp)
            handle._activate()
            return handle

    def begin_append(self,side,values=None,clock=0):
        with self._lock:
            handle=AppendHandle(self,side,values,clock,False)
            handle._activate()
            return handle

    def eos_ids(self,side):
        return (248044,248046)

    def translate(self,values,metadata):
        import torch
        with self._lock:
            clock=next(v.shape[2] for k,v in values.items() if k.startswith("K."))
            bridge_values={k:v.to(torch.bfloat16) if k[0] in "HC" else v for k,v in values.items()}
            result=self.bridge.translate(bridge_values,clock,metadata)
            self._trim_torch_cache()
            return result

    def warm(self):
        with self._lock:
            for side in self.paths:
                self._model(side)
            if self.bridge is not None:
                self.bridge.warm()

    def residency(self):
        import torch
        self.warm()
        torch.cuda.synchronize(0)
        free,total=torch.cuda.mem_get_info(0)
        return {"device":"cuda:0","gpu_total_bytes":int(total),"gpu_free_bytes":int(free),
                "torch_allocated_bytes":int(torch.cuda.memory_allocated(0)),
                "torch_reserved_bytes":int(torch.cuda.memory_reserved(0)),
                "torch_cache_reclaims":self._cache_reclaims,
                "source_resident":"source" in self._models,"target_resident":"target" in self._models,
                "bridge_resident":self.bridge is not None,
                "mtp_resident":bool(self.options.mtp_drafts and "target" in self._models),
                "native_auxiliary_bytes":sum(x[1].auxiliary_bytes for x in self._models.values()),
                "context_capacity":self.options.context_size,"cache_dtype":"bf16",
                "mtp_heads":self._models["target"][1].mtp_heads if "target" in self._models else 0,
                "paged_attention_sharing":False,"snapshot_sharing":True,
                "native":{side:self.trace(side) for side in self._models}}

    def trace(self,side):
        with self._lock:
            native,_=self._model(side);trace=Trace()
            self.binding.check(self.lib.lp_get_trace(native,ct.byref(trace)))
            return {name:getattr(trace,name) for name,_ in trace._fields_}

    def close(self):
        with self._lock:
            for handle in self._active.values():
                handle.values=None;handle.closed=True
            self._active.clear()
            for handle,_ in self._models.values():
                self.lib.lp_close(handle)
            self._models.clear();self._closed=True
            if self.bridge is not None:
                self.bridge._loaded=None


class DecodeHandle:
    def __init__(self,engine,side,values,clock,mtp):
        self.engine,self.side,self.values,self.clock,self.mtp=engine,side,values,clock,mtp
        self.dirty=False;self.closed=False
        self.proposed=0;self.accepted=0;self.warmup=0;self.batches=0
        self.install_timings={}

    def _activate(self):
        if self.closed:
            raise StateError("Decode handle is closed")
        e=self.engine
        if e._active.get(self.side) is not self:
            e._park(self.side)
            self.install_timings=e._install(self.side,self.values,self.clock)
            e._active[self.side]=self

    def logits(self):
        with self.engine._lock:
            self._activate()
            return self.engine._logits(self.side)

    def enable_mtp(self):
        if self.side != "target" or not self.engine.options.mtp_drafts:
            raise CapabilityError("This native context was built without MTP")
        self.mtp = True

    def commit(self,token):
        with self.engine._lock:
            self._activate()
            e=self.engine;native,_=e._model(self.side)
            e._ensure_room(self.side,1)
            e.binding.check(e.lib.lp_commit(native,token))
            self.clock=e.binding.check(e.lib.lp_clock(native));self.dirty=True
            if self.mtp and self.batches==0:
                self.warmup+=1
            return {"committed_clock":self.clock,"rejected_drafts_rolled_back":True}

    def greedy_batch(self,limit,eos):
        if not self.mtp:
            raise CapabilityError("Batched speculation was not enabled on this handle")
        with self.engine._lock:
            self._activate()
            e=self.engine;native,_=e._model(self.side)
            e._ensure_room(self.side,min(limit,9),partial=True)
            stops=tuple(eos);eos_array=(ct.c_int32*len(stops))(*stops)
            out=(ct.c_int32*9)();receipt=Receipt()
            e.binding.check(e.lib.lp_greedy(native,limit,eos_array,len(stops),out,ct.byref(receipt)))
            if not receipt.rolled_back or not 1<=receipt.count<=min(9,limit) or receipt.clock!=self.clock+receipt.count:
                raise StateError("Invalid native speculative commit receipt")
            tokens=tuple(out[:receipt.count])
            if any(t in stops for t in tokens[:-1]):
                raise StateError("Native MTP committed tokens beyond EOS")
            self.clock=receipt.clock;self.dirty=True
            self.proposed+=receipt.proposed;self.accepted+=receipt.accepted;self.warmup+=receipt.warmup
            self.batches+=int(receipt.proposed>0)
            return tokens

    def snapshot(self):
        with self.engine._lock:
            self._activate()
            self.values=self.engine._snapshot(self.side);self.dirty=False
            return self.values

    def stats(self):
        return {"mtp_proposed_tokens":self.proposed,"mtp_accepted_tokens":self.accepted,
                "mtp_warmup_tokens":self.warmup,"mtp_batches":self.batches,**self.install_timings}

    def close(self):
        with self.engine._lock:
            if self.engine._active.get(self.side) is self:
                self.engine._active.pop(self.side)
            self.values=None;self.closed=True


class AppendHandle(DecodeHandle):
    """Resident append transaction; exports only at explicit snapshot boundaries."""
    def _activate(self):
        if self.closed:
            raise StateError("Append handle is closed or failed")
        e=self.engine
        if e._active.get(self.side) is not self:
            e._park(self.side)
            if self.values is None and self.clock==0:
                native,_=e._model(self.side)
                e.binding.check(e.lib.lp_reset(native))
            else:
                self.install_timings=e._install(self.side,self.values,self.clock)
            e._active[self.side]=self

    @property
    def physical(self):
        with self.engine._lock:
            self._activate()
            e=self.engine;native,_=e._model(self.side)
            n=e.binding.check(e.lib.lp_kv_positions(native,None,0)) if e.binding.sparse_state else 0
            return n or self.clock

    def append(self,ids):
        if not ids:
            raise StateError("append requires one or more tokens")
        with self.engine._lock:
            self._activate()
            e=self.engine
            e._ensure_room(self.side,len(ids))
            native,_=e._model(self.side)
            tokens=(ct.c_int32*len(ids))(*ids)
            try:
                e.binding.check(e.lib.lp_prefill(native,tokens,len(ids)))
                self.clock=e.binding.check(e.lib.lp_clock(native));self.dirty=True
                return self.logits()
            except BaseException:
                self.close()
                raise
