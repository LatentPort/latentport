"""CPU checks for extended-position APIs, ownership and native dispatch."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from latentport import ContextOptions, Model, ModelIdentity, NativeOptions, State, load_model
from latentport.context import YARN4, PINNED_REVISION, configure_text, execution_weights
from latentport.errors import CapabilityError, CompatibilityError, StateError
from latentport._positions import physical_length, continuation_inputs
from latentport._reference.bf16 import BF16Backend
from latentport._reference.nf4 import NF4Backend, NF4_CONFIG
from latentport._reference.native import _geometry, reference_kernels
from latentport._native.engine import Engine


@pytest.mark.parametrize("clock", [65536,131072,262143,262144,262145,524288,786432,1000000,1048575])
def test_logical_positions_extend_without_allocating_a_million_kv_slots(clock):
    ContextOptions("yarn4").check(clock,3,1)
    inputs=continuation_inputs({"P":torch.tensor([0,17,clock-1])},clock,1,device="cpu",dtype=torch.bfloat16)
    assert inputs["position_ids"].tolist()==[[clock]]
    assert inputs["attention_mask"]["full_attention"].shape==(1,1,1,4)


def test_caps_are_independent_and_invalid_policies_do_not_disable_them():
    for kwargs in ({"position_policy":"dynamic"},{"physical_limit":1048576},{"physical_limit":True}):
        with pytest.raises(ValueError):ContextOptions(**kwargs)
    with pytest.raises(StateError,match="Logical"):ContextOptions().check(262144,3,1)
    with pytest.raises(StateError,match="Logical"):ContextOptions("yarn4").check(1048576,3,1)
    with pytest.raises(StateError,match="Physical"):ContextOptions("yarn4").check(1000000,262144,1)


def test_execution_identity_preserves_legacy_and_rejects_other_position_rules(tmp_path):
    base="bf16-sha256:fixture"
    assert execution_weights(base,ContextOptions())==base
    yarn=execution_weights(base,ContextOptions("yarn4"))
    assert yarn!=base and yarn!=execution_weights(base,ContextOptions("yarn4",131072))
    identity=ModelIdentity("Qwen/Qwen3.8-27B",PINNED_REVISION,"bf16",yarn,"pinned","template", "qwen3.5.complete-hybrid.v1")
    state=State(identity,1000000,"context_ready","fixture",_components={"P":torch.tensor([0,999999])},stream_tail_ids=(3,4))
    loaded=State.load(state.save(tmp_path/"state"))
    assert loaded.stream_tail_ids==(3,4) and loaded.clock==1000000
    with pytest.raises(CompatibilityError):loaded.require_model(replace(identity,weights=base))


def base_config():
    geo=_geometry("target")
    return SimpleNamespace(layer_types=geo["types"],hidden_size=geo["hidden"],num_key_value_heads=geo["kv_heads"],
                           head_dim=geo["head_dim"],linear_num_value_heads=geo["recurrent_heads"],
                           linear_key_head_dim=geo["recurrent_k"],linear_value_head_dim=geo["recurrent_v"],
                           linear_num_key_heads=16,linear_conv_kernel_dim=geo["conv_width"],
                           rope_parameters=deepcopy(geo["rope"]),max_position_embeddings=262144)


def test_static_rope_is_applied_from_zero_without_changing_base_geometry_checks():
    cfg=base_config();out=configure_text(cfg,ContextOptions("yarn4"))
    assert out.max_position_embeddings==1048576 and out.rope_parameters==YARN4
    assert cfg.max_position_embeddings==262144 and cfg.rope_parameters["rope_type"]=="default"
    cfg.hidden_size+=1
    with pytest.raises(CompatibilityError):configure_text(cfg,ContextOptions("yarn4"))


@pytest.mark.parametrize("backend", ["bf16","nf4","gguf"])
def test_public_loader_binds_the_same_position_policy_for_every_backend(tmp_path,backend):
    metadata=tmp_path/"metadata";metadata.mkdir()
    for name in ("config.json","chat_template.jinja","tokenizer.json","tokenizer_config.json"):(metadata/name).write_text("{}")
    (metadata/"model.safetensors").write_bytes(b"fixture weights")
    gguf=tmp_path/"weights.gguf";gguf.write_bytes(b"GGUFfixture")
    with patch("latentport._dependencies.require_backend"),patch("latentport.hub.model_snapshot",return_value=metadata),patch("latentport._native.engine.library_path",return_value=Path("fixture.dll")):
        model=load_model(backend=backend,position_policy="yarn4",model_path=gguf if backend=="gguf" else metadata)
    assert model.backend.options.position_policy=="yarn4"
    assert model.backend.options.logical_limit==1048576
    assert ":execution:" in model.identity.weights and model.identity.backend==backend


def test_extended_loader_rejects_other_revision_before_downloading():
    with patch("latentport.hub.model_snapshot") as download:
        with pytest.raises(CompatibilityError):load_model(position_policy="yarn4",revision="main")
        with pytest.raises(CompatibilityError):load_model(backend="nf4",revision="main")
        download.assert_not_called()


def test_nf4_dependency_failure_is_reported_before_any_model_download():
    from latentport._dependencies import missing_dependencies
    with patch("latentport._dependencies.find_spec",return_value=None),patch("latentport.hub.model_snapshot") as download:
        missing=missing_dependencies("nf4")
        assert {"torch","accelerate","bitsandbytes","fla","causal_conv1d"}.issubset(missing)
        with pytest.raises(CapabilityError,match=r"latentport\[nf4\]"):load_model(backend="nf4")
        download.assert_not_called()


def native_fixture(policy="yarn4",clock=1000000,physical=3,capacity=65536):
    engine=object.__new__(Engine)
    engine.options=NativeOptions(context_size=capacity,position_policy=policy)
    engine.binding=SimpleNamespace(sparse_state=True,check=lambda x:x)
    engine.lib=SimpleNamespace(lp_clock=lambda _:clock,lp_kv_positions=lambda *a:physical)
    engine._model=lambda _: (None,SimpleNamespace(context_size=capacity))
    return engine


def test_native_python_guard_allows_sparse_million_position_continuation():
    native_fixture()._ensure_room("target",2)
    with pytest.raises(StateError,match="Logical"):native_fixture("default")._ensure_room("target",1)
    with pytest.raises(StateError,match="Physical"):native_fixture(physical=262144)._ensure_room("target",1)


def test_old_native_binary_is_rejected_for_extended_policy_before_model_open():
    engine=native_fixture();engine._closed=False;engine._models={};engine.paths={"target":"fixture.gguf"}
    engine.binding.extended_context=False
    with pytest.raises(CapabilityError,match="rebuild"):Engine._model(engine,"target")


def test_new_native_entrypoint_receives_logical_policy_and_physical_cap():
    engine=native_fixture();engine._closed=False;engine._models={};engine.paths={"target":"fixture.gguf"}
    engine.binding.extended_context=True;calls=[]
    engine.lib.lp_open_extended=lambda *args:calls.append(args) or 17
    engine.geometry = {"target": {"types": [], "hidden": 0, "kv_heads": 0, "head_dim": 0,
                                  "conv_channels": 0, "recurrent_heads": 0}}
    engine.lib.lp_get_info=lambda *args:0
    Engine._model(engine,"target")
    assert calls[0][-2:]==(1,262144)
    assert calls[0][2]==65536


def test_extended_native_dense_cache_can_grow_to_physical_cap_before_first_compaction():
    engine=native_fixture(clock=65536,physical=0,capacity=65536)
    engine._snapshot=lambda _: {"fixture":True}
    sizes=[];engine._resize=lambda side,n:sizes.append(n)
    engine._fields=lambda values:([],[])
    engine.lib.lp_install=lambda *args:0
    engine._ensure_room("target",1024)
    assert sizes==[131072]


@contextmanager
def tiny_backend(kind):
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
    from latentport._reference.architecture import geometry
    torch.set_num_threads(1);torch.manual_seed(136)
    cfg=Qwen3_5TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                        num_attention_heads=2,num_key_value_heads=2,head_dim=16,linear_num_key_heads=2,
                        linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
                        layer_types=["linear_attention"]*3+["full_attention"],rope_parameters=deepcopy(YARN4),max_position_embeddings=1048576)
    cfg._attn_implementation="sdpa"
    net=Qwen3_5ForCausalLM(cfg).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    backend=object.__new__(kind);backend.options=ContextOptions("yarn4");backend.device="cpu"
    backend.kernels=reference_kernels;backend._model=lambda _:net
    identity=ModelIdentity("Qwen/Qwen3.8-27B",PINNED_REVISION,kind.name,"RANDOM_CPU_FIXTURE", "fixture","fixture",backend.state_abi)
    with patch("latentport._reference.native._geometry",return_value=geometry(cfg)):
        yield backend,identity,net


@pytest.mark.parametrize("kind", [BF16Backend,NF4Backend])
def test_torch_append_and_decode_use_original_positions_and_enforce_the_logical_end(kind,tmp_path):
    with tiny_backend(kind) as (backend,identity,net):
        with backend.live(identity) as live:
            live.append([2,3,5,7]);parent=live.snapshot(release=True)
        values=parent._read_tensors()
        fixture=replace(parent,clock=1000000,_components={**values,"P":torch.tensor([0,17,262144,999999])})
        before={k:v.clone() for k,v in fixture._read_tensors().items()}
        out=backend.advance(identity,fixture._tensors(),fixture.clock,[11,13])
        assert out["P"].tolist()==[0,17,262144,999999,1000000,1000001]
        assert all(torch.equal(v,fixture._read_tensors()[k]) for k,v in before.items())
        end=replace(fixture,clock=1048575)
        handle=backend.begin_decode(identity,end._tensors(),end.clock)
        handle.step(17)
        assert handle.clock==1048576
        with pytest.raises(StateError,match="Logical"):handle.step(19)
        handle.close()


@pytest.mark.parametrize("kind", [BF16Backend,NF4Backend])
def test_resident_compaction_calls_public_state_operation_and_keeps_parent_immutable(kind):
    with tiny_backend(kind) as (backend,identity,net):
        model=Model(identity,backend,role="target")
        with model.live() as live:
            live.append([2,3,5,7,11,13,17,19])
            parent=live.snapshot();before={k:v.clone() for k,v in parent._read_tensors().items()}
            calls=[]
            def selector_operation(state,**kwargs):
                calls.append(kwargs)
                values={k:v.index_select(2,torch.tensor([0,7])) if k.startswith(("K.","V.")) else v for k,v in state._read_tensors().items()}
                values["P"]=torch.tensor([0,7])
                return replace(state,_components=values)
            with patch.object(State,"compact",selector_operation):
                compacted=live.compact(ratio=8)
            assert calls==[{"ratio":8,"force":False}]
            assert live.physical==2 and live.clock==8
            live.append([23,29])
            assert compacted.clock==8 and compacted._read_tensors()["P"].tolist()==[0,7]
            assert all(torch.equal(v,parent._read_tensors()[k]) for k,v in before.items())
            assert live.snapshot().stream_tail_ids==(2,3,5,7,11,13,17,19,23,29)


def test_raw_stream_snapshot_can_prepare_a_question_without_historical_text(tmp_path):
    from test_runtime import pair
    _,model,_,_=pair(tmp_path)
    old=model.capture("input\n")
    minimal=replace(old,rendered_text=None,rendered_token_ids=None,messages=(),stream_tail_ids=old.rendered_token_ids[-32:])
    prepared=model.session(minimal).prepare("new question")
    assert prepared.clock>old.clock and prepared.rendered_token_ids is None
    assert len(prepared.stream_tail_ids)<=32


def test_nf4_loader_uses_the_exact_quantization_and_static_position_settings(tmp_path):
    cfg=base_config()
    loaded=SimpleNamespace(generation_config=SimpleNamespace(),is_loaded_in_4bit=True)
    loaded.parameters=lambda:[SimpleNamespace(device=SimpleNamespace(type="cuda",index=0))]
    loaded.buffers=lambda:[]
    loaded.eval=lambda:loaded;loaded.requires_grad_=lambda _:loaded
    backend=object.__new__(NF4Backend);backend.options=ContextOptions("yarn4");backend._models={}
    backend._side=lambda _:"target";backend._path=lambda _:tmp_path
    identity=SimpleNamespace(revision=PINNED_REVISION)
    with patch("torch.cuda.is_available",return_value=True),patch("torch.cuda.device_count",return_value=1), \
         patch("transformers.AutoConfig.from_pretrained",return_value=SimpleNamespace(text_config=cfg)), \
         patch("transformers.BitsAndBytesConfig") as quant, \
         patch("transformers.Qwen3_5ForCausalLM.from_pretrained",return_value=(loaded,{"missing_keys":[],"unexpected_keys":[]})) as load:
        assert backend._model(identity) is loaded
    assert quant.call_args.kwargs=={**NF4_CONFIG,"bnb_4bit_compute_dtype":torch.bfloat16,"bnb_4bit_quant_storage":torch.uint8}
    assert load.call_args.kwargs["config"].rope_parameters==YARN4
    assert load.call_args.kwargs["device_map"]=={"":0}


def test_failed_compaction_restores_live_cache_without_replaying_input():
    with tiny_backend(BF16Backend) as (backend,identity,net):
        with backend.live(identity) as live:
            live.append([2,3,5,7]);before=live.snapshot()
            with patch.object(State,"compact",side_effect=RuntimeError("selector failed")):
                with pytest.raises(RuntimeError,match="selector failed"):live.compact()
            after=live.snapshot()
            assert live.clock==4 and live.physical==4
            assert all(torch.equal(v,after._read_tensors()[k]) for k,v in before._read_tensors().items())
            live.append([11])
            assert live.clock==5


def test_concurrent_torch_live_handles_do_not_share_endpoint_hooks():
    from concurrent.futures import ThreadPoolExecutor
    import time
    with tiny_backend(BF16Backend) as (backend,identity,net):
        real_forward=net.forward
        def delayed_forward(*args,**kwargs):
            time.sleep(.02)
            return real_forward(*args,**kwargs)
        with backend.live(identity) as left,backend.live(identity) as right:
            with patch.object(net,"forward",side_effect=delayed_forward),ThreadPoolExecutor(max_workers=2) as workers:
                tasks=[workers.submit(live.append,[2,3,5]) for live in (left,right)]
                for task in tasks:assert task.result(timeout=15).shape==(1,128)
            assert left.clock==right.clock==3


def test_native_resident_ingestion_calls_state_compact_and_preserves_public_snapshots():
    from latentport.gguf import GGUFBackend
    from latentport._positions import extend_positions
    geo={"types":["linear_attention","full_attention"],"hidden":8,"kv_heads":2,"head_dim":4,
         "conv_channels":3,"conv_width":4,"recurrent_heads":2,"recurrent_k":3,"recurrent_v":3}
    identity=ModelIdentity("Qwen/Qwen3.8-27B",PINNED_REVISION,"gguf","fixture","fixture","fixture",GGUFBackend.state_abi)
    values={"H":torch.ones(1,1,8,dtype=torch.bfloat16),"K.1":torch.zeros(1,2,4,4,dtype=torch.bfloat16),
            "V.1":torch.zeros(1,2,4,4,dtype=torch.bfloat16),"C.0":torch.ones(1,3,4,dtype=torch.bfloat16),
            "R.0":torch.ones(1,2,3,3),"P":torch.tensor([0,17,262144,999999])}
    parent=State(identity,1000000,"context_ready","fixture",_components=values,stream_tail_ids=(3,4))
    class Handle:
        def __init__(self,values,clock):self.values={k:v.clone() for k,v in values.items()};self.clock=clock
        @property
        def physical(self):return self.values["K.1"].shape[2]
        def append(self,ids):
            self.values={k:torch.cat((v,torch.zeros(1,2,len(ids),4,dtype=v.dtype)),dim=2) if k.startswith(("K.","V.")) else v for k,v in self.values.items()}
            self.values["P"]=extend_positions(self.values,self.clock,len(ids));self.clock+=len(ids)
        def snapshot(self):return {k:v.clone() for k,v in self.values.items()}
        def close(self):pass
    engine=SimpleNamespace(begin_append=lambda side,values,clock:Handle(values,clock))
    backend=object.__new__(GGUFBackend);backend.options=NativeOptions(position_policy="yarn4")
    backend._side=lambda _:"target";backend._native=lambda:engine
    with patch("latentport._reference.native._geometry",return_value=geo):
        with Model(identity,backend,role="target").live(parent) as live:
            def operation(state,**kwargs):
                out={k:v[:,:,:2].clone() if k.startswith(("K.","V.")) else v for k,v in state._read_tensors().items()}
                out["P"]=out["P"][:2]
                return replace(state,_components=out)
            with patch.object(State,"compact",autospec=True,side_effect=operation) as compact:
                child=live.compact(ratio=8)
            compact.assert_called_once()
            live.append([5,7])
            assert live.clock==1000002 and live.physical==4
            assert child.physical_kv_length==2 and child.clock==1000000
            assert parent._read_tensors()["P"].tolist()==[0,17,262144,999999]


def test_physical_length_reads_actual_attention_shape_and_survives_save_load(tmp_path):
    identity=ModelIdentity("fixture","revision","bf16","weights","tokenizer","template","abi")
    values={"P":torch.tensor([0,999999]),"K.0":torch.zeros(1,1,2,2,dtype=torch.bfloat16),"V.0":torch.zeros(1,1,2,2,dtype=torch.bfloat16)}
    state=State(identity,1000000,"context_ready","fixture",_components=values)
    assert state.physical_kv_length==2
    assert State.load(state.save(tmp_path/"sparse")).physical_kv_length==2
    with pytest.raises(StateError):replace(state,_components={**values,"P":torch.tensor([0])}).physical_kv_length


@pytest.mark.parametrize("backend,abi", [("bf16","qwen3.5.complete-hybrid.v1"),("nf4","qwen3.5.complete-hybrid.v1"),
                                         ("gguf","llama.cpp.b19cbe9.qwen3.5.complete-hybrid.bf16.v2")])
def test_real_compact_operation_accepts_million_clock_without_rebasing(backend,abi):
    geo={"types":["linear_attention","full_attention"],"hidden":8,"kv_heads":2,"head_dim":4,
         "conv_channels":3,"conv_width":4,"recurrent_heads":2,"recurrent_k":3,"recurrent_v":3}
    n=8192
    identity=ModelIdentity("Qwen/Qwen3.8-27B",PINNED_REVISION,backend,"fixture","fixture","fixture",abi)
    values={"H":torch.ones(1,1,8,dtype=torch.bfloat16),"K.1":torch.zeros(1,2,n,4,dtype=torch.bfloat16),
            "V.1":torch.zeros(1,2,n,4,dtype=torch.bfloat16),"C.0":torch.ones(1,3,4,dtype=torch.bfloat16),
            "R.0":torch.ones(1,2,3,3),"P":torch.arange(n)*122}
    values["P"][-1]=999999
    parent=State(identity,1000000,"context_ready","fixture",_components=values)
    with patch("latentport._reference.native._geometry",return_value=geo), \
         patch("latentport._compact._device",return_value=torch.device("cpu")), \
         patch("latentport._compact.weights.score",return_value=[-1.]*n):
        child=parent.compact(ratio=8,force=True)
    assert child.clock==1000000 and child.physical_kv_length==1024
    assert int(child._read_tensors()["P"][-1])==999999
    assert all(torch.equal(values[k],child._read_tensors()[k]) for k in ("H","R.0","C.0"))
