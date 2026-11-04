"""Bounded CUDA runtime correctness and optional latency checks."""
from __future__ import annotations
import argparse
import ctypes as ct
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import latentport as lp
from latentport.hub import bulk_home


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-is-free",action="store_true",help="Confirm the GPU is available for runtime checks")
    parser.add_argument("--source-path",type=Path,required=True)
    parser.add_argument("--target-path",type=Path,required=True)
    parser.add_argument("--bridge-gguf-path",type=Path)
    parser.add_argument("--context-size",type=int,default=18432)
    parser.add_argument("--context-tokens",type=int,default=512)
    parser.add_argument("--mtp-drafts",type=int,default=4)
    parser.add_argument("--performance-contexts",type=int,nargs="*",default=[],
                        help="Optional small latency check at these approximate context lengths")
    parser.add_argument("--output-root",type=Path,default=bulk_home()/"gguf-smoke")
    args=parser.parse_args()
    if not args.gpu_is_free:
        parser.error("Pass --gpu-is-free when the GPU is available for these checks")
    if not 1 <= args.mtp_drafts <= 8:
        parser.error("This test exercises MTP; --mtp-drafts must be 1..8")
    if not 32 <= args.context_tokens < args.context_size-512:
        parser.error("Reserve at least 512 positions after the source context")
    if any(n<32 or n>args.context_size-512 for n in args.performance_contexts):
        parser.error("Performance contexts must reserve at least 512 positions")
    import torch
    output=args.output_root.resolve()/uuid4().hex
    output.mkdir(parents=True,exist_ok=False)
    report={"quality_evaluation":False,"checks":{},"success":False}
    port=None
    try:
        cold_started=perf_counter()
        port=lp.load(backend="gguf",source_path=args.source_path,target_path=args.target_path,
                     bridge_gguf_path=args.bridge_gguf_path,
                     native_options=lp.NativeOptions(context_size=args.context_size,mtp_drafts=args.mtp_drafts))
        report["configuration"]={"source":asdict(port.source.identity),"target":asdict(port.target.identity),
                                 "context_size":args.context_size,"mtp_drafts":args.mtp_drafts,
                                 "source_path":str(args.source_path),"target_path":str(args.target_path)}
        engine=port.source.backend._native()
        report["residency_before"]=port.warm(mtp=True)
        # Resolve and validate cached tokenizers before timed requests too.
        port.source.codec;port.target.codec
        report["cold_prepare_s"]=perf_counter()-cold_started
        prompt=("A cache stores previous results. Entries have keys, values, expiry times and versions. "*
                (args.context_tokens//18+1))
        prompt=port.source.codec.decode(port.source.codec.encode(prompt)[:args.context_tokens],special=True)
        source=port.source.capture((("user",prompt+"\nExplain ten cache failure modes in detail."),),generation_prompt=True)
        assert source.info.tensor_count==65
        report["source_clock"]=source.clock
        path=source.save(output/"source.lpstate")
        restored=lp.State.load(path)
        delta=port.source.codec.encode(" Additional facts.")
        # Compare continuation of the live native cache with typed reinstallation.
        native,_=engine._model("source")
        array=(ct.c_int32*len(delta))(*delta)
        engine.binding.check(engine.lib.lp_prefill(native,array,len(delta)))
        uninterrupted=engine._snapshot("source")
        appended=port.source.append(restored,delta)
        for name,value in appended._read_tensors().items():
            torch.testing.assert_close(value,uninterrupted[name],rtol=1e-5,atol=1e-5)
        report["checks"]["native_save_restore_append"]=True
        del uninterrupted,appended,restored

        # Also check the target's larger recurrent-head layout against its own
        # live cache before installing any translated state.
        native_ids=engine.tokenize("target","<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n<think>\n")
        native_state=engine.capture("target",native_ids)
        native_logits=engine._logits("target").copy()
        native,_=engine._model("target")
        engine.binding.check(engine.lib.lp_prefill(native,array,len(delta)))
        live_target=engine._snapshot("target")
        endpoint=engine.begin_decode("target",native_state,len(native_ids))
        import numpy as np
        np.testing.assert_allclose(endpoint.logits(),native_logits,rtol=1e-4,atol=5e-4)
        endpoint.close()
        copied_target=engine.advance("target",native_state,len(native_ids),delta)
        for name,value in live_target.items():
            torch.testing.assert_close(value,copied_target[name],rtol=1e-5,atol=1e-5)
        report["checks"]["native_target_restore_append"]=True
        del native_state,live_target,copied_target,endpoint

        before=engine.trace("target")
        # Verify that omitting unused parent K/V projections leaves every
        # released bridge output bit-for-bit unchanged on actual reader state.
        parent=engine.bridge._loaded[1]
        parent.base._emit_kv=True
        try:
            unpruned=port.bridge.translate(source)
        finally:
            parent.base._emit_kv=False
        translated=port.bridge.translate(source)
        for name,value in translated._read_tensors().items():
            assert torch.equal(value,unpruned._read_tensors()[name]),f"K/V pruning changed {name}"
        del unpruned
        report["checks"]["unused_parent_kv_pruning_exact"]=True
        assert translated.info.tensor_count==129 and translated.target_prompt_tokens==0
        plain=port.target.session(translated).generate(max_new_tokens=24)
        speculative=port.target.session(translated).generate(max_new_tokens=24,mtp=True)
        after=engine.trace("target")
        assert after["prefill_tokens"]==before["prefill_tokens"],"target processed the prompt"
        assert plain.token_ids==speculative.token_ids,"MTP differs from greedy native target"
        assert speculative.state.clock==translated.clock+len(speculative.token_ids)
        report["checks"]["zero_target_prompt"]=True
        report["checks"]["translated_mtp_matches_greedy"]=True
        report["ordinary_stats"]=asdict(plain.stats)
        report["mtp_stats"]=asdict(speculative.stats)

        # Proposal overrides force both rollback outcomes. The target graph,
        # recurrent checkpoint machinery and acceptance test are unchanged.
        if len(plain.token_ids)<6:
            raise AssertionError("Prompt ended too early to cover forced MTP branches; choose a longer-answer prompt")
        force=engine.lib.lp_test_force_drafts
        force.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int32),ct.c_int];force.restype=ct.c_int
        for reject in (False,True):
            handle=engine.begin_decode("target",translated._read_tensors(),translated.clock,mtp=True)
            try:
                handle.commit(plain.token_ids[0])
                native,info=engine._model("target")
                n=min(args.mtp_drafts,info.mtp_heads if info.mtp_heads>1 else args.mtp_drafts,len(plain.token_ids)-3)
                proposals=list(plain.token_ids[2:2+n])
                if reject:proposals[0]=(proposals[0]+1)%info.vocabulary
                forced=(ct.c_int32*len(proposals))(*proposals)
                engine.binding.check(force(native,forced,len(proposals)))
                committed=handle.greedy_batch(n+1,engine.eos_ids("target"))
                expected=plain.token_ids[1:2] if reject else plain.token_ids[1:2+n]
                assert committed==expected,"forced speculative boundary was committed incorrectly"
                snapshot=handle.snapshot()
                # Continue the rolled-back snapshot through ordinary decoding.
                probe=engine.begin_decode("target",snapshot,handle.clock)
                try:
                    next_id=int(probe.logits()[0].argmax())
                    assert next_id==plain.token_ids[1+len(committed)],"rollback endpoint logits differ"
                    probe.commit(next_id)
                    checked=probe.snapshot()
                    assert len(checked)==129
                finally:probe.close()
                report["checks"]["forced_rejection" if reject else "forced_acceptance"]=True
            finally:handle.close()

        # The public convenience layer and live one-way escalation.
        answer=port.composite.session(policy="strict").ask("Explain cache invalidation in detail.",max_new_tokens=12,mtp=True)
        assert answer.state.target_prompt_tokens==0
        session=port.source.session();session.prepare("Explain ten cache invalidation strategies.")
        small=session.generate(max_new_tokens=2)
        if small.finished:raise AssertionError("Source ended before the live escalation boundary")
        moved=session.escalate(target=port.target,bridge=port.bridge)
        assert moved.target_prompt_tokens==0
        continuation=session.generate(max_new_tokens=8,mtp=True)
        assert continuation.state.clock>moved.clock
        report["checks"]["composite_and_escalation"]=True

        # Two real conversation turns, with the first deliberately split
        # across generate calls to exercise continuation metadata.
        chat=port.composite.session(policy="strict")
        target_before=engine.trace("target")["prefill_tokens"]
        first=chat.ask("Remember this code word: orchid. Reply only OK.",max_new_tokens=8)
        for _ in range(2):
            if first.finished:break
            first=chat.generate(max_new_tokens=192)
        assert first.finished,"First chat turn did not reach EOS within the bounded check"
        reader_prefix=chat.reader.state.rendered_token_ids
        committed=port.target.codec.decode(first.state.generated_token_ids,special=True)
        expected=reader_prefix+port.source.codec.encode(committed)
        second=chat.ask("What is the code word? Reply with only that word.",max_new_tokens=192)
        second_turn_stats=second.stats
        if not second.finished:second=chat.generate(max_new_tokens=192)
        assert second.finished,"Second chat turn did not reach EOS within the bounded check"
        assert chat.reader.state.rendered_token_ids[:len(expected)]==expected
        assert engine.trace("target")["prefill_tokens"]==target_before
        assert second.state.target_prompt_tokens==0
        report["checks"]["two_turn_composite_exact_reader_sync"]=True
        report["chat"]={"first":port.target.codec.decode(first.state.generated_token_ids),
                        "second":port.target.codec.decode(second.state.generated_token_ids),
                        "second_stats":asdict(second_turn_stats)}
        visible=port.target.codec.decode(second.state.generated_token_ids,special=True).split("</think>")[-1]
        report["chat"]["expected_word_recalled"]="orchid" in visible.lower()
        # A persistent native session already owns its prior history. Prepare
        # the exact same committed conversation outside the second-turn timer.
        native_history=replace(first.state,origin="native_capture",target_prompt_tokens=first.state.clock,
                               _components=port.target.backend.capture(port.target.identity,first.state.rendered_token_ids))
        native_second=port.target.session(native_history).ask(
            "What is the code word? Reply with only that word.",max_new_tokens=16)
        report["chat"]["native_persistent_second_stats"]=asdict(native_second.stats)
        report["chat"]["second_turn_ttft_speedup"]=(native_second.stats.request_ttft_s/
                                                    second_turn_stats.request_ttft_s)
        del native_history,native_second
        del chat,first,second

        shared=port.bridge.translate(port.source.capture("Shared fact: caches retain earlier results."))
        left,right=shared.fork(2)
        baseline={k:v.clone() for k,v in shared._read_tensors().items()}
        a=port.target.session(left).ask("Explain a cache hit.",max_new_tokens=8)
        b=port.target.session(right).ask("Explain cache eviction.",max_new_tokens=8)
        assert a.state.target_prompt_tokens>0 and b.state.target_prompt_tokens>0
        for name,value in shared._read_tensors().items():
            assert torch.equal(value,baseline[name]),f"branch mutated shared {name}"
        report["checks"]["independent_branches"]=True
        # Drop diagnostic snapshots before measuring longer contexts.
        del source,translated,plain,speculative,answer,session,small,moved,continuation
        del shared,left,right,baseline,a,b,handle,probe,snapshot,checked
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        report["latency_samples"]=[]
        for count in args.performance_contexts:
            text=("A cache entry has a key, value, expiry time and version. "*(count//10+1))
            text=port.source.codec.decode(port.source.codec.encode(text)[:count],special=True)
            question=text+"\nExplain three cache failure modes."
            row={"requested_context_tokens":count}
            for name in ("native","composite"):
                session=port.target.session() if name=="native" else port.composite.session(policy="strict")
                emitted=[]
                started=perf_counter()
                result=session.ask(question,max_new_tokens=16,on_token=lambda _:emitted.append(perf_counter()))
                row[name]={"observed_ttft_s":emitted[0]-started,
                           "observed_total_s":perf_counter()-started,"stats":asdict(result.stats),
                           "input_tokens":result.state.clock-len(result.token_ids),
                           "tokens":len(result.token_ids)}
                del result,session
                gc.collect()
                torch.cuda.empty_cache()
            row["ttft_speedup"]=row["native"]["observed_ttft_s"]/row["composite"]["observed_ttft_s"]
            row["gpu_free_bytes"]=int(torch.cuda.mem_get_info(0)[0])
            report["latency_samples"].append(row)
        report["residency_after"]=engine.residency()
        report["success"]=True
    except BaseException as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output/"result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        if port is not None and port.source.backend._engine is not None:
            port.source.backend._engine.close()
        print(output/"result.json")


if __name__=="__main__":main()
