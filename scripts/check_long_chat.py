"""One 16K two-turn memory/latency check, retaining the previous answer."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import torch
import latentport as lp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path",required=True)
    parser.add_argument("--target-path",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu-is-free",action="store_true")
    parser.add_argument("--policy",choices=("strict","cached_handoff"),default="strict")
    args=parser.parse_args()
    if not args.gpu_is_free:parser.error("Confirm GPU availability with --gpu-is-free")
    report={"success":False,"context_tokens":16384,"policy":args.policy}
    port=None
    try:
        port=lp.load(backend="gguf",source_path=args.source_path,target_path=args.target_path,
                     native_options=lp.NativeOptions(context_size=18432,mtp_drafts=4))
        report["resident"]=port.warm(mtp=True)
        codec=port.source.codec
        text="A cache entry has a key, value, expiry time and version. "*1700
        text=codec.decode(codec.encode(text)[:16384],special=True)
        chat=port.composite.session(policy=args.policy)
        engine=port.source.backend._native()
        before=engine.trace("target")["prefill_tokens"]
        torch.cuda.reset_peak_memory_stats()
        first=chat.ask(text+"\nThe code word is orchid. Reply only OK.",max_new_tokens=192)
        for _ in range(2):
            if first.finished:break
            first=chat.generate(max_new_tokens=192)
        assert first.finished,"First turn did not end within cap"
        ids=first.state.rendered_token_ids
        metadata={name:getattr(first.state,name) for name in (
            "clock","phase","rendered_text","rendered_token_ids","messages","generated_token_ids")}
        assert engine.trace("target")["prefill_tokens"]==before
        report["initial_target_prompt_tokens"]=first.state.target_prompt_tokens
        source_before=engine.trace("source")["prefill_tokens"]
        if args.policy=="strict":
            prefix=chat.reader.state.rendered_token_ids
            feedback=codec.encode(port.target.codec.decode(first.state.generated_token_ids,special=True))
        report["free_after_first"]=int(torch.cuda.mem_get_info()[0])
        question="What is the code word? Reply only with that word."
        second=chat.ask(question,max_new_tokens=192)
        report["second_stats"]=asdict(second.stats)
        for _ in range(2):
            if second.finished:break
            second=chat.generate(max_new_tokens=192)
        assert second.finished,"Second turn did not end within cap"
        if args.policy=="strict":
            assert chat.reader.state.rendered_token_ids[:len(prefix)+len(feedback)]==prefix+feedback
            assert engine.trace("target")["prefill_tokens"]==before
            report["zero_target_prompt"]=True
            report["exact_reader_sync"]=True
        else:
            assert chat.reader.state is None
            assert engine.trace("source")["prefill_tokens"]==source_before
            assert second.state.rendered_token_ids[:len(ids)]==ids
            suffix=port.target.codec.render((("user",question),),generation_prompt=True).token_ids
            expected=len(suffix)+len(port.target.codec.encode("\n"))
            assert engine.trace("target")["prefill_tokens"]-before==expected
            assert second.state.target_prompt_tokens==expected
            assert report["second_stats"]["translation_s"]==0
            assert report["second_stats"]["reader_sync_s"]==0
            report["followup_target_prompt_tokens"]=expected
            report["cached_prefix_preserved"]=True
            report["reader_inactive"]=True
        answer=port.target.codec.decode(second.state.generated_token_ids,special=True).split("</think>")[-1]
        report["answer"]=answer
        report["expected_word_recalled"]="orchid" in answer.lower()
        assert report["expected_word_recalled"]
        report["free_after_second"]=int(torch.cuda.mem_get_info()[0])
        report["torch_peak_allocated_bytes"]=int(torch.cuda.max_memory_allocated())
        report["torch_reserved_after_second"]=int(torch.cuda.memory_reserved())
        report["cache_reclaims"]=engine._cache_reclaims
        # Release conversation states before preparing the separate baseline.
        del chat,first,second
        gc.collect();torch.cuda.empty_cache()
        values=port.target.backend.capture(port.target.identity,ids)
        state=lp.State(identity=port.target.identity,origin="native_capture",target_prompt_tokens=len(ids),
                       _components=values,**metadata)
        native=port.target.session(state).ask(question,max_new_tokens=16)
        report["native_second_stats"]=asdict(native.stats)
        report["second_turn_ttft_speedup"]=(native.stats.request_ttft_s/
                                          report["second_stats"]["request_ttft_s"])
        report["success"]=True
    except BaseException as exc:
        report["error"]=str(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        if port is not None:port.close()
        print(args.output,flush=True)


if __name__=="__main__":main()
