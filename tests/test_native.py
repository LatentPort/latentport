"""CPU checks for native file format, ownership and speculative session plumbing."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import torch

from latentport import NativeOptions
from latentport._native.gguf_reader import BridgeGGUF
from latentport.errors import CompatibilityError, StateError
from latentport.composite import Composite
from test_runtime import pair, FakeHandle


def gguf_bytes(*, kind=0, offset=0, truncate=0):
    def string(value):
        data=value.encode();return struct.pack("<Q",len(data))+data
    data=b"GGUF"+struct.pack("<IQQ",3,1,1)
    data+=string("general.architecture")+struct.pack("<I",8)+string("latentport_bridge_v1")
    data+=string("parent.matrix")+struct.pack("<IQQIQ",2,3,2,kind,offset)
    data+=bytes((-len(data))%32)
    data+=np.arange(6,dtype="<f4").tobytes()
    return data[:-truncate] if truncate else data


class NativeTests(unittest.TestCase):
    def test_f32_gguf_dimensions_and_values(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"bridge.gguf";path.write_bytes(gguf_bytes())
            reader=BridgeGGUF(path)
            np.testing.assert_array_equal(reader.tensors["parent.matrix"],np.arange(6).reshape(2,3))
            # mmap mode=c makes accidental writes private, never edits the artifact.
            reader.tensors["parent.matrix"][0,0]=100
            self.assertEqual(path.read_bytes(),gguf_bytes())
            del reader

    def test_gguf_rejects_unsupported_quantization_truncation_and_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"bridge.gguf"
            for kwargs in ({"kind":2},{"truncate":1},{"offset":1},{"offset":4096}):
                with self.subTest(**kwargs):
                    path.write_bytes(gguf_bytes(**kwargs))
                    with self.assertRaises(CompatibilityError):BridgeGGUF(path)

    def test_native_options_bound_memory_expansion(self):
        for kwargs in ({"context_size":0},{"batch_size":8192},{"mtp_drafts":9},{"threads":0}):
            with self.assertRaises(ValueError):NativeOptions(**kwargs)

    def test_mtp_batches_commit_once_and_stream_only_verified_tokens(self):
        class Speculative(FakeHandle):
            mtp=True
            closed=False
            calls=0
            def greedy_batch(self,limit,eos):
                self.calls+=1
                self.clock+=2
                return (66,33)
            def stats(self):
                return {"mtp_proposed_tokens":3,"mtp_accepted_tokens":1,"mtp_warmup_tokens":1,"mtp_batches":1}
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as d:
            source,target,bridge,backend=pair(Path(d))
            backend.supports_mtp=True
            state=bridge.translate(source.capture((("user","x"),),generation_prompt=True))
            handle=Speculative(state.clock)
            backend.begin_decode=lambda *a,**k:handle
            emitted=[]
            answer=target.session(state).generate(max_new_tokens=4,mtp=True,on_token=emitted.append)
            self.assertEqual(answer.token_ids,(65,66,33))
            self.assertEqual(emitted,[65,66,33])
            self.assertEqual(answer.state.clock,state.clock+3)
            self.assertEqual(handle.calls,1)
            self.assertTrue(handle.closed)
            self.assertTrue(answer.stats.mtp_used)
            self.assertEqual(answer.stats.mtp_proposed_tokens,3)
            self.assertEqual(answer.stats.target_prompt_tokens,0)

    def test_failed_mtp_chunk_preserves_public_state_and_closes_native_handle(self):
        class Broken(FakeHandle):
            mtp=True
            closed=False
            def greedy_batch(self,limit,eos):raise StateError("forced rollback failure")
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as d:
            source,target,bridge,backend=pair(Path(d));backend.supports_mtp=True
            state=bridge.translate(source.capture((("user","x"),),generation_prompt=True))
            handle=Broken(state.clock);backend.begin_decode=lambda *a,**k:handle
            session=target.session(state)
            with self.assertRaises(StateError):session.generate(max_new_tokens=4,mtp=True)
            self.assertIs(session.state,state)
            self.assertTrue(handle.closed)

    def test_one_token_request_does_not_claim_speculation(self):
        class Warmup(FakeHandle):
            mtp=True
            def stats(self):return {"mtp_batches":0,"mtp_warmup_tokens":1}
        with tempfile.TemporaryDirectory() as d:
            source,target,bridge,backend=pair(Path(d));backend.supports_mtp=True
            state=bridge.translate(source.capture((("user","x"),),generation_prompt=True))
            backend.begin_decode=lambda *a,**k:Warmup(state.clock)
            result=target.session(state).generate(max_new_tokens=1,mtp=True)
            self.assertFalse(result.stats.mtp_used)
            self.assertTrue(result.stats.mtp_requested)

    def test_multiturn_ttft_counts_reader_synchronization_once(self):
        with tempfile.TemporaryDirectory() as d:
            source,target,bridge,backend=pair(Path(d))
            state=bridge.translate(source.capture((("user","x"),),generation_prompt=True))
            result=target.session(state).generate(max_new_tokens=1)
            result=replace(result,stats=replace(result.stats,first_token_s=1.0))
            chat=Composite(source,target,bridge).session()
            with patch.object(chat,"_sync",return_value=2.0), \
                 patch("latentport.session.Session.generate",return_value=result), \
                 patch("latentport.composite.perf_counter",side_effect=[0.0,5.0,8.0,14.0]):
                answer=chat.ask("Question")
            self.assertEqual(answer.stats.reader_sync_s,2.0)
            self.assertEqual(answer.stats.reader_prefill_s,3.0)
            self.assertEqual(answer.stats.translation_s,3.0)
            self.assertEqual(answer.stats.request_ttft_s,9.0)

    def test_static_composite_uses_the_same_zero_prompt_policy(self):
        with tempfile.TemporaryDirectory() as d:
            source,target,bridge,backend=pair(Path(d))
            answer=Composite(source,target,bridge).run((("user","Question"),),max_new_tokens=2)
            self.assertEqual(answer.token_ids,(65,33))
            self.assertEqual(answer.stats.target_prompt_tokens,0)

    def test_bridge_accepts_readonly_inference_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            source,target,bridge,backend=pair(Path(d))
            with torch.inference_mode():
                state=source.capture((("user","x"),),generation_prompt=True)
            translated=bridge.translate(state)
            self.assertGreater(translated.clock,0)
            self.assertEqual(translated.target_prompt_tokens,0)


if __name__=="__main__":unittest.main()
