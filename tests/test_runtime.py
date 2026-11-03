"""CPU contract tests for state ownership and execution policies."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from latentport.bridge import Bridge
from latentport.codec import TokenRecord
from latentport.composite import Composite
from latentport.errors import CapabilityError, StateError
from latentport.identity import ModelIdentity
from latentport.model import Model
from latentport.state import State
from latentport.gguf import GGUFBackend, _NativeDecodeHandle


class FakeCodec:
    chat_eos_id = 33

    def record(self, text):
        ids = tuple(ord(c) for c in text)
        spans = tuple((i, i + 1) for i in range(len(ids)))
        return TokenRecord(text, text, ids, spans)

    def render(self, messages, *, generation_prompt):
        text = "".join(f"<{role}>{content}!" for role, content in messages)
        if generation_prompt:
            text += "<assistant>"
        return self.record(text)

    def encode(self, text):
        return tuple(ord(c) for c in text)

    def decode(self, ids, *, special=False):
        text = "".join(chr(i) for i in ids)
        return text if special else text.replace("!", "")


class FakeHandle:
    def __init__(self, clock):
        self.clock = clock
        self.logits = self._logits(65)

    def _logits(self, token):
        values = np.zeros((1, 256), dtype=np.float32)
        values[0, token] = 1
        return values

    def step(self, token_id):
        self.clock += 1
        self.logits = self._logits(33)

    def snapshot(self):
        return {"H": torch.tensor([self.clock], dtype=torch.bfloat16)}


class FakeBackend:
    name = "fake"
    state_abi = "fake.complete.v1"
    supports_mtp = False
    device = "cpu"

    def __init__(self):
        self.calls = []
        self.fail_target = False

    def codec(self, identity):
        return FakeCodec()

    def capture(self, identity, ids):
        self.calls.append((identity.repo_id, "capture", tuple(ids)))
        return {"H": torch.tensor([len(ids)], dtype=torch.bfloat16)}

    def advance(self, identity, components, clock, ids):
        self.calls.append((identity.repo_id, "advance", tuple(ids)))
        return {"H": torch.tensor([clock + len(ids)], dtype=torch.bfloat16)}

    def begin_decode(self, identity, components, clock, *, mtp=False):
        if self.fail_target and identity.repo_id == "target":
            raise RuntimeError("target allocation failed")
        self.calls.append((identity.repo_id, "begin_decode", ()))
        return FakeHandle(clock)

    def install(self, identity, components, clock):
        self.calls.append((identity.repo_id, "install", ()))

    def first_logits(self, identity, components, clock):
        return FakeHandle(clock).logits

    def eos_ids(self, identity):
        return frozenset({33})


class FakeTranslator:
    def translate(self, components, clock, metadata):
        assert metadata["source_length"] == clock
        return {"H": torch.tensor([metadata["target_length"]], dtype=torch.bfloat16)}


def pair(root):
    manifest = {"schema": "latentport.bridge-bundle.v1", "candidate": "E119_joint_step0016",
                "release_id": "test", "models": {
                    "source": {"repo_id": "source", "revision": "one"},
                    "target": {"repo_id": "target", "revision": "two"}}}
    (root / "latentport.json").write_text(json.dumps(manifest), encoding="utf-8")
    backend = FakeBackend()
    def model(side, rev):
        identity = ModelIdentity(side, rev, "fake", "test", rev, "template", backend.state_abi)
        return Model(identity, backend, role=side)
    source, target = model("source", "one"), model("target", "two")
    bridge = Bridge(bundle_root=root, source=source, target=target, translator=FakeTranslator())
    return source, target, bridge, backend


class RuntimeTests(unittest.TestCase):
    def test_snapshot_roundtrip_fork_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            source, _, _, _ = pair(Path(temp))
            state = source.capture((("user", "Q"),), generation_prompt=True)
            state_a, state_b = state.fork(2)
            self.assertEqual(state_a.parent_id, state.state_id)
            self.assertTrue(state_a.info.shares_storage)
            self.assertEqual(state_b.info.tensor_bytes, state.info.tensor_bytes)
            stored = state.save(Path(temp) / "state")
            loaded = State.load(stored)
            self.assertEqual(loaded.identity, state.identity)
            self.assertEqual(loaded.info.representation, "disk")
            self.assertEqual(int(loaded._tensors()["H"][0]), state.clock)
            with (stored / "tensors.safetensors").open("ab") as stream:
                stream.write(b"broken")
            with self.assertRaises(StateError):
                State.load(stored)

    def test_composite_reads_question_only_on_source_and_syncs(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target, bridge, backend = pair(Path(temp))
            chat = Composite(source, target, bridge).session()
            emitted = []
            first = chat.ask("Q1", max_new_tokens=4, on_token=emitted.append)
            self.assertTrue(first.finished)
            self.assertEqual(tuple(emitted), first.token_ids)
            self.assertEqual(first.state.target_prompt_tokens, 0)
            second = chat.ask("Q2", max_new_tokens=4)
            self.assertTrue(second.finished)
            self.assertEqual(second.state.target_prompt_tokens, 0)
            source_work = [x for x in backend.calls if x[0] == "source"]
            target_work = [x for x in backend.calls if x[0] == "target"]
            self.assertTrue(any(x[1] == "advance" for x in source_work))
            self.assertFalse(any(x[1] in ("capture", "advance") for x in target_work))

    def test_escalation_keeps_source_on_failed_install(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target, bridge, backend = pair(Path(temp))
            session = source.session()
            session.prepare("Q")
            session.generate(max_new_tokens=1)
            old = session.state
            backend.fail_target = True
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                session.escalate(target=target, bridge=bridge)
            self.assertIs(session.model, source)
            self.assertIs(session.state, old)
            backend.fail_target = False
            moved = session.escalate(target=target, bridge=bridge)
            self.assertEqual(moved.phase, "in_answer")
            self.assertEqual(moved.target_prompt_tokens, 0)
            self.assertIs(session.model, target)

    def test_state_cannot_be_installed_on_different_model(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target, _, _ = pair(Path(temp))
            state = source.capture("abc")
            with self.assertRaises(Exception):
                target.session(state)

    def test_target_branch_question_is_counted_as_target_prefill(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target, bridge, backend = pair(Path(temp))
            shared = bridge.translate(source.capture((("user", "Context"),)))
            left, right = shared.fork(2)
            a = target.session(left).ask("A")
            b = target.session(right).ask("B")
            self.assertGreater(a.state.target_prompt_tokens, 0)
            self.assertGreater(b.state.target_prompt_tokens, 0)
            self.assertEqual(shared.target_prompt_tokens, 0)

    def test_raw_context_can_branch_to_different_questions(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target, bridge, _ = pair(Path(temp))
            shared = bridge.translate(source.capture("Shared context"))
            left, right = shared.fork(2)
            a = target.session(left).ask("First question")
            b = target.session(right).ask("Second question")
            self.assertNotEqual(a.state.rendered_token_ids, b.state.rendered_token_ids)
            self.assertEqual(shared.target_prompt_tokens, 0)

    def test_native_mtp_receipt_must_report_rollback(self):
        class Broken:
            def logits(self):
                return np.zeros((1, 256), dtype=np.float32)
            def commit(self, token):
                return {"committed_clock": 6, "rejected_drafts_rolled_back": False}
        handle = _NativeDecodeHandle(Broken(), "target", 5, True)
        with self.assertRaises(StateError):
            handle.step(1)

    def test_gguf_requires_real_native_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("small.gguf", "large.gguf"):
                (root / name).write_bytes(b"GGUF")
            backend = GGUFBackend(root / "small.gguf", root / "large.gguf", root,
                                  {"models": {}}, engine=None)
            with patch("latentport._native.engine.library_path", side_effect=CapabilityError("missing")):
                with self.assertRaises(CapabilityError):
                    backend.require_extension()


if __name__ == "__main__":
    unittest.main()
