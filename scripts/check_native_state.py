"""Small real-GPU capture/install check without loading the bridge or target."""
from __future__ import annotations
import argparse
import ctypes as ct
import json
import numpy as np
import torch
from latentport._native import Engine, NativeOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--side", choices=("source", "target"), default="source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = Engine(args.model, args.model, "unused", ".",
                    NativeOptions(context_size=1024, mtp_drafts=0))
    report = {"success": False, "side": args.side}
    try:
        ids = engine.tokenize(args.side, "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n<think>\n")
        state = engine.capture(args.side, ids)
        original = engine._logits(args.side).copy()
        native, _ = engine._model(args.side)
        delta = engine.tokenize(args.side, "Let me think.")
        array = (ct.c_int32 * len(delta))(*delta)
        engine.binding.check(engine.lib.lp_prefill(native, array, len(delta)))
        live = engine._snapshot(args.side)
        handle = engine.begin_decode(args.side, state, len(ids))
        restored = handle.logits()
        report["head_max_logit_error"] = float(np.abs(original-restored).max())
        report["head_same_greedy_token"] = int(original.argmax()) == int(restored.argmax())
        handle.close()
        copied = engine.advance(args.side, state, len(ids), delta)
        report["append_max_errors"] = {
            k: float((live[k].float()-copied[k].float()).abs().max()) for k in live}
        for name in live:
            torch.testing.assert_close(live[name], copied[name], rtol=1e-5, atol=1e-5)
        assert report["head_same_greedy_token"]
        np.testing.assert_allclose(original, restored, rtol=1e-4, atol=5e-4)
        report["success"] = True
    finally:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        engine.close()


if __name__ == "__main__":
    main()
