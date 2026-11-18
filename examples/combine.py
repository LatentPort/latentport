"""Combine two independent contexts and query the resulting state."""
import argparse
from pathlib import Path

import latentport as lp


parser = argparse.ArgumentParser()
parser.add_argument("left", type=Path)
parser.add_argument("right", type=Path)
parser.add_argument("question")
parser.add_argument("--backend", choices=("bf16", "nf4", "gguf"), default="nf4")
parser.add_argument("--model-path")
parser.add_argument("--method", choices=("auto", "tokens", "cache"), default="auto")
args = parser.parse_args()

with lp.load_model("latentport/LPCombine-Qwen3.8-27b", backend=args.backend,
                   model_path=args.model_path) as model:
    left = model.prefill(args.left.read_text(encoding="utf-8") + "\n")
    right = model.prefill(args.right.read_text(encoding="utf-8"))
    merged = left.combine(right, model=model, method=args.method)
    with model.session(merged) as session:
        print(session.ask(args.question).text)
