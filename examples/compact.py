"""Add a state operation to an ordinary single-model LatentPort session."""
import argparse
from pathlib import Path
import latentport as lp

parser = argparse.ArgumentParser()
parser.add_argument("history", type=Path)
parser.add_argument("question")
parser.add_argument("--backend", choices=("bf16", "nf4", "gguf"), default="nf4")
parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
parser.add_argument("--model-path")
args = parser.parse_args()

with lp.load_model(args.model_id, backend=args.backend, model_path=args.model_path,
                   native_options=(lp.NativeOptions(context_size=132096, mtp_drafts=0)
                                   if args.backend == "gguf" else None)) as model:
    state = model.prefill(args.history.read_text(encoding="utf-8"))
    state = state.compact()
    if state.compaction:
        print(state.compaction)
    print(model.session(state).ask(args.question).text)
