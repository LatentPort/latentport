"""Post-E120 GPU smoke test. Run only when the GPU is free.

This validates package plumbing and target token ownership. It is not a
model-quality evaluation or an efficiency benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4

import latentport as lp
from latentport.hub import bulk_home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "gguf"), default="reference")
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--target-path", type=Path)
    parser.add_argument("--bridge-gguf-path", type=Path)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--output-root", type=Path, default=bulk_home() / "smoke")
    args = parser.parse_args()

    port = lp.load(backend=args.backend, source_path=args.source_path,
                   target_path=args.target_path, bridge_gguf_path=args.bridge_gguf_path)
    residency = port.warm(mtp=args.mtp)
    output = args.output_root.resolve() / uuid4().hex
    output.mkdir(parents=True, exist_ok=False)

    source = port.source.capture((("user", "Explain why a cache can speed up repeated reads."),),
                                 generation_prompt=True)
    if source.info.tensor_count != 65:
        raise AssertionError("Source did not capture all 65 fields")
    snapshot = source.save(output / "source.lpstate")
    restored = lp.State.load(snapshot)
    if restored.identity != source.identity or restored.clock != source.clock:
        raise AssertionError("Source snapshot identity or clock changed")
    extended = port.source.append(restored, " Additional detail")
    if extended.clock <= restored.clock:
        raise AssertionError("Restored source state could not be extended")

    target = port.bridge.translate(source)
    if target.info.tensor_count != 129 or target.target_prompt_tokens != 0:
        raise AssertionError("Target state is incomplete or prompt was replayed")
    emitted = []
    trace = []
    hook = None
    if args.backend == "reference":
        model = port.target.backend._model(port.target.identity)
        hook = model.model.embed_tokens.register_forward_pre_hook(
            lambda _, params: trace.extend(int(x) for row in params[0].detach().cpu().tolist() for x in row))
    try:
        answer = port.target.session(target).generate(max_new_tokens=16, mtp=args.mtp,
                                                      on_token=emitted.append)
    finally:
        if hook is not None:
            hook.remove()
    if tuple(emitted) != answer.token_ids or answer.state.clock != target.clock + len(emitted):
        raise AssertionError("Committed target tokens or clock changed")
    if args.backend == "reference" and trace != emitted:
        raise AssertionError("Target embedded historical prompt or omitted generated tokens")

    small = port.source.session()
    small.prepare("Reason through three ways a cache can become stale, then give a mitigation.")
    small.generate(max_new_tokens=8)
    moved = small.escalate(target=port.target, bridge=port.bridge)
    if moved.target_prompt_tokens != 0 or moved.info.tensor_count != 129:
        raise AssertionError("Escalation violated the complete zero-prompt handoff")
    continued = small.generate(max_new_tokens=16, mtp=args.mtp)

    context = port.source.capture((("user", "Shared fact: a cache stores previously fetched data."),))
    shared = port.bridge.translate(context)
    left, right = shared.fork(2)
    a = port.target.session(left).ask("What is a cache hit?", max_new_tokens=8, mtp=args.mtp)
    b = port.target.session(right).ask("What is cache eviction?", max_new_tokens=8, mtp=args.mtp)
    if not a.state.target_prompt_tokens or not b.state.target_prompt_tokens:
        raise AssertionError("Target branch question accounting is missing")

    report = {
        "backend": args.backend,
        "residency": residency,
        "source_fields": source.info.tensor_count,
        "target_fields": target.info.tensor_count,
        "target_prompt_tokens": target.target_prompt_tokens,
        "reference_target_embedding_trace_matches_generated": trace == emitted if args.backend == "reference" else None,
        "composite_first_answer": {"tokens": len(answer.token_ids), "stats": asdict(answer.stats)},
        "escalation": {"target_clock": moved.clock, "continued_tokens": len(continued.token_ids),
                       "handoff_seconds": small.last_handoff_s, "stats": asdict(continued.stats)},
        "branches": {"left_target_prompt_tokens": a.state.target_prompt_tokens,
                     "right_target_prompt_tokens": b.state.target_prompt_tokens},
        "status": "PASS",
    }
    (output / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output / "RESULT.json")


if __name__ == "__main__":
    main()
