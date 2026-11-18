"""Replay a text file through a pinned LPCompact model with bounded attention memory."""
import argparse
from pathlib import Path

import latentport as lp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("text", type=Path, help="UTF-8 text to replay")
    parser.add_argument("--backend", default="bf16")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-logical-positions", type=int, help="Logical context length including continuation")
    parser.add_argument("--capacity", type=int, help="Physical cache limit (defaults to the model limit)")
    parser.add_argument("--chunk-size", type=int, default=16384, help="Maximum input tokens per append")
    parser.add_argument("--output", type=Path, help="Save the final immutable state")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    with lp.load_model(args.model, backend=args.backend, model_path=args.model_path,
                       max_logical_positions=args.max_logical_positions, max_kv_positions=args.capacity) as model:
        ids = model.codec.encode(args.text.read_text(encoding="utf-8"))
        if not 1 <= len(ids) <= model.backend.options.logical_limit:
            parser.error(f"Input has {len(ids)} tokens, outside this model's replay budget")
        with model.live() as session:
            capacity = model.backend.options.physical_limit
            start = 0
            while start < len(ids):
                if session.physical == capacity:
                    session.compact(reserve_tokens=1)
                count = min(args.chunk_size, capacity - session.physical, len(ids) - start)
                session.append(ids[start:start + count])
                start += count
            state = session.snapshot(release=True)
        print(f"Replayed {state.clock:,} tokens with {state.physical_kv_length:,} retained")
        if args.output:
            state.save(args.output)


if __name__ == "__main__":
    main()
