"""Run a composite conversation: python examples/chat.py --help."""
import argparse
import latentport as lp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Explain how database indexes work.")
    parser.add_argument("--followup")
    parser.add_argument("--source-path")
    parser.add_argument("--target-path")
    parser.add_argument("--policy", choices=("strict", "cached_handoff"), default="cached_handoff")
    parser.add_argument("--mtp", action="store_true")
    args = parser.parse_args()
    with lp.load(backend="gguf", source_path=args.source_path, target_path=args.target_path) as port:
        port.warm(mtp=args.mtp)
        chat = port.composite.session(policy=args.policy)
        for question in filter(None, (args.prompt, args.followup)):
            answer = chat.ask(question, max_new_tokens=256, mtp=args.mtp)
            print(f"TTFT: {answer.stats.request_ttft_s:.3f}s", flush=True)
            for chunk in range(8):
                print(answer.text, end="", flush=True)
                if answer.finished:
                    break
                if chunk == 7:
                    raise RuntimeError("Generation cap reached; resume with chat.generate()")
                answer = chat.generate(max_new_tokens=256, mtp=args.mtp)
            print()


if __name__ == "__main__":
    main()
