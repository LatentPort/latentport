"""Run a composite conversation: python examples/chat.py --help."""
import argparse
import latentport as lp
from latentport.hub import DEFAULT_REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Bridge repository or verified local bundle")
    parser.add_argument("--revision", help="Exact bridge revision (uses the pinned default when omitted)")
    parser.add_argument("--backend", choices=("gguf", "bf16", "nf4"), default="gguf")
    parser.add_argument("--context", help="Context to read on the source before asking the question")
    parser.add_argument("--prompt", default="Explain how database indexes work.")
    parser.add_argument("--followup")
    parser.add_argument("--source-path")
    parser.add_argument("--target-path")
    parser.add_argument("--policy", choices=("strict", "cached_handoff"), default="cached_handoff")
    parser.add_argument("--mtp", action="store_true",
                        help="Enable MTP; requires the pinned GGUF profile with a qualified MTP head")
    args = parser.parse_args()
    if args.mtp and args.backend != "gguf":
        parser.error("--mtp requires --backend gguf and a qualified MTP profile")
    with lp.load(args.repo, revision=args.revision, backend=args.backend,
                 source_path=args.source_path, target_path=args.target_path,
                 native_options=(lp.NativeOptions(mtp_drafts=4 if args.mtp else 0)
                                 if args.backend == "gguf" else None)) as port:
        port.warm(mtp=args.mtp)
        context = [("user", args.context)] if args.context is not None else None
        chat = port.composite.session(context, policy=args.policy)
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
