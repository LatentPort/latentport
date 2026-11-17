"""Answer a question after transferring the evidence prefix to Qwen3.5-9B."""
import latentport as lp

REPO = "latentport/LPBridge-Qwen3.5-4b-Qwen3.5-9b"
SYSTEM = ("Use supplied evidence as authoritative. Later accepted corrections replace only "
          "what they explicitly change. Complete the final request exactly.")


def answer(port, evidence, question):
    history = [("system", SYSTEM), ("user", "Evidence:\n" + evidence)]
    prefix = port.source.prefill(history)
    request = port.target.codec.render([*history, ("user", question)], generation_prompt=True)
    if request.token_ids[:prefix.clock] != prefix.rendered_token_ids:
        raise ValueError("Source and target disagree on the evidence prefix")
    ready = port.handoff(prefix, target_suffix_ids=request.token_ids[prefix.clock:])
    with port.target.session(ready) as session:
        return session.generate(max_new_tokens=16).text.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nf4", "bf16", "gguf"), default="nf4")
    args = parser.parse_args()
    with lp.load(REPO, backend=args.backend, thinking=False) as port:
        print(answer(port, "The ticket is red.", "What color is the ticket? Reply with one word: red or blue."))
