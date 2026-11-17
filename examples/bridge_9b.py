"""Answer a question after transferring the evidence prefix to Qwen3.5-9B."""
import latentport as lp

REPO = "latentport/LPBridge-Qwen3.5-4b-Qwen3.5-9b"
SYSTEM = ("Use supplied evidence as authoritative. Later accepted corrections replace only "
          "what they explicitly change. Complete the final request exactly.")


def answer(port, evidence, question):
    history = [("system", SYSTEM), ("user", "Evidence:\n" + evidence)]
    state = port.source.capture(history)
    transferred = port.bridge.translate(state)
    with port.target.session(transferred) as session:
        return session.ask(question, max_new_tokens=16).text.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nf4", "bf16", "gguf"), default="nf4")
    args = parser.parse_args()
    with lp.load(REPO, backend=args.backend, thinking=False) as port:
        print(answer(port, "The ticket is red.", "What color is the ticket? Reply with one word: red or blue."))
