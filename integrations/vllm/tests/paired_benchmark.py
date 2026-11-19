"""Compare answer quality with identical token inputs across vLLM serving modes."""

import argparse
import hashlib
import json
import random
from array import array
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from transformers import AutoTokenizer

TASKS = ("repository", "latest_decision", "records", "two_hop")


def workload(task, rng, index):
    key = f"PROJECT-{rng.randrange(10_000, 99_999)}"
    code = f"REF-{rng.randrange(100_000, 999_999)}"
    point = (0.05, 0.24, 0.55, 0.82)[index % 4]
    if task == "repository":
        facts = [(point, f"File adapters/{key}.py registers function {code}.")]
        question = f"Which function does adapters/{key}.py register?"
    elif task == "latest_decision":
        facts = [
            (point * 0.73, f"January 4: {key} previously chose REF-000001."),
            (0.91, f"February 7: {key} replaced that choice with {code}."),
        ]
        question = f"What is the most recent choice for {key}?"
    elif task == "records":
        facts = [(point, json.dumps({"item": key, "receipt": code, "state": "accepted"}))]
        question = f"What is the receipt for item {key}?"
    else:
        region = f"ZONE-{rng.randrange(10_000, 99_999)}"
        facts = [
            (point * 0.82, f"Routing map: {key} is assigned to {region}."),
            (0.92, f"Zone directory: the access code for {region} is {code}."),
        ]
        question = f"What is the access code for the zone assigned to {key}?"
    return facts, question, code


def samples(tokenizer, lengths, per_task, seed):
    rng = random.Random(seed)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    marker = "LATENTPORT_ARCHIVE_SLOT"
    template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer only with the requested code."},
            {"role": "user", "content": f"Use the archive below.\n{marker}\n"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    head, tail = template.split(marker)
    head_ids = encode(head)
    filler = encode("".join(
        f"Routine report {i}: item normal. Region {rng.randrange(1000)} "
        f"closed on schedule. Archive checksum {rng.randrange(100_000)}.\n"
        for i in range(256)
    ))
    for length in lengths:
        for task in TASKS:
            for index in range(per_task):
                facts, question, answer = workload(task, rng, index)
                tail_ids = encode(tail.replace(
                    "<|im_end|>", question + "<|im_end|>", 1
                ))
                records = [(point, encode(f"\n{fact}\n")) for point, fact in facts]
                remaining = length - len(head_ids) - len(tail_ids) - sum(
                    len(ids) for _, ids in records
                )
                if remaining <= 0:
                    raise ValueError("Prompt length is too small for the benchmark")
                background = (filler * ((remaining + len(filler) - 1) // len(filler)))[:remaining]
                prompt = list(head_ids)
                cursor = 0
                for point, fact_ids in records:
                    cut = int(point * remaining)
                    prompt.extend(background[cursor:cut])
                    prompt.extend(fact_ids)
                    cursor = cut
                prompt.extend(background[cursor:])
                prompt.extend(tail_ids)
                assert len(prompt) == length
                yield {
                    "case": f"{length}-{task}-{index}",
                    "task": task,
                    "answer": answer,
                    "length": length,
                    "prompt_sha256": hashlib.sha256(array("I", prompt)).hexdigest(),
                }, prompt


def complete(url, model, prompt):
    request = Request(
        f"{url}/v1/completions",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": 16,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = perf_counter()
    first = None
    pieces = []
    usage = None
    with urlopen(request, timeout=900) as stream:
        for line in stream:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk["choices"] and (piece := chunk["choices"][0]["text"]):
                if first is None:
                    first = perf_counter()
                pieces.append(piece)
    elapsed = perf_counter() - start
    if usage is None or usage["prompt_tokens"] != len(prompt):
        raise RuntimeError(f"Invalid prompt accounting: {usage}")
    return "".join(pieces).strip(), {
        "first_token_s": first - start if first is not None else None,
        "elapsed_s": elapsed,
        "completion_tokens": usage["completion_tokens"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--mode", choices=("dense", "4x", "8x"), required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=(65555, 131091))
    parser.add_argument("--cases-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=291163)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    complete(args.url, args.model, tokenizer.encode("Return OK."))

    with args.output.open("w", encoding="utf-8") as output:
        for record, prompt in samples(tokenizer, args.lengths, args.cases_per_task, args.seed):
            response, timing = complete(args.url, args.model, prompt)
            result = {
                **record,
                **timing,
                "mode": args.mode,
                "response": response,
                "exact": response == record["answer"],
            }
            print(json.dumps(result), file=output, flush=True)


if __name__ == "__main__":
    main()
