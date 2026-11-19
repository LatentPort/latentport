"""Measure warm resident session capacity and concurrent turn throughput."""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from time import perf_counter, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from transformers import AutoTokenizer

from paired_benchmark import samples

SESSIONS = "/v1/latentport/sessions"


def call(url, path, key, body=None, *, method=None, expect=200):
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if body is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(body).encode()
    request = Request(url + path, data=body, method=method, headers=headers)
    try:
        with urlopen(request, timeout=900) as response:
            assert response.status == expect, response.status
            return None if expect == 204 else json.load(response)
    except HTTPError as error:
        if error.code != expect:
            raise RuntimeError(
                f"{error.code} {request.get_method()} {error.read().decode()}"
            ) from error
        return json.load(error)


def metrics(url):
    with urlopen(url + "/metrics", timeout=30) as response:
        lines = response.read().decode()

    def read(name):
        values = re.findall(rf"^vllm:{name}\{{[^}}]*\}} ([^ \n]+)", lines, re.MULTILINE)
        assert len(values) == 1, (name, values)
        return float(values[0])

    return {
        "kv_cache_fraction": read("kv_cache_usage_perc"),
        "preemptions": int(read("num_preemptions_total")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--session-count", type=int, required=True)
    parser.add_argument("--matched-count", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 0 < args.matched_count <= args.session_count <= 8
    key = os.environ["VLLM_API_KEY"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    question = (
        "<|im_start|>user\nRepeat the reference code from your last reply four "
        "times, separated by spaces. Give only the code repetitions."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    def emit(stream, **data):
        print(json.dumps(data), file=stream, flush=True)

    call(args.url, SESSIONS, None, {}, expect=401)
    initial = metrics(args.url)
    records = iter(
        item for item in samples(tokenizer, (131091,), 8, 21961)
        if item[0]["task"] == "latest_decision"
    )
    active = []

    def append_turn(session, prompt, *, max_tokens):
        session_id = session["id"]
        start = perf_counter()
        result = call(args.url, f"{SESSIONS}/{session_id}/turns", key, {
            "turn": session["turn"],
            "token_ids": prompt,
            "max_tokens": max_tokens,
        })
        elapsed = perf_counter() - start
        assert result["turn"] == session["turn"]
        assert result["logical_tokens"] == (
            session["logical_tokens"] + len(prompt) + len(result["token_ids"])
        ), result
        session.update({
            "turn": session["turn"] + 1,
            "last_token_id": result["token_ids"][-1],
            "logical_tokens": result["logical_tokens"],
        })
        return result, elapsed

    def warm_one(output):
        record, prompt = next(records)
        created = call(args.url, SESSIONS, key, {}, expect=201)
        session = {
            "id": created["id"], "turn": 1, "logical_tokens": 0,
            "case": record["case"], "answer": record["answer"],
        }
        active.append(session)
        result, elapsed = append_turn(session, prompt, max_tokens=16)
        status = call(args.url, f"{SESSIONS}/{session['id']}", key)
        assert status["next_turn"] == 2 and status["last_turn"] == result
        call(args.url, f"{SESSIONS}/{session['id']}/turns", key, {
            "turn": 1, "token_ids": [1],
        }, expect=409)
        emit(output, kind="warm", case=record["case"],
             prompt_sha256=record["prompt_sha256"], prompt_tokens=len(prompt),
             elapsed_s=elapsed, exact=result["text"].strip() == session["answer"],
             response=result["text"])

    def rounds(output, sessions, phase):
        for run in range(args.rounds):
            start_gate = Barrier(len(sessions) + 1)

            def next_turn(session):
                prefix = "\n" if session["last_token_id"] == im_end else "<|im_end|>\n"
                suffix = tokenizer.encode(prefix + question, add_special_tokens=False)
                start_gate.wait(timeout=30)
                result, elapsed = append_turn(session, suffix, max_tokens=32)
                return {
                    "case": session["case"],
                    "suffix_tokens": len(suffix),
                    "edge_tokens": 1,
                    "output_tokens": len(result["token_ids"]),
                    "elapsed_s": elapsed,
                    "response": result["text"],
                    "code_matches": result["text"].count(session["answer"]),
                }

            with ThreadPoolExecutor(max_workers=len(sessions)) as pool:
                futures = [pool.submit(next_turn, session) for session in sessions]
                start = perf_counter()
                start_gate.wait(timeout=30)
                turns = [future.result() for future in futures]
                elapsed = perf_counter() - start
            emit(output, kind="round", phase=phase, round=run + 1,
                 concurrency=len(sessions), elapsed_s=elapsed,
                 turns_per_s=len(turns) / elapsed,
                 output_tokens_per_s=sum(turn["output_tokens"] for turn in turns) / elapsed,
                 turns=turns)

    with args.output.open("w", encoding="utf-8") as output:
        emit(output, kind="initial", **initial)
        try:
            for _ in range(args.matched_count):
                warm_one(output)
            emit(output, kind="occupancy", phase="matched", sessions=len(active), **metrics(args.url))
            rounds(output, active, "matched")
            if args.matched_count < args.session_count:
                for _ in range(args.session_count - args.matched_count):
                    warm_one(output)
                emit(output, kind="occupancy", phase="capacity", sessions=len(active),
                     **metrics(args.url))
                rounds(output, active, "capacity")
            call(args.url, SESSIONS, key, {}, expect=429)
            final = metrics(args.url)
            emit(output, kind="capacity_limit", sessions=len(active), **final)
            assert final["preemptions"] == initial["preemptions"], final
        finally:
            for session in active:
                call(args.url, f"{SESSIONS}/{session['id']}", key, method="DELETE", expect=204)
            sleep(0.5)
            emit(output, kind="released", **metrics(args.url))
        reopened = call(args.url, SESSIONS, key, {}, expect=201)
        call(args.url, f"{SESSIONS}/{reopened['id']}", key, method="DELETE", expect=204)


if __name__ == "__main__":
    main()
