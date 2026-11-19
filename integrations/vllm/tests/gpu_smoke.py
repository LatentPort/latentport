"""Exercise live compaction and continuation through vLLM's HTTP server."""

import argparse
import concurrent.futures
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen


def complete(base_url, model, rows, output, skip):
    prompt = [1000 + (i + skip) % 997 for i in range(rows)]
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": output,
            "ignore_eos": True,
        }
    ).encode()
    request = Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=1800) as response:
        result = json.load(response)
    usage = result["usage"]
    assert usage["prompt_tokens"] == rows, usage
    assert usage["completion_tokens"] == output, usage
    assert result["choices"][0]["finish_reason"] == "length", result
    return usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18881")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--rows", type=int, default=3073)
    parser.add_argument("--output", type=int, default=16)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--expected-compactions", type=int, default=1)
    parser.add_argument("--with-short-request", action="store_true")
    args = parser.parse_args()
    offset = args.log.stat().st_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        long = executor.submit(
            complete, args.url, args.model, args.rows, args.output, 0
        )
        if args.with_short_request:
            short = executor.submit(complete, args.url, args.model, 127, 8, 13)
            print("short", short.result())
        print("long", long.result())
    with args.log.open("rb") as log:
        log.seek(offset)
        events = re.findall(
            rb"LPCompact artifact=.*? logical=(\d+) rows=(\d+)->(\d+) "
            rb"pages_reclaimed=(\d+)",
            log.read(),
        )
    changes = [tuple(map(int, event)) for event in events]
    assert len(changes) >= args.expected_compactions, changes
    assert all(
        logical >= before > after and pages > 0
        for logical, before, after, pages in changes
    ), changes
    print("logical, rows before, rows after, reclaimed pages", changes)


if __name__ == "__main__":
    main()
