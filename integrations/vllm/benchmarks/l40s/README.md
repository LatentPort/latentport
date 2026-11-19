# LPCompact on vLLM 0.34.0 on an NVIDIA L40S

All runs used one L40S and the pinned Qwen3.5-9B BF16 checkpoint at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The LPCompact selector was
`latentport/LPCompact-Qwen3.5-9B` at
`dae5691996e1810f62466a355f3b23cec9c39dad`. The launcher used synchronous eager
execution, BF16 KV with prefix caching disabled, the Triton GDN prefill
backend, 2,048 batched tokens and a 135,168-token logical context limit.

## Paired answer quality

| Prompt tokens | Mode | Exact answers | Median first token | Median complete response |
| ---: | --- | ---: | ---: | ---: |
| 65,555 | Dense | 16/16 | 6.60 s | 6.81 s |
| 65,555 | 4× | 16/16 | 6.77 s | 6.96 s |
| 65,555 | 8× | 16/16 | 6.80 s | 7.00 s |
| 131,091 | Dense | 16/16 | 17.01 s | 17.22 s |
| 131,091 | 4× | 16/16 | 13.80 s | 13.98 s |
| 131,091 | 8× | 16/16 | 13.47 s | 13.67 s |

Both compacted arms matched all 32 dense answers. The 4× arm reclaimed
85 pages per event and the 8× arm reclaimed 99. All sixteen 131,091-token
cases compacted twice in each compacted arm.

Each serving mode saw the same 32 seeded prompts through
`/v1/completions`. Each length had four cases of repository lookup, latest
decision, structured record lookup and two-hop lookup. Answer scoring is
greedy exact match after stripping outer whitespace. The case files contain
the answer, response, token count, prompt hash, first-token time and
end-to-end time. This synthetic corpus follows the task families from the
[standalone evaluation](../../../../docs/lpcompact-qwen35-validation.json).
See the [dense](paired-dense.jsonl), [4×](paired-4x.jsonl), and
[8×](paired-8x.jsonl) records.

The dense arm set the compaction trigger to 262,144 rows, beyond the
configured logical context. Both compacted arms used a 65,536-row trigger.
All arms served one active request at a time with GPU memory utilization
set to 0.91. Each arm did a short warmup. The 8× arm also ran the
65,555-token image smoke test before the timed requests.

Image digest for this run
`sha256:938085fa90fdb16c6ee57c857aea6d721c6c6c70faed6a0ef72a89fbd8a3db0d`.

The reproducible client is [paired_benchmark.py](../../tests/paired_benchmark.py).
Run it with the cached pinned tokenizer directory, one of
`--mode dense`, `--mode 4x` or `--mode 8x`, and the server URL. The defaults
reproduce the lengths, seed and number of cases.

## Resident-session capacity and throughput

Both arms used the same 10 GiB preallocated KV pool
(`--kv-cache-memory-bytes 10737418240`). The dense arm used the out-of-range
trigger and the compact arm used 8× at 65,536 rows. Each session was primed
with 131,091 tokens and then sent short suffixes. For matched concurrency,
both arms primed the same two prompts. At higher concurrency, the compact
arm also primed two more prompts. Each phase submitted three rounds of
simultaneous turns and measured end-to-end latency to the complete response.

KV pool occupancy comes from the vLLM metric after warming the sessions.
Peak GPU memory comes from 250 ms samples over each server run. Throughput
is the sum of completed turns or generated tokens over the measured round
times, and p95 uses the nearest rank. Scheduler resume records show the
historical rows already processed and the newly appended rows, including
one edge token for each resumed turn.

The reproducible client is
[session_benchmark.py](../../tests/session_benchmark.py). Set the same
`VLLM_API_KEY` for the client and server, pass the cached pinned tokenizer
directory and use `--matched-count 2` for both runs. Set
`--lpcompact-max-sessions` on the server and `--session-count` in the client
to two for dense and four for 8×.
