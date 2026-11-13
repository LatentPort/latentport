# Record the live TTFT demo

Run `examples/ttft_demo.py` with the installed LatentPort runtime. It loads both
models, warms both paths, and pauses at **READY TO BEGIN**. Start recording your
terminal, then press Enter. No model downloads or cold loading occur in the timed
comparison.

## On the validated local installation

In PowerShell:

```powershell
cd 'C:\Users\Simon\OneDrive\Documents\Python Scripts\latentport-package\latentport'
$env:LATENTPORT_HOME = 'F:\LatentPort\runtime\launch-clean-home'
$env:HF_HOME = "$env:LATENTPORT_HOME\hf-home"
$env:HF_XET_CACHE = "$env:LATENTPORT_HOME\xet"
& 'F:\LatentPort\runtime\launch-clean-venv\Scripts\python.exe' examples/ttft_demo.py
```

For another installation, follow [installation](installation.md), activate that
environment, set `LATENTPORT_HOME` to its asset cache, and run:

```powershell
python examples/ttft_demo.py
```

Use Windows Terminal with at least 110 columns and around 35 rows, a readable font,
and no other GPU workloads. The demo presents three paired rounds, two live elapsed-time bars on a shared seconds scale,
actual generated-token previews, and a final median paired speedup. Native backend
logs go to a file so they do not obscure the display. Keep all rounds in the recording.

For a shorter single-pair recording:

```powershell
python examples/ttft_demo.py --rounds 1
```

## What the viewer is seeing

- The same approximately 16K-token prompt processed by native 27B and LatentPort.
- Fresh sessions for each request, with both models already resident on the GPU.
- One full-context warmup per path, excluded from the final result.
- Alternating execution order across measured pairs; greedy decoding, MTP off for both.
- Wall-clock time from immediately before the request to its first token callback.
  This includes source prefill, translation, target-state installation and first-token
  computation for the composite. It excludes setup, model loading and downloads.
- Explicit checks that native 27B processes the full prompt, while the composite
  processes **zero prompt tokens on the 27B**.

The default prompt is a synthetic cache incident log covering stale prices, a
request stampede and memory pressure. Both models receive the same task: identify
the three failures and suggest fixes. This differs from the original repeated-note
timing fixture; use the live measurements rather than its historical timings. The first
emitted token may be a reasoning or control token: this is not time to the first
useful answer or total response speed. The display reports the actual ratio,
including results below 2x; it does not select the fastest run.

Suggested narration: "Both models are already loaded. Each receives the same fresh
16K context. Native 27B reads it itself; LatentPort reads it with 4B and translates
that state so 27B can start generating without processing the prompt. These are
live request-to-first-token measurements."

## Your own document and evidence

```powershell
python examples/ttft_demo.py --context-file F:\LatentPort\demo-document.txt
```

The UTF-8 document is truncated to at most 16,384 source tokens; shorter documents
are not repeated. Use `--question "Your question"` to set a task appropriate to the document. Different content and lengths can produce different timings.

Each run saves `results.json`, the exact `prompt.txt`, and `native.log` beneath
`LATENTPORT_HOME/demos/`. The JSON contains every measured arm, excluded warmups,
model identities, prompt hash, runtime stats and timing aggregation. `--output`
selects a new output directory; `--yes` skips the recording pause for automation.
Do not publish a private document or its saved state/logs without reviewing them.

The dashboard keeps both models visible, freezes each bar at its measured first
token, and ends with a large speedup display. Bars show elapsed seconds, not
estimated completion percentages. The shared scale is set from warmup timings
and expands for both bars together if necessary.

The task stays visible above the bars. Actual generated tokens stream into a
rolling output panel, including any reasoning/control tokens. The default is
192 generated tokens per arm; use `--preview-tokens 512` for a longer continuation.
This is a capped preview, not a guarantee of a complete answer. Full generated text
and token IDs are saved in the report. TTFT freezes at the first token while output
continues streaming. Use a terminal at least 100 columns wide and 35 rows high.
