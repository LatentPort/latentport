# Telepathy for LLMs

A local browser demo: 4B reads a broken checkout repository, transfers its full
hybrid state to 27B, and forks the completed diagnosis into repair and regression
specialists. The model supplies a patch and test cases. The harness executes those
cases against the original and repaired code before showing success.

## Run on the validated local installation

From the LatentPort runtime repository in PowerShell:

```powershell
$env:LATENTPORT_HOME = 'F:\LatentPort\runtime\launch-clean-home'
$env:HF_HOME = "$env:LATENTPORT_HOME\hf-home"
& 'F:\LatentPort\runtime\launch-clean-venv\Scripts\python.exe' examples/memory_transplant/server.py
```

The browser opens at **http://127.0.0.1:8768**. Wait for **READY TO BEGIN**, start
recording, and click **Run native 27B**. The presentation uses four fixed-height
chapters, with no page scrolling:

1. **Telepathy:** see what crosses the model boundary: internal state.
2. **Transfer:** compare native 27B with 4B reading and transferring state to 27B.
3. **Branch:** inspect the repair proposal, then run the independent test engineer.
4. **Result:** see measured speedup, zero original-prompt processing by 27B, and two branches; executed tests supply supporting proof.

The run pauses at each key reveal. Use **Try LatentPort**, **Fork this context**,
**Run test engineer**, then **Apply & test** to advance. Earlier chapter tabs let you
review completed steps; **Return to live step** returns to the current stage.
Generated text can scroll inside its own panel; **Follow latest tokens** resumes
following the end of a stream. No page auto-scroll or focus mode is needed.

**Streaming subtitles** appear without a heading, aligned to the left. Each caption
streams word by word from actual server events and remains visible for four seconds
after it finishes. Rapid internal stages are described in one complete sentence so
subtitles keep pace with the live demo. Presenter pauses keep the last caption on
screen until you advance. Reduced-motion preference displays complete captions.

### LinkedIn recording layout

The primary composition is portrait **1080 x 1350 (4:5)**, with larger recording
text. The interface also fits 1080 x 1920 and phone-sized viewports. Use a portrait
browser viewport; if your display is too short, use the browser's responsive/device
view at 1080 x 1350 with Fit to window, and crop your recording to that viewport.
The full demo, subtitle band and primary action stay within the viewport. The
fullscreen button remains available for appropriately sized screens.

The appearance follows latentport.com's actual logo, navy palette, blue gradients,
background grid and type hierarchy. Ctrl+C stops the server after the run. Restart
the server after updating these files so the new presenter checkpoints take effect.

On another configured installation, activate its Python environment and run
`python examples/memory_transplant/server.py`. All large artifacts go below
`LATENTPORT_HOME/demos/transplant-...`. The server binds only to localhost and makes
no cloud inference calls. First-time model downloads follow the normal runtime setup.

## The story

1. Two headphones at $59.95 should cost $119.90. The gateway receives $1.19.
2. Native 27B reads the complete repository and task; its first-token time is recorded.
3. 4B reads exactly the same prompt. The bridge translates H/K/V/C/R and 27B generates.
4. The finished diagnosis is forked into two independent sessions. One proposes the
   single-line repair; the other specifies regression cases for USD, JPY and KWD.
5. The generated cases fail against the original code. After applying the generated
   patch, the harness runs the complete test suite. The checkout turns green only on success.

The repository contains meaningful, runnable code and tests, rather than repeated
padding. Its initial prompt is 10,513 tokens with the qualified tokenizer. Source
files are inspectable through **Explore the repository**. Generated reasoning streams
as it arrives; once an answer completes, its final response remains visible. The full
committed stream remains in the report.

## Recording script

"Telepathy for LLMs. One model reads the prompt. LatentPort translates its internal
state, and a different model inherits that context.

"First, the normal route: the large model processes the full prompt. Now watch
LatentPort. The small model reads it. The bridge translates its working state.
The large model starts generating with zero original-prompt tokens processed.

"Now fork that inherited context. Two independent sessions start from the same
prepared state. Each gets its own new instruction. The original history stays in state.

"Here is the measured first-token speedup, the verified zero prompt replay, and
the two independent branches. Their output also passes executable tests.
Transfer context across models. Fork it. Continue. Telepathy for LLMs."

The checkout is the demonstration workload, not the product story. The hook refers
to cross-model state transfer. It does not mean token-free inference: the reader
processes tokens, the target generates tokens, and branches accept new instructions.
The zero-token counter specifically measures original-prompt processing by the target.

## Evidence and execution boundaries

Each run writes the exact prompt, original and repaired repositories, native logs,
and `results.json`. The report contains model identities, raw generated token IDs,
text, timing, prefill counters, patch, test cases, subprocess output and checks.

- Native and composite run sequentially on the same GPU, after one full-context
  warmup each. Both start fresh, with both models resident. Bars share a time scale.
- TTFT is measured from request start to the first committed token callback,
  including source prefill and state translation for the composite. Model loading
  and warmups are excluded. It is not full-answer speed or time to useful prose.
- Native emits a 192-token preview. Composite continues until its real turn boundary,
  allowing new branch questions. The first 192-token generation calls use the same
  cap; subsequent continuation does not change TTFT.
- Source prefill must equal the complete prompt. Composite target prefill must be
  zero. Each specialist may process its own new instruction, but not replay history.
- Branches execute sequentially. They share the completed diagnosis and inherited
  repository context. There is one source capture and translation in the measured workflow.
- The demo harness turns the model's JSON test specifications into executable unittest
  assertions and validates expected amounts independently. It does not pretend the
  model emitted the unittest boilerplate.
- The model's repair must be a single assignment calling the existing conversion
  helper. Arbitrary generated Python is never executed. Changes occur only in the
  disposable checkout under the run directory. This is a focused demonstration,
  not a general coding-agent benchmark.
- Errors remain errors: the app does not substitute a canned patch, test or response.

## Controls

```powershell
# Run the real inference and verification once, without a browser:
python examples/memory_transplant/server.py --headless

# Inspect the interface without loading GPU models; Run remains disabled:
python examples/memory_transplant/server.py --ui-only

# Choose a different local port / suppress opening a browser:
python examples/memory_transplant/server.py --port 8770 --no-browser
```

The default generation budget is 3072 tokens per completed answer. Set
`--max-tokens 4096` if a run needs a larger budget. A capped incomplete answer stops
the demo rather than manufacturing a completed turn. Restart the server for a fresh run.
