# Live state demos

Run these from the repository in the same environment as the TTFT demo. Each loads
both models, pauses at **READY TO BEGIN**, then shows the task, elapsed-time bar,
live committed-token output and verified state operations. Use a terminal at least
100 columns by 35 rows. No extra dependencies are needed beyond the installed runtime.

```powershell
$env:LATENTPORT_HOME = 'F:\LatentPort\runtime\launch-clean-home'
$env:HF_HOME = "$env:LATENTPORT_HOME\hf-home"
$python = 'F:\LatentPort\runtime\launch-clean-venv\Scripts\python.exe'
```

| Demo | Command | Story |
| --- | --- | --- |
| Converse | `& $python examples/state_demo.py cached` | Read the brief once, then ask a follow-up using the retained target state. |
| Branch | `& $python examples/state_demo.py branch` | One capture and translation supply an incident responder and release planner. |
| Resume | `& $python examples/state_demo.py resume` | Save context to disk, release the source snapshot, load it and ask a new question. |
| Extend | `& $python examples/state_demo.py extend` | Append updated telemetry with the 4B, translate the updated state and investigate. |
| Full tour | `& $python examples/state_demo.py all` | All four demos, keeping models loaded between demos. |

The shared story is the Atlas release: ORCHID is its codename, Maya owns the release,
Leon owns rollback, and checkout must achieve p95 below 200 ms. The initial incident
has 410 ms latency. The Extend demo appends new evidence that request coalescing has
reduced p95 to 120 ms and Maya approved the gate. The source text is generated demo
data, saved alongside the report.

## What to say

- **Converse:** "The small model reads the initial context. The large model keeps its
  working state for the follow-up. Watch how few new tokens it needs to process."
- **Branch:** "Prepare the context once. Give two specialists different questions.
  Each receives an independent continuation of the same prepared context."
- **Resume:** "The context now has a lifetime of its own. Save it, restore it, and
  continue without rereading the brief."
- **Extend:** "New information arrives. The small model reads only that update.
  Transfer the updated state and ask the large model what changed."

## Resume in a separate process

The first `resume` run writes `atlas.lpstate` below its evidence directory. Exit the
process, then start a new one pointing to that saved directory:

```powershell
& $python examples/state_demo.py resume --state 'F:\LatentPort\runtime\launch-clean-home\demos\state-YOUR-RUN\atlas.lpstate'
```

This second run loads the existing state and skips capture and save. The default
resume demo demonstrates disk round-trip in one process; this command additionally
demonstrates persistence across processes. Model loading is separate from state-load
and new-question timings.

## Controls and evidence

Default source context is approximately 4K tokens. Use `--context-tokens 512` for a
short functional check or `8192` for a larger context. `--yes` skips Enter pauses. Each generated preview stays on screen until you press
Enter, giving viewers time to read before the next step.
The first cached answer must reach a real end-of-turn boundary before the follow-up;
`--answer-tokens` sets its maximum budget (default 2048). The script stops if it cannot
finish that turn rather than pretending the boundary has been reached. Other answers
are 192-token previews and may show intermediate reasoning rather than a completed answer.

Each run saves `results.json`, `context.txt`, `native.log`, and any persisted state
under `LATENTPORT_HOME/demos/`. The report includes real token IDs and text, step
latencies, source/target prefill deltas, and the assertions that passed. Timings are
resident-runtime observations, not comparative speedup claims. Initial and follow-up
questions differ; their TTFTs are displayed separately, not as a speedup ratio.

Branch execution is sequential on the single GPU. Branches read their own new questions
on the target; the shared history is not replayed. The bridge is applied once for the
branch demo and once after append in Extend. Append does not edit earlier context.
These checks validate runtime mechanics, not answer correctness. Live escalation
remains experimental and is not included in this promotional set.
