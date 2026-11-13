"""Live, recording-friendly native 27B vs LatentPort TTFT comparison.

Run: python examples/ttft_demo.py
Uses the installed runtime. No training or quality benchmark is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import textwrap
from time import perf_counter


def speedup(pairs):
    """Median of paired native/composite ratios; never select the best run."""
    return statistics.median(p["native"]["observed_ttft_s"] /
                             p["composite"]["observed_ttft_s"] for p in pairs)


@contextmanager
def quiet_native(log_path):
    # Windows ConsoleIO retains OS handles rather than following dup2(). Replace
    # the Python streams as well as the CRT descriptors, and update Win32's
    # standard handles (used by subprocesses and native libraries).
    original = (sys.stdout, sys.stderr)
    for stream in original:
        stream.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    console = os.fdopen(os.dup(saved[0]), "w", buffering=1,
                        encoding="utf-8", errors="replace")

    def sync_windows_handles():
        if os.name == "nt":
            import ctypes
            import msvcrt
            setter = ctypes.windll.kernel32.SetStdHandle
            setter.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
            setter.restype = ctypes.c_int
            for number, fd in ((-11, 1), (-12, 2)):
                if not setter(number & 0xffffffff, msvcrt.get_osfhandle(fd)):
                    raise ctypes.WinError()

    redirected = []
    with log_path.open("w", encoding="utf-8") as log:
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            sync_windows_handles()
            redirected = [os.fdopen(os.dup(fd), "w", buffering=1,
                          encoding="utf-8", errors="replace") for fd in (1, 2)]
            sys.stdout, sys.stderr = redirected
            yield console
        finally:
            for stream in redirected:
                stream.flush()
            for fd, backup in zip((1, 2), saved):
                os.dup2(backup, fd)
                os.close(backup)
            sync_windows_handles()
            if os.name == "nt":
                # The original ConsoleIO objects refer to handles closed by dup2.
                sys.stdout, sys.stderr = [os.fdopen(os.dup(fd), "w", buffering=1,
                    encoding="utf-8", errors="replace") for fd in (1, 2)]
            else:
                sys.stdout, sys.stderr = original
            for stream in redirected:
                stream.close()
            console.close()


class Screen:
    def __init__(self, out):
        self.out = out
        self.live = out.isatty()
        self.height = 0
        self.timer_scale = 10.0
        self.task = "Diagnose the cache incident and suggest fixes."
        if self.live and os.name == "nt":
            import ctypes
            import msvcrt
            mode = ctypes.c_ulong()
            handle = msvcrt.get_osfhandle(out.fileno())
            kernel = ctypes.windll.kernel32
            kernel.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            kernel.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            self.live = bool(kernel.GetConsoleMode(handle, ctypes.byref(mode)) and
                             kernel.SetConsoleMode(handle, mode.value | 4))

    def size(self):
        # stdout points to the log; query the dedicated console handle instead.
        try:
            return os.get_terminal_size(self.out.fileno())
        except (OSError, ValueError, AttributeError):
            return os.terminal_size((100, 35))

    def say(self, text=""):
        self.out.write(text + "\n"); self.out.flush()
        self.height = 0

    def draw(self, lines, *, final=False):
        if not self.live:
            if final: self.say("\n".join(lines))
            return
        # A fixed dashboard keeps both bars visible without scrolling.
        columns, rows = self.size()
        width = max(10, min(110, columns - 2))
        self.out.write("\x1b[H")
        for line in lines[:max(1, rows - 1)]:
            color = "\x1b[0m"
            if "LATENTPORT" in line or "27B prompt tokens: 0" in line:
                color = "\x1b[1;96m"
            elif "NATIVE 27B" in line:
                color = "\x1b[1;93m"
            elif "FIRST TOKEN" in line or "TTFT SPEEDUP" in line:
                color = "\x1b[1;92m"
            self.out.write("\r\x1b[2K" + color + line[:width] + "\x1b[0m\n")
        self.out.write("\x1b[J")
        self.out.flush()
        self.height = len(lines)


def timer_bar(seconds, scale, width=40):
    """Elapsed-time ruler, not a fabricated percentage-complete estimate."""
    if seconds is None:
        return "[" + " " * width + "]   ---.--- s"
    filled = min(width, max(0, int(seconds / scale * width)))
    return "[" + "#" * filled + "-" * (width-filled) + f"]   {seconds:7.3f} s"


def dashboard(screen, heading, name, elapsed, first, other, tokens, *, proof=None, preview=""):
    times = {"native": None, "composite": None}
    times[name] = elapsed
    times["composite" if name == "native" else "native"] = other
    # Both bars always have the SAME scale, including when a run exceeds it.
    scale = max(screen.timer_scale, math.ceil(max(t or 0 for t in times.values())))
    width = max(10, min(46, screen.size().columns - 24))
    text_width = max(20, min(100, screen.size().columns - 10))
    task = textwrap.wrap(screen.task, width=text_width)[:2]
    lines = ["LATENTPORT 4+27B   /   LIVE INFERENCE", "=" * 76, heading,
             *["  TASK: " + task[0] if i == 0 else "        " + line for i, line in enumerate(task)], "",
             "  NATIVE 27B       Context -> 27B reads -> generates",
             "  " + timer_bar(times["native"], scale, width),
             "  LATENTPORT       Context -> 4B reads -> bridge -> 27B generates",
             "  " + timer_bar(times["composite"], scale, width),
             f"  Same 0-{scale:g}s elapsed-time scale | shorter is faster", ""]
    label = "NATIVE 27B" if name == "native" else "LATENTPORT"
    lines += [f"  {label}: FIRST TOKEN in {elapsed:.3f}s | {tokens} tokens generated" if first else
              f"  {label}: processing the context; waiting for first token"]
    if proof is not None:
        lines += [f"  VERIFIED | 4B prompt tokens: {proof['source']:,} | 27B prompt tokens: {proof['target']:,}"]
    else:
        lines += ["  Generating continuation..." if first else
                  "  Reading context and preparing the first token..."]
    if other is not None and first:
        native, composite = times['native'], times['composite']
        lines += [f"  TTFT SPEEDUP: {native/composite:.2f}x | {native-composite:.3f}s less waiting"]
    else:
        lines += ["  Sequential runs. Same prompt. Both models resident."]
    # Render the actual accumulated token stream, preserving newlines and showing
    # its latest lines. No decoding or rendering runs in the measured callback.
    safe = ''.join(c for c in preview if c in '\n\t' or (ord(c) >= 32 and ord(c) != 127))
    wrapped = []
    for paragraph in safe.expandtabs(4).splitlines():
        wrapped.extend(textwrap.wrap(paragraph, width=text_width) or [''])
    available = max(1, min(7, screen.size().lines - len(lines) - 5))
    tail = wrapped[-available:] or ['Waiting for generated tokens...']
    lines += ["", f"  {label} / LIVE MODEL OUTPUT" + (" (latest lines)" if len(wrapped) > available else "")]
    lines += ['  ' + line for line in tail]
    lines += [''] * (available - len(tail))
    lines += ["", "  Read once. Transfer state. Start generating."]
    return lines


def large_number(text):
    font = {
        '0': ('###','# #','# #','# #','###'), '1': (' # ','## ',' # ',' # ','###'),
        '2': ('###','  #','###','#  ','###'), '3': ('###','  #','###','  #','###'),
        '4': ('# #','# #','###','  #','  #'), '5': ('###','#  ','###','  #','###'),
        '6': ('###','#  ','###','# #','###'), '7': ('###','  #','  #','  #','  #'),
        '8': ('###','# #','###','# #','###'), '9': ('###','# #','###','  #','###'),
        '.': ('   ','   ','   ','   ',' # '), 'x': ('   ','# #',' # ','# #','   ')}
    return ['  ' + '  '.join(font[c][row] for c in text) for row in range(5)]


def make_prompt(codec, count, context_file, question):
    if context_file:
        text = context_file.read_text(encoding="utf-8")
        label = f"Custom document: {context_file.name}"
    else:
        entries = []
        for i in range(count // 12 + 1):
            entries.append(f"Incident sample {i+1}: " + (
                "catalog: cached price version 41 served after database advanced to version 42; invalidation event missing. "
                if i % 3 == 0 else
                "checkout: hot key expired; 500 requests simultaneously queried the database; no request coalescing. "
                if i % 3 == 1 else
                "session: cache memory at capacity; keys evicted repeatedly; cache misses and database latency rising. "))
        text = "CACHE INCIDENT LOG (synthetic demonstration data)\n" + "\n".join(entries)
        label = "Synthetic cache incident log: stale prices, request stampede, memory pressure"
    ids = codec.encode(text)[:count]
    if not ids: raise ValueError("Context is empty")
    text = codec.decode(ids, special=True)
    return text + "\n\nTASK: " + question, len(ids), label


def run_arm(port, prompt, name, preview_tokens, screen, heading, other=None):
    import torch
    # Same cleanup before both arms. Never clear a prompt cache for only one arm.
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    session = port.target.session() if name == "native" else port.composite.session()
    engine = port.source.backend._native()
    before = {side: engine.trace(side)["prefill_tokens"] for side in ("source", "target")}
    state = {"started": None, "first": None, "tokens": [], "answer": None, "error": None}
    done = threading.Event()

    def callback(token):
        stamp = perf_counter()  # Measure before any rendering or decoding.
        if state["first"] is None: state["first"] = stamp
        state["tokens"].append(token)

    def work():
        try:
            state["started"] = perf_counter()
            state["answer"] = session.ask(prompt, max_new_tokens=preview_tokens,
                                          mtp=False, on_token=callback)
            state["ended"] = perf_counter()
        except BaseException as exc:
            state["error"] = exc
        finally:
            done.set()

    title = "NATIVE 27B" if name == "native" else "LATENTPORT 4+27B"
    if not screen.live: screen.say(f"{heading}: {title} running...")
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while not done.wait(0.08):
        elapsed = perf_counter() - state["started"] if state["started"] else 0
        first = state["first"]
        observed = first - state["started"] if first else elapsed
        token_snapshot = list(state["tokens"])
        live_text = port.target.codec.decode(token_snapshot, special=True) if token_snapshot else ""
        screen.draw(dashboard(screen, heading, name, observed, bool(first), other,
                              len(token_snapshot), preview=live_text))
    worker.join()
    if state["error"] is not None: raise state["error"]
    answer = state["answer"]
    if state["first"] is None: raise RuntimeError("No first-token callback")
    elapsed = state["first"] - state["started"]
    after = {side: engine.trace(side)["prefill_tokens"] for side in before}
    delta = {side: after[side]-before[side] for side in before}
    side = "target" if name == "native" else "source"
    model = port.target if name == "native" else port.source
    expected = len(model.codec.render((("user", prompt),), generation_prompt=True).token_ids)
    if delta[side] != expected:
        raise RuntimeError(f"Fresh prefill check failed: {name}: {delta} expected {expected}")
    if name == "composite" and (delta["target"] != 0 or answer.state.target_prompt_tokens != 0):
        raise RuntimeError("Composite processed prompt tokens on the target")
    if name == "native" and delta["source"] != 0:
        raise RuntimeError("Native baseline unexpectedly used the source")
    preview = port.target.codec.decode(state["tokens"], special=True)
    result = {"observed_ttft_s": elapsed, "total_s": state["ended"]-state["started"],
              "first_token_id": state["tokens"][0], "tokens": state["tokens"],
              "preview": preview, "finished": answer.finished, "prefill_tokens": delta, "stats": asdict(answer.stats)}
    screen.draw(dashboard(screen, heading, name, elapsed, True, other,
                          len(state["tokens"]), proof=delta, preview=preview), final=True)
    # Do not retain large snapshots between runs.
    state.clear(); del answer, session
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--target-path", type=Path)
    parser.add_argument("--context-file", type=Path, help="Optional UTF-8 document; uses at most --context-tokens")
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--preview-tokens", type=int, default=192,
                        help="Tokens to stream per arm after first-token timing (default: 192)")
    parser.add_argument("--question", default="Identify the three cache failures in the log. Give one concrete fix for each. Be concise.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--yes", action="store_true", help="Skip the pause before measured runs")
    args = parser.parse_args()
    if not 128 <= args.context_tokens <= 16384: parser.error("context-tokens must be 128..16384")
    if not 1 <= args.rounds <= 10: parser.error("rounds must be 1..10")
    if not 1 <= args.preview_tokens <= 1024: parser.error("preview-tokens must be 1..1024")
    if not args.question.strip(): parser.error("question must not be empty")
    import latentport as lp
    from latentport.hub import bulk_home
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or bulk_home()/"demos"/f"ttft-{stamp}-{os.getpid()}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"success": False, "started_utc": stamp, "rounds": [], "warmups": {},
              "method": "Sequential fresh sessions; alternating order; both models resident; greedy/MTP off; external request-to-first-callback timer",
              "aggregation": "median of paired native/composite TTFT ratios",
              "scope": "Warm initial-prompt latency, not cached-prefix hits, first useful answer content, full-answer latency or quality"}
    port = None
    def save():
        (output/"results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with quiet_native(output/"native.log") as console:
        screen = Screen(console)
        try:
            screen.say("LATENTPORT 4+27B | LIVE TIME TO FIRST TOKEN")
            screen.say("Preparing models and a fair, resident native-27B comparison...")
            screen.say(f"Native runtime logs: {output / 'native.log'}")
            port = lp.load(backend="gguf", source_path=args.source_path, target_path=args.target_path)
            report["residency"] = port.warm()
            report["source"] = asdict(port.source.identity)
            report["target"] = asdict(port.target.identity)
            prompt, count, label = make_prompt(port.source.codec, args.context_tokens, args.context_file, args.question)
            screen.task = args.question
            (output/"prompt.txt").write_text(prompt, encoding="utf-8")
            report.update(context_tokens=count, fixture=label, question=args.question,
                          prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            screen.say(f"{count:,} context tokens + question/template | {label}")
            screen.say("Warming BOTH paths on this context length (excluded from results).")
            for name in ("native", "composite"):
                report["warmups"][name] = run_arm(port, prompt, name, 1, screen, "WARMUP / NOT MEASURED")
                save()
            screen.timer_scale = max(1, math.ceil(max(
                r["observed_ttft_s"] for r in report["warmups"].values()) * 1.25))
            screen.draw(["LATENTPORT 4+27B", "=" * 76, "", "  READY TO BEGIN", "",
                         f"  {count:,} context tokens. Two ways to reach the first token.", "",
                         "  NATIVE 27B   Context -> 27B reads -> 27B generates", "",
                         "  LATENTPORT   Context -> 4B reads -> bridge -> 27B generates", "",
                         f"  {args.rounds} paired rounds. Live timers. Measured speedup.",
                         "  Same prompt. Fresh state. Both models already loaded.", "",
                         *textwrap.wrap("CONTEXT: " + label, width=76),
                         *textwrap.wrap("TASK: " + args.question, width=76), "",
                         "  Press Enter to begin.", "",
                         "  Watch the first token arrive. Then follow the answer live."], final=True)
            if not args.yes:
                if not sys.stdin.readline(): raise RuntimeError("No interactive input; use --yes for unattended runs")
            for index in range(args.rounds):
                order = ("native", "composite") if index % 2 == 0 else ("composite", "native")
                pair = {"order": list(order)}
                report["rounds"].append(pair)
                for name in order:
                    other = pair.get(order[0], {}).get("observed_ttft_s") if name == order[1] else None
                    pair[name] = run_arm(port, prompt, name, args.preview_tokens, screen,
                                         f"ROUND {index+1}/{args.rounds} | {count:,}-TOKEN CONTEXT | {order[0].upper()} FIRST", other)
                    save()
                ratio = pair["native"]["observed_ttft_s"] / pair["composite"]["observed_ttft_s"]
                screen.say(f"  ROUND {index+1}: LatentPort TTFT speedup = {ratio:.2f}x\n")
            ratio = speedup(report["rounds"])
            report.update(success=True, median_paired_speedup=ratio,
                          median_native_ttft_s=statistics.median(p["native"]["observed_ttft_s"] for p in report["rounds"]),
                          median_composite_ttft_s=statistics.median(p["composite"]["observed_ttft_s"] for p in report["rounds"]))
            native = report['median_native_ttft_s']
            composite = report['median_composite_ttft_s']
            scale = max(native, composite, 0.001)
            screen.draw(["LATENTPORT 4+27B   /   THE RESULTS", "=" * 76,
                         f"  {len(report['rounds'])} measured pair(s) | {count:,}-token context", "",
                         *large_number(f"{ratio:.2f}x"),
                         "  TTFT SPEEDUP / median of all paired ratios", "",
                         "  NATIVE 27B  " + timer_bar(native, scale, 32),
                         "  LATENTPORT  " + timer_bar(composite, scale, 32), "",
                         "  27B prompt tokens: 0 in every composite run. Verified.",
                         "  4B reads the prompt. The bridge transfers state. 27B generates.", "",
                         "  LESS WAITING. MORE BUILDING.",
                         "  Small-model prefill. Large-model generation."], final=True)
            screen.say(f"Evidence: {output / 'results.json'}")
        except BaseException as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            screen.say(f"DEMO STOPPED: {report['error']}")
            screen.say(f"Details: {output / 'native.log'}")
            raise
        finally:
            save()
            if port is not None: port.close()


if __name__ == "__main__":
    main()
