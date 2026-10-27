"""Where a streamed decode (Whisper.decode) spends its time: decoder step replays (GPU, CUDA events), window
loads (encoder copies, prefills, state resets) and the host's checks between them; for each schedule of the
host's checks (Search.stream's check_every, ahead), whose results must be the same.

    python scripts/profile_stream.py WAVES.npz --lang ja --batch 8 16 32 [--schedules 4,0 1,2]
"""

import argparse
import functools
import json
import time

import numpy as np
import torch

from whisper_trt import Whisper
from whisper_trt.search import Search


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--batch", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--schedules", nargs="+", default=["1,1"])
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = []
    for i in range(len(meta)):
        w = data[f"w{i}"]
        waves.append(torch.from_numpy(w.astype(np.float32) / 32768 if w.dtype == np.int16 else w))
    audio_s = sum(len(w) for w in waves) / 16000
    model = Whisper.from_pretrained()
    prompt = model.prompt(args.lang)
    for batch in args.batch:
        model.decode(waves[: batch * 2], prompt, beam_size=args.beam, batch_size=batch)  # warm-up / capture
        search = model._search("beam" if args.beam > 1 else "greedy", batch, args.beam, len(prompt))
        first = None
        for sched in args.schedules:
            every, ahead = map(int, sched.split(","))
            search.stream = functools.partial(Search.stream, search, check_every=every, ahead=ahead)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = model.decode(waves, prompt, beam_size=args.beam, batch_size=batch)
            torch.cuda.synchronize()
            plain = time.perf_counter() - t0
            tokens = [r.tokens for r in res]
            first = first or tokens
            same = sum(a == b for a, b in zip(tokens, first))
            stats = {"steps": 0, "step_events": [], "start_events": [], "start_host": 0.0}
            run_step, start = search._run_step, search._start

            def timed_step():
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                run_step()
                b.record()
                stats["steps"] += 1
                stats["step_events"].append((a, b))

            def timed_start(*a, **k):
                t0 = time.perf_counter()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                out = start(*a, **k)
                e1.record()
                stats["start_events"].append((e0, e1))
                stats["start_host"] += time.perf_counter() - t0
                return out

            search._run_step, search._start = timed_step, timed_start
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.decode(waves, prompt, beam_size=args.beam, batch_size=batch)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            search._run_step, search._start = run_step, start
            del search.stream
            steps = sum(a.elapsed_time(b) for a, b in stats["step_events"]) / 1000
            starts = sum(a.elapsed_time(b) for a, b in stats["start_events"]) / 1000
            n = stats["steps"]
            print(f"batch {batch} check every {every} read {ahead} later: wall {plain:.2f} s (RTFx "
                  f"{audio_s / plain:.0f}), same results {same}/{len(res)}; timed: wall {wall:.2f} s, {n} steps, "
                  f"GPU time in steps {steps:.2f} s ({steps / n * 1000:.2f} ms each, {steps / wall:.0%} of the "
                  f"wall); window loads {starts:.2f} s GPU span / {stats['start_host']:.2f} s host; rest "
                  f"{wall - steps - starts:.2f} s", flush=True)

if __name__ == "__main__":
    main()
