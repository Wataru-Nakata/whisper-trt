"""Where a streamed decode (Whisper.decode) spends its time: decoder step replays (GPU, CUDA events), window
loads (encoder copies, prefills, state resets) and the host's checks between them.

    python scripts/profile_stream.py WAVES.npz --lang ja --batch 8 16 32
"""

import argparse
import json
import time

import numpy as np
import torch

from whisper_trt import Whisper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--batch", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--beam", type=int, default=5)
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
        steps = sum(a.elapsed_time(b) for a, b in stats["step_events"]) / 1000
        starts = sum(a.elapsed_time(b) for a, b in stats["start_events"]) / 1000
        n = stats["steps"]
        print(f"batch {batch}: wall {wall:.2f} s (RTFx {audio_s / wall:.0f}); {n} steps, GPU time in steps "
              f"{steps:.2f} s ({steps / n * 1000:.2f} ms each, {steps / wall:.0%} of the wall); window loads "
              f"{starts:.2f} s GPU span / {stats['start_host']:.2f} s host; rest {wall - steps - starts:.2f} s",
              flush=True)


if __name__ == "__main__":
    main()
