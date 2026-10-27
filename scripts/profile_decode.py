"""Where the time goes: features, encoder, prefill + decode steps, per batch of windows (CUDA events).

    python scripts/profile_decode.py WAVES.npz --lang en --beam 5 --batch 8
"""

import argparse
import json
import time

import numpy as np
import torch

from whisper_trt import Whisper


def build_options(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        out[k] = v.lower() == "true" if v.lower() in ("true", "false") else int(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--build", nargs="*", default=[], help="engine build options KEY=VALUE (build.BuildOptions)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = [torch.from_numpy(data[f"w{i}"]) for i in range(len(meta))]
    audio_s = sum(len(w) for w in waves) / 16000
    model = Whisper.from_pretrained(args.model, **build_options(args.build))
    prompt = model.prompt(args.lang)
    totals = {"features": 0.0, "encoder": 0.0, "decode": 0.0}
    steps = 0
    for rep in range(2):
        totals = dict.fromkeys(totals, 0.0)
        steps, tokens = 0, 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(waves), args.batch):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            with torch.cuda.stream(model.stream):
                ev[0].record()
                feats = model.features(waves[i : i + args.batch])
                ev[1].record()
                cross = model.encode(feats)
                ev[2].record()
                res = model.generate(cross, prompt, beam_size=args.beam)
                ev[3].record()
            torch.cuda.synchronize()
            for k, (a, b) in zip(totals, zip(ev, ev[1:])):
                totals[k] += a.elapsed_time(b) / 1000
            steps += max(len(r.tokens) for r in res) + 1
            tokens += sum(len(r.tokens) for r in res)
        wall = time.perf_counter() - t0
    n_batches = -(-len(waves) // args.batch)
    print(f"{len(waves)} windows, {audio_s:.0f} s, beam {args.beam}, batch {args.batch}: wall {wall:.3f} s "
          f"(RTFx {audio_s / wall:.0f})")
    for k, v in totals.items():
        print(f"  {k}: {v:.3f} s ({v / n_batches * 1000:.1f} ms per batch)")
    print(f"  decode: ~{steps} steps over {n_batches} batches -> {totals['decode'] / max(1, steps) * 1000:.2f} ms per "
          f"step (upper bound, prefill included); {tokens} tokens")
    print(f"  host/other: {wall - sum(totals.values()):.3f} s")


if __name__ == "__main__":
    main()
