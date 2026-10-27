"""Per-layer times of the TensorRT encoder on one window (TensorRT's layer profiler): the most expensive layers,
and their total by name pattern (numbers dropped).

    python scripts/trt_profile_encoder.py [--batch 1]
"""

import argparse
import re
from collections import defaultdict

import tensorrt as trt
import torch

from whisper_trt import Whisper
from whisper_trt.config import N_FRAMES


class Profiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.times = defaultdict(float)

    def report_layer_time(self, name, ms):
        self.times[name] += ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    model = Whisper.from_pretrained(decoder="triton")
    be = model.backend.encoder if hasattr(model.backend, "encoder") else model.backend
    with torch.cuda.stream(model.stream):
        feats = torch.randn(args.batch, model.cfg.n_mels, N_FRAMES, device="cuda")
        cross = model.cross(args.batch)
        for _ in range(3):
            be.encode(feats, cross)
        torch.cuda.synchronize()
        ctx = next(iter(be._enc_ctxs.values()))
        prof = Profiler()
        ctx.profiler = prof
        reps = 5
        for _ in range(reps):
            be.encode(feats, cross)
            torch.cuda.synchronize()
    total = sum(prof.times.values()) / reps
    print(f"batch {args.batch}: profiled layers {total:.2f} ms, {len(prof.times)} layers", flush=True)
    by = defaultdict(lambda: [0.0, 0])
    for name, ms in prof.times.items():
        key = re.sub(r"\d+", "#", name)[:150]
        by[key][0] += ms / reps
        by[key][1] += 1
    for name, (ms, c) in sorted(by.items(), key=lambda x: -x[1][0])[:25]:
        print(f"   {ms:7.3f} ms ({ms / total:4.0%}) {c:4d}x  {name}", flush=True)


if __name__ == "__main__":
    main()
