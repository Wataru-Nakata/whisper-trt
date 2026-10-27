"""Per-layer times of one TensorRT decoder step (TensorRT's layer profiler), summed by kind of layer.

    python scripts/trt_profile.py WAVES.npz --rows 1 40 --windows 1 8
"""

import argparse
import re
from collections import defaultdict

import numpy as np
import tensorrt as trt
import torch

from whisper_trt import Whisper


class Profiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.times = defaultdict(float)
        self.runs = 0

    def report_layer_time(self, name, ms):
        self.times[name] += ms


def kind(name: str) -> str:
    n = name.lower()
    for key, pat in (("self_attn plugin", r"self_attn"), ("cross_attn plugin", r"cross_attn"),
                     ("logits GEMM", r"embed_tokens|logits|matmul_\d+\b.*vocab"), ("fc1/fc2/proj GEMM", r"gemm|matmul|linear|myelin"),
                     ("layer_norm", r"norm"), ("gelu", r"gelu|erf"), ("cast/copy/reformat", r"cast|copy|reformat|shuffle")):
        if re.search(pat, n):
            return key
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--build", nargs="*", default=[], help="engine build options KEY=VALUE (build.BuildOptions)")
    ap.add_argument("--rows", nargs="+", type=int, default=[1, 40])
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--cache", type=int, default=128)
    ap.add_argument("--offset", type=int, default=60)
    args = ap.parse_args()
    data = np.load(args.waves)
    from profile_decode import build_options

    model = Whisper.from_pretrained(args.model, decoder="tensorrt", **build_options(args.build))
    be, cfg = model.backend, model.cfg
    L, H, dk, V = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size
    with torch.cuda.stream(model.stream):
        for n, w in zip(args.rows, args.windows):
            model.encode(model.features([torch.from_numpy(data[f"w{i}"]) for i in range(w)]))
            C = args.cache
            cache = torch.randn(L, 2, n, H, C, dk, device="cuda").half()
            src = torch.arange(n, dtype=torch.int32, device="cuda")[:, None].repeat(1, C)
            tok = torch.full((n, 1), 1000, dtype=torch.long, device="cuda")
            off = torch.full((n,), args.offset, dtype=torch.long, device="cuda")
            logits = torch.zeros(n, 1, V, device="cuda")
            new_kv = torch.zeros(L, 2, n, H, 1, dk, dtype=torch.float16, device="cuda")
            fn = be.step_fn(tok, off, cache, logits, new_kv, src, cross=model.cross(w))
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            ev[0].record()
            for _ in range(50):
                fn()
            ev[1].record()
            torch.cuda.synchronize()
            wall = ev[0].elapsed_time(ev[1]) / 50
            ctx = fn.keep[0]
            prof = Profiler()
            ctx.profiler = prof
            reps = 20
            for _ in range(reps):
                fn()
                torch.cuda.synchronize()
            by = defaultdict(float)
            for name, ms in prof.times.items():
                by[kind(name)] += ms / reps
            total = sum(by.values())
            print(f"rows {n}, windows {w}: {wall:.3f} ms per call (events); profiled layers sum {total:.3f} ms, "
                  f"{len(prof.times)} layers", flush=True)
            for k, v in sorted(by.items(), key=lambda x: -x[1]):
                print(f"   {k}: {v:.3f} ms ({v / total:.0%})")
                if k in ("cast/copy/reformat", "other", "gelu"):  # what they are: layer-0 names (numbers dropped)
                    names = defaultdict(lambda: [0.0, 0])
                    for name, ms in prof.times.items():
                        if kind(name) == k:
                            key = re.sub(r"_\d+", "", name)[:140]
                            names[key][0] += ms / reps
                            names[key][1] += 1
                    for name, (ms, c) in sorted(names.items(), key=lambda x: -x[1][0])[:6]:
                        print(f"        {ms:.3f} ms {c:3d}x  {name}")
            top = sorted(prof.times.items(), key=lambda x: -x[1])[:12]
            for name, ms in top:
                print(f"     {ms / reps:.4f} ms  {name[:150]}")


if __name__ == "__main__":
    main()
