"""Per-kernel GPU time of one Triton decoder step (runtime.TritonDecoderBackend), from torch.profiler.

    python scripts/profile_triton_decoder.py [--cases 40x8 160x32] [--gemm triton|cublas]
"""

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from whisper_trt import Whisper, hub
from whisper_trt.modeling import load_model
from whisper_trt.runtime import TritonDecoderBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["40x8", "160x32"])
    ap.add_argument("--gemm", default="triton", choices=("triton", "cublas"))
    args = ap.parse_args()
    trt = Whisper.from_pretrained()
    cfg = trt.cfg
    model = load_model(hub.resolve("openai/whisper-large-v3"), cfg, dtype=torch.float16)
    del model.encoder
    tri = TritonDecoderBackend(model, trt.backend, gemm=args.gemm)
    L, H, dk, V, C = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size, 228
    for case in args.cases:
        n, w = map(int, case.split("x"))
        cross = trt.cross(w)
        cross.normal_()
        cache = torch.randn(L, 2, n, H, C, dk, device="cuda").half()
        src = torch.arange(n, dtype=torch.int32, device="cuda")[:, None].repeat(1, C)
        tok = torch.full((n, 1), 1000, dtype=torch.long, device="cuda")
        off = torch.full((n,), 80, dtype=torch.long, device="cuda")
        logits = torch.zeros(n, 1, V, device="cuda")
        new_kv = torch.zeros(L, 2, n, H, 1, dk, dtype=torch.float16, device="cuda")
        fn = tri.step_fn(tok, off, cache, logits, new_kv, src, cross=cross)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
        by = defaultdict(lambda: [0.0, 0])
        for e in prof.events():
            if e.device_type.name == "CUDA":
                by[e.name][0] += e.device_time_total / 10
                by[e.name][1] += 1
        total = sum(v[0] for v in by.values())
        print(f"rows {n} windows {w}: kernels {total / 1000:.3f} ms per step", flush=True)
        for name, (us, cnt) in sorted(by.items(), key=lambda x: -x[1][0])[:12]:
            print(f"   {us / 1000:.3f} ms  {cnt // 10:4d}x  {name[:90]}", flush=True)
        # the autotuner's choices
        from whisper_trt import decoder_kernels as dk_

        for kern in (dk_._ln_linear_kernel, dk_._linear_residual_kernel):
            print("  ", kern.fn.__name__, {k: str(v) for k, v in kern.cache.items()}, flush=True)


if __name__ == "__main__":
    main()
