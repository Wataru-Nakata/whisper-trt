"""cuBLAS on the encoder's GEMMs (32 layers x QKV, out, fc1, fc2 for 1500 frames a window, fp16, fp32
accumulation), in a CUDA graph: time per window and TFLOP/s, to set against the TensorRT encoder's GEMM time
(scripts/trt_profile_encoder.py).

    python scripts/bench_encoder_gemms.py [--windows 1 4 8]
"""

import argparse

import torch

SHAPES = [(3840, 1280), (1280, 1280), (5120, 1280), (1280, 5120)]  # (N, K)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 4, 8])
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    weights = [[(torch.randn(n, k, device="cuda").half() * 0.02, torch.randn(n, device="cuda").half())
                for n, k in SHAPES] for _ in range(32)]
    for w in args.windows:
        m = 1500 * w
        xs = {k: torch.randn(m, k, device="cuda").half() for k in (1280, 5120)}
        outs = {n: torch.empty(m, n, device="cuda", dtype=torch.float16) for n in (1280, 3840, 5120)}

        def body():
            for layer in weights:
                for (wt, b), (n, k) in zip(layer, SHAPES):
                    torch.addmm(b, xs[k], wt.t(), out=outs[n])

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        g.replay()
        a.record()
        for _ in range(10):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        ms = a.elapsed_time(e) / 10
        flops = 2 * m * sum(n * k for n, k in SHAPES) * 32
        print(f"{w} windows: {ms / w:.2f} ms a window ({flops / ms / 1e9:.0f} TFLOP/s)", flush=True)


if __name__ == "__main__":
    main()
