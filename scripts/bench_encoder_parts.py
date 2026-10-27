"""The parts an encoder outside TensorRT would be made of, for one window (1500 frames) x 32 layers, in CUDA
graphs: self-attention by each scaled_dot_product_attention backend (against TensorRT's fused attention,
scripts/trt_profile_encoder.py), the LayerNorm kernel of decoder_kernels and an fp32 GELU.

    python scripts/bench_encoder_parts.py [--windows 1 4]
"""

import argparse

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from whisper_trt import decoder_kernels as dk


def graph_ms(fn, reps: int = 10) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 4])
    args = ap.parse_args()
    L, H, T, D = 32, 20, 1500, 64
    for w in args.windows:
        q, k, v = (torch.randn(w, H, T, D, device="cuda", dtype=torch.float16) for _ in range(3))
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION),
                              ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
            try:
                with sdpa_kernel(backend):
                    ms = graph_ms(lambda: [F.scaled_dot_product_attention(q, k, v) for _ in range(L)])
                    err = (F.scaled_dot_product_attention(q, k, v).float() - ref).abs().max().item()
                flops = 4 * w * H * T * T * D * L
                print(f"{w} windows: attention {name} {ms / w:.3f} ms a window ({flops / ms / 1e9:.0f} TFLOP/s, "
                      f"max |err| {err:.2g})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{w} windows: attention {name} unavailable: {str(e)[:100]}", flush=True)
        x = torch.randn(w * T, 1280, device="cuda", dtype=torch.float16)
        g, b = torch.ones(1280, device="cuda"), torch.zeros(1280, device="cuda")
        bias = torch.zeros(1280, device="cuda", dtype=torch.float16)
        ms = graph_ms(lambda: [dk.layer_norm_bias(x, g, b, bias) for _ in range(2 * L)])
        print(f"{w} windows: LayerNorm (Triton, {2 * L} calls) {ms / w:.3f} ms a window", flush=True)
        h = torch.randn(w * T, 5120, device="cuda", dtype=torch.float16)
        ms = graph_ms(lambda: [F.gelu(h.float()).half() for _ in range(L)])
        ms2 = graph_ms(lambda: [F.gelu(h) for _ in range(L)])
        print(f"{w} windows: GELU fp32 via casts {ms / w:.3f} ms, F.gelu on fp16 (fp32 math) {ms2 / w:.3f} ms a "
              f"window", flush=True)


if __name__ == "__main__":
    main()
