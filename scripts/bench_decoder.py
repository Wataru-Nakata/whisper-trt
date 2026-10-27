"""One decoder step, captured in a CUDA graph: the TensorRT engine against the Triton decoder
(runtime.TritonDecoderBackend: cuBLAS GEMMs; and with Triton GEMMs), each timed twice in turn, at several rows x
windows; and each one's logits against the fp32 PyTorch decoder's.

    python scripts/bench_decoder.py [--cases 1x1 40x8 80x16 160x32]
"""

import argparse

import torch

from whisper_trt import Whisper, hub
from whisper_trt.modeling import load_model
from whisper_trt.runtime import TorchBackend, TritonDecoderBackend


def time_step(fn, reps: int = 100) -> float:
    for _ in range(3):
        fn()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["1x1", "40x8", "80x16", "160x32"])
    ap.add_argument("--build", nargs="*", default=[])
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    from profile_decode import build_options

    trt = Whisper.from_pretrained(decoder="tensorrt", **build_options(args.build))
    cfg = trt.cfg
    model = load_model(hub.resolve("openai/whisper-large-v3"), cfg, dtype=torch.float16)
    del model.encoder
    tri = TritonDecoderBackend(model, trt.backend)
    tri_gemm = TritonDecoderBackend(model, trt.backend, gemm="triton")
    ref32 = load_model(hub.resolve("openai/whisper-large-v3"), cfg, dtype=torch.float32)
    del ref32.encoder
    ref = TorchBackend(ref32)
    L, H, dk, V, C = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size, 228
    for case in args.cases:
        n, w = map(int, case.split("x"))
        cross = trt.cross(w)
        cross.normal_()
        cache = torch.randn(L, 2, n, H, C, dk, device="cuda").half()
        src = torch.arange(n, dtype=torch.int32, device="cuda")[:, None].repeat(1, C)
        tok = torch.full((n, 1), 1000, dtype=torch.long, device="cuda")
        off = torch.full((n,), 60, dtype=torch.long, device="cuda")
        out = {}
        backends = (("tensorrt", trt.backend), ("triton", tri), ("triton GEMMs", tri_gemm))
        for rep in range(2):
            for name, be in backends:
                logits = torch.zeros(n, 1, V, device="cuda")
                new_kv = torch.zeros(L, 2, n, H, 1, dk, dtype=torch.float16, device="cuda")
                c = cache.clone()  # (the Triton decoder writes its new keys/values there)
                fn = be.step_fn(tok, off, c, logits, new_kv, src, cross=cross)
                ms = time_step(fn)
                out[name if rep == 0 else f"{name} again"] = (ms, logits.clone())
        logits = torch.zeros(n, 1, V, device="cuda")
        new_kv = torch.zeros(L, 2, n, H, 1, dk, device="cuda")
        with torch.inference_mode():
            ref.step_fn(tok, off, cache.float(), logits, new_kv, src, cross=cross.float())()
        d = {k_: (v[1] - logits).abs().max().item() for k_, v in out.items()}
        print(f"rows {n} windows {w}: " + ", ".join(f"{k_} {v[0]:.3f} ms (logits max |diff| from fp32 {d[k_]:.3g})"
                                                     for k_, v in out.items()), flush=True)


if __name__ == "__main__":
    main()
