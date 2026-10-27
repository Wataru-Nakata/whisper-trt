"""The PyTorch fp16 encoder (cuBLAS GEMMs, flash attention; modeling.Precision as exported) captured in a CUDA
graph against the TensorRT encoder: time per window by batch size, and each one's cross-attention keys/values
against the fp32 PyTorch encoder's (real audio).

    python scripts/bench_encoder_torch.py WAVES.npz [--batches 1 4 8]
"""

import argparse
import json

import numpy as np
import torch

from whisper_trt import Whisper, hub
from whisper_trt.modeling import load_model
from whisper_trt.runtime import TorchBackend


def graph_ms(fn, reps: int = 10) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
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
    ap.add_argument("waves")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = []
    for i in range(min(len(meta), max(args.batches))):
        w = data[f"w{i}"]
        waves.append(torch.from_numpy(w.astype(np.float32) / 32768 if w.dtype == np.int16 else w))
    trt = Whisper.from_pretrained(decoder="triton")
    cfg = trt.cfg
    path = hub.resolve("openai/whisper-large-v3")
    m16 = load_model(path, cfg, device="cpu", dtype=torch.float16)
    m32 = load_model(path, cfg, device="cpu", dtype=torch.float32)
    for m in (m16, m32):
        m.encoder.cuda()
        for layer in m.decoder.layers:
            layer.encoder_attn.cuda()
    t16, t32 = TorchBackend(m16), TorchBackend(m32)
    with torch.cuda.stream(trt.stream):
        for n in args.batches:
            feats = trt.features(waves[:n])
            c_trt = trt.cross(n)
            c16 = torch.empty_like(c_trt)
            c32 = torch.empty(c_trt.shape, device="cuda")
            ms_trt = graph_ms(lambda: trt.backend.encode(feats, c_trt))
            ms_torch = graph_ms(lambda: t16.encode(feats, c16))
            t32.encode(feats, c32)
            torch.cuda.synchronize()
            e_trt = (c_trt.float() - c32).abs()
            e16 = (c16.float() - c32).abs()
            print(f"batch {n}: tensorrt {ms_trt / n:.2f} ms a window (error vs fp32: mean {e_trt.mean():.2e}, max "
                  f"{e_trt.max():.3g}); torch fp16 {ms_torch / n:.2f} ms a window (mean {e16.mean():.2e}, max "
                  f"{e16.max():.3g})", flush=True)


if __name__ == "__main__":
    main()
