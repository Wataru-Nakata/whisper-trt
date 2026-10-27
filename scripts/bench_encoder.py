"""The TensorRT encoder's time per window by batch size (features -> cross-attention keys/values; CUDA events,
the model's stream), and the log-mel features' time for comparison.

    python scripts/bench_encoder.py [--batches 1 2 4 8 16 32]
"""

import argparse

import torch

from whisper_trt import Whisper
from whisper_trt.config import N_FRAMES
from whisper_trt.features import log_mel


def timed(fn, reps: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    args = ap.parse_args()
    model = Whisper.from_pretrained(decoder="triton")
    with torch.cuda.stream(model.stream):
        for n in args.batches:
            feats = torch.randn(n, model.cfg.n_mels, N_FRAMES, device="cuda")
            cross = model.cross(n)
            ms = timed(lambda: model.backend.encode(feats, cross))
            waves = [torch.randn(16000 * 30) * 0.1 for _ in range(n)]
            fm = timed(lambda: log_mel(waves, model.cfg.n_mels, "cuda", N_FRAMES))
            print(f"batch {n}: encoder {ms:.2f} ms ({ms / n:.2f} ms a window), features {fm:.2f} ms "
                  f"({fm / n:.2f} ms a window, host waves)", flush=True)


if __name__ == "__main__":
    main()
