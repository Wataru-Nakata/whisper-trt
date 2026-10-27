"""The beam bookkeeping in Triton kernels (beam_kernels) against the PyTorch one on real windows: streamed beam 5
decodes with each (WHISPER_TRT_FUSED_BEAM), their tokens and scores compared, and their wall times.

    python scripts/fused_check.py WAVES.npz --lang ja [--batch 8 32]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from whisper_trt import Whisper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--batch", nargs="+", type=int, default=[8, 32])
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
        out = {}
        for fused in ("1", "0", "1", "0"):  # (each twice: the second times count)
            os.environ["WHISPER_TRT_FUSED_BEAM"] = fused
            model._searches.clear()
            model.decode(waves[: 2 * batch], prompt, beam_size=5, batch_size=batch)  # (capture)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = model.decode(waves, prompt, beam_size=5, batch_size=batch)
            torch.cuda.synchronize()
            out[fused] = (time.perf_counter() - t0, res)
        (t1, a), (t0, b) = out["1"], out["0"]
        same = sum(x.tokens == y.tokens for x, y in zip(a, b))
        ds = max(abs(x.hypothesis.normalized - y.hypothesis.normalized) for x, y in zip(a, b))
        dn = max(abs(x.no_speech_prob - y.no_speech_prob) for x, y in zip(a, b))
        print(f"batch {batch}: fused RTFx {audio_s / t1:.0f}, PyTorch RTFx {audio_s / t0:.0f}; same tokens "
              f"{same}/{len(a)}, max |score diff| {ds:.2g}, max |no_speech diff| {dn:.2g}", flush=True)


if __name__ == "__main__":
    main()
