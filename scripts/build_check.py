"""Engine builds from an empty cache: encoder only (decoder="triton"), then the decoder engine added to the same
directory (decoder="tensorrt"); build times, and both models' tokens on a few windows.

    python scripts/build_check.py WAVES.npz --lang ja [--windows 16]
"""

import argparse
import json
import tempfile
import time

import numpy as np
import torch

from whisper_trt import Whisper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--windows", type=int, default=16)
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = []
    for i in range(min(len(meta), args.windows)):
        w = data[f"w{i}"]
        waves.append(torch.from_numpy(w.astype(np.float32) / 32768 if w.dtype == np.int16 else w))
    with tempfile.TemporaryDirectory() as cache:
        out = {}
        for dec in ("triton", "tensorrt"):
            t0 = time.perf_counter()
            model = Whisper.from_pretrained(decoder=dec, cache_dir=cache)
            print(f"decoder {dec}: ready in {time.perf_counter() - t0:.0f} s", flush=True)
            res = model.decode(waves, model.prompt(args.lang), beam_size=5, batch_size=8)
            out[dec] = [r.tokens for r in res]
            del model
            torch.cuda.empty_cache()
        same = sum(a == b for a, b in zip(out["triton"], out["tensorrt"]))
        print(f"same tokens, triton vs tensorrt decoder: {same}/{len(waves)}", flush=True)


if __name__ == "__main__":
    main()
