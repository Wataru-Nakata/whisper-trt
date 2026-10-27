"""Which host lines of a streamed decode synchronize with the GPU (torch.cuda.set_sync_debug_mode): each
distinct call stack (whisper_trt frames) that ran a synchronizing CUDA operation, with its count.

    python scripts/sync_check.py WAVES.npz --lang ja [--windows 40] [--batch 8]
"""

import argparse
import json
import traceback
import warnings
from collections import Counter

import numpy as np
import torch

from whisper_trt import Whisper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = []
    for i in range(min(len(meta), args.windows)):
        w = data[f"w{i}"]
        waves.append(torch.from_numpy(w.astype(np.float32) / 32768 if w.dtype == np.int16 else w))
    model = Whisper.from_pretrained()
    prompt = model.prompt(args.lang)
    model.decode(waves, prompt, beam_size=5, batch_size=args.batch)  # warm-up / capture
    seen: Counter = Counter()

    def show(message, category, filename, lineno, file=None, line=None):
        stack = [f for f in traceback.extract_stack()[:-1] if "whisper_trt" in f.filename and "sync_check" not in
                 f.filename]
        seen[" <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno} {f.line}" for f in reversed(stack[-4:]))] += 1

    warnings.showwarning = show
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    model.decode(waves, prompt, beam_size=5, batch_size=args.batch)
    torch.cuda.set_sync_debug_mode("default")
    for where, n in seen.most_common():
        print(f"{n:5d}x  {where}", flush=True)


if __name__ == "__main__":
    main()
