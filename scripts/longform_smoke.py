"""Long-form transcription on a few recordings of a long-*.npz (Whisper.transcribe): segments and words printed,
the time taken.

    python scripts/longform_smoke.py WAVES.npz [--lang ja] [--n 3] [--word-timestamps]

Without --lang, each recording's language is detected.
"""

import argparse
import json
import time

import numpy as np
import torch

from whisper_trt import Whisper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--lang")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--word-timestamps", action="store_true")
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = [torch.from_numpy(data[f"w{i}"].astype(np.float32) / 32768) for i in range(min(args.n, len(meta)))]
    model = Whisper.from_pretrained()
    t0 = time.perf_counter()
    out = model.transcribe(waves, args.lang, word_timestamps=args.word_timestamps, batch_size=8)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    audio_s = sum(len(w) for w in waves) / 16000
    print(f"{len(waves)} recordings, {audio_s:.0f} s of audio in {sec:.1f} s (RTFx {audio_s / sec:.0f})", flush=True)
    for i, segs in enumerate(out):
        print(f"--- recording {i}: {len(segs)} segments, language {segs.language} ({segs.language_probability:.2f})",
              flush=True)
        for s in segs[:6]:
            print(f"[{s.start:7.2f} -> {s.end:7.2f}] t={s.temperature} {s.text[:80]}", flush=True)
            if s.words:
                print("    " + " ".join(f"{w.word}({w.start:.2f}-{w.end:.2f})" for w in s.words[:8]), flush=True)


if __name__ == "__main__":
    main()
