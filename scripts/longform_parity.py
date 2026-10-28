"""Long-form transcription (Whisper.transcribe) against faster-whisper's (scripts/ref_long.py) on long-*.npz
recordings: recordings with the same segments (tokens, start, end), the share of segments the same before a
recording's first difference, the error rate of our whole transcripts against faster-whisper's, word timestamps
of the same segments, and the time taken.

    python scripts/longform_parity.py WAVES.npz REF.json --lang ja [--word-timestamps] [--batch 8 32]
        [--backend torch --dtype float32] [--out OURS.json]
"""

import argparse
import json
import time

import numpy as np
import torch

from whisper_trt import Whisper

CER_LANGS = {"ja", "zh", "th", "ko", "yue"}


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("ref")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--word-timestamps", action="store_true")
    ap.add_argument("--batch", nargs="+", type=int, default=[8])
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out")
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = [torch.from_numpy(data[f"w{i}"].astype(np.float32) / 32768) for i in range(len(meta))]
    audio_s = sum(len(w) for w in waves) / 16000
    ref = json.load(open(args.ref))
    model = Whisper.from_pretrained(backend=args.backend, dtype=args.dtype)
    units = (lambda t: list(t.replace(" ", ""))) if args.lang in CER_LANGS else (lambda t: t.split())  # noqa: E731
    out = {}
    for batch in args.batch:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model.transcribe(waves, args.lang, word_timestamps=args.word_timestamps, batch_size=batch)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        rows = [[{"start": s.start, "end": s.end, "text": s.text, "tokens": s.tokens, "temperature": s.temperature,
                  "words": [[w.start, w.end, w.word, w.probability] for w in s.words] if s.words else None}
                 for s in segs] for segs in res]
        out[f"b{batch}"] = rows
        print(f"{args.backend}/{args.dtype} batch {batch}: RTFx {audio_s / sec:.1f} ({sec:.1f} s for {audio_s:.0f} s)",
              flush=True)
        for key, run in ref["runs"].items():
            same_rec, prefix, errs, n_units, word_d, fallback = 0, [], 0, 0, [], 0
            for ours, theirs in zip(rows, run["recordings"]):
                keyed = [(s["tokens"], round(s["start"], 3), round(s["end"], 3)) for s in ours]
                ref_keyed = [(s["tokens"], round(s["start"], 3), round(s["end"], 3)) for s in theirs]
                same_rec += keyed == ref_keyed
                k = next((i for i, (a, b) in enumerate(zip(keyed, ref_keyed)) if a != b), min(len(keyed), len(ref_keyed)))
                prefix.append(k / max(len(ref_keyed), 1))
                a = units(" ".join(s["text"].strip() for s in ours))
                b = units(" ".join(s["text"].strip() for s in theirs))
                errs += edit_distance(a, b)
                n_units += len(b)
                fallback += sum(s["temperature"] > 0 for s in theirs)
                if args.word_timestamps:
                    for s, t in zip(ours[:k], theirs[:k]):
                        for w, v in zip(s["words"] or [], t["words"] or []):
                            word_d.append(max(abs(w[0] - v[0]), abs(w[1] - v[1])))
            line = (f"    vs {key} (RTFx {run['rtfx']:.1f}): recordings with the same segments {same_rec}/{len(rows)}, "
                    f"segments the same before the first difference {np.mean(prefix):.1%}, "
                    f"{'CER' if args.lang in CER_LANGS else 'WER'} of ours against it {errs / max(n_units, 1):.2%}, "
                    f"its fallback segments {fallback}")
            if word_d:
                d = np.array(word_d)
                line += (f"; words of the same segments: {len(d)}, same times {np.mean(d < 1e-6):.1%}, "
                         f"within 0.02 s {np.mean(d <= 0.0200001):.1%}, max diff {d.max():.2f} s")
            print(line, flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
