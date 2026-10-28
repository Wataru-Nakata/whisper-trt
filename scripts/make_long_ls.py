"""Long clean-speech recordings for long-form tests: consecutive LibriSpeech test-clean utterances of a chapter
(.cache/eval/ls-test-clean.npz) joined, with 0.3 s of silence between them, up to --seconds each; with their
reference text.

    python scripts/make_long_ls.py LS.npz OUT.npz [--seconds 240] [--max 20]
"""

import argparse
import json
from itertools import groupby

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--max", type=int, default=20)
    args = ap.parse_args()
    data = np.load(args.src)
    meta = json.loads(str(data["meta"]))
    gap = np.zeros(int(0.3 * 16000), dtype=np.int16)
    waves, out_meta = {}, []
    for chapter, items in groupby(range(len(meta)), key=lambda i: meta[i]["id"].rsplit("-", 1)[0]):
        parts, texts, n = [], [], 0
        for i in items:
            w = data[f"w{i}"]
            if n + len(w) > args.seconds * 16000:
                break
            parts += [w, gap]
            texts.append(meta[i]["text"])
            n += len(w) + len(gap)
        if n < 60 * 16000:
            continue
        waves[f"w{len(out_meta)}"] = np.concatenate(parts).astype(np.int16)
        out_meta.append({"id": chapter, "text": " ".join(texts), "lang": "en"})
        if len(out_meta) >= args.max:
            break
    np.savez(args.out, meta=json.dumps(out_meta), **waves)
    total = sum(len(w) for w in waves.values()) / 16000
    print(f"{len(out_meta)} recordings, {total / 3600:.2f} h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
