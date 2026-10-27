"""Clips of up to 30 s from consecutive utterances of a manifest (like the pipeline's planned clips), as a waves
file for scripts/ref_ct2.py and scripts/parity.py.

    python scripts/make_clips.py MANIFEST OUT.npz --lang ja --clips 64 [--seconds 30] [--gap 0.3]
"""

import argparse
import json

import numpy as np
import torch

from whisper_trt.audio import load_audio

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("out")
ap.add_argument("--lang", required=True, help="Whisper language code of the recordings")
ap.add_argument("--clips", type=int, default=64)
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--gap", type=float, default=0.3, help="silence between joined utterances")
args = ap.parse_args()

items = json.load(open(args.manifest))
gap = torch.zeros(int(args.gap * 16000))
limit = int(args.seconds * 16000)
clips, meta, cur, texts = {}, [], [], []
sep = "" if args.lang in ("ja", "zh", "yue", "th") else " "
for it in items:
    w = load_audio(it["path"])
    if len(w) > limit:
        continue
    n = sum(len(x) for x in cur) + len(gap) * len(cur)
    if cur and n + len(gap) + len(w) > limit:
        clips[f"w{len(meta)}"] = torch.cat([y for x in cur for y in (x, gap)][:-1]).numpy()
        meta.append({"id": f"clip{len(meta)}", "text": sep.join(texts), "lang": args.lang})
        cur, texts = [], []
        if len(meta) == args.clips:
            break
    cur.append(w)
    texts.append(it["text"])
np.savez(args.out, meta=json.dumps(meta, ensure_ascii=False), **clips)
total = sum(len(w) for w in clips.values()) / 16000
print(f"{len(meta)} clips, {total:.1f} s (mean {total / max(1, len(meta)):.1f} s) -> {args.out}")
