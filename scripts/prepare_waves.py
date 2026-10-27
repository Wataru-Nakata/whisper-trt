"""Decode a manifest's audio once to 16 kHz float32 (so that both implementations see the same samples).

    python scripts/prepare_waves.py MANIFEST OUT.npz [--limit N]
"""

import argparse
import json

import numpy as np

from whisper_trt.audio import load_audio

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("out")
ap.add_argument("--limit", type=int)
ap.add_argument("--int16", action="store_true", help="store 16-bit samples (half the size)")
args = ap.parse_args()
items = json.load(open(args.manifest))[: args.limit]
waves = {f"w{i}": load_audio(it["path"]).numpy() for i, it in enumerate(items)}
if args.int16:
    waves = {k: np.clip(np.round(w * 32768), -32768, 32767).astype(np.int16) for k, w in waves.items()}
meta = [{"id": it["id"], "text": it["text"], "lang": it.get("lang")} for it in items]
np.savez(args.out, meta=json.dumps(meta, ensure_ascii=False), **waves)
print(f"{len(items)} recordings, {sum(len(w) for w in waves.values()) / 16000:.1f} s -> {args.out}")
