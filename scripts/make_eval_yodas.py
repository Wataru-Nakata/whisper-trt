"""Real-world windows from YODAS v3 (web audio: music, noise, silence, several speakers) for agreement tests.

For each language: recordings in the order of its first audio tar, decoded with PyAV to 16 kHz mono as
faster-whisper decodes files, cut into windows from the start: 30 s long, or (one in three) of a random length
between 0.3 and 30 s, at most --per-recording windows per recording. No VAD, so silence and non-speech stay.
Saved as int16 waves (``w{i}``) with meta (recording id, offset, language) for scripts/ref_ct2.py and
scripts/parity.py. Runs in the reference env (PyAV).

    python scripts/make_eval_yodas.py OUT_DIR --langs ja en de --per-lang 300
"""

import argparse
import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio

ROOT = Path("/groups/gai51740/acc12576tt/datasets/yodas3/data")
TO_WHISPER = {"iw": "he", "jv": "jw", "fil": "tl"}  # YODAS directory codes that Whisper spells differently


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--per-lang", type=int, default=300)
    ap.add_argument("--per-recording", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for lang in args.langs:
        path = out / f"yodas-{lang}.npz"
        if path.exists():
            print(f"{path} exists", flush=True)
            continue
        waves, meta = {}, []
        tar = sorted((ROOT / lang / "audio").glob("*.tar"))[0]
        with tarfile.open(tar) as tf:
            for member in tf:
                if len(meta) >= args.per_lang:
                    break
                if not member.isfile():
                    continue
                try:
                    audio = decode_audio(io.BytesIO(tf.extractfile(member).read()), sampling_rate=16000)
                except Exception as e:  # noqa: BLE001 - undecodable files are skipped
                    print(f"skip {member.name}: {e}", flush=True)
                    continue
                pos = 0
                for _ in range(args.per_recording):
                    n = 480000 if rng.random() < 2 / 3 else int(rng.uniform(0.3, 30) * 16000)
                    piece = audio[pos : pos + n]
                    if len(piece) < 0.3 * 16000 or len(meta) >= args.per_lang:
                        break
                    waves[f"w{len(meta)}"] = np.clip(piece * 32768, -32768, 32767).astype(np.int16)
                    meta.append({"id": f"{Path(member.name).stem}@{pos / 16000:.1f}", "text": "",
                                 "lang": TO_WHISPER.get(lang, lang)})
                    pos += n
        np.savez(path, meta=json.dumps(meta), **waves)
        total = sum(len(w) for w in waves.values()) / 16000
        print(f"{lang}: {len(meta)} windows, {total / 3600:.2f} h -> {path}", flush=True)


if __name__ == "__main__":
    main()
