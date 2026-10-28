"""Long recordings from YODAS v3 for long-form transcription tests: the first minutes of each recording (decoded
as faster-whisper decodes files, scripts/make_eval_yodas.py), at least --min-seconds long, as int16 waves with
meta for scripts/ref_long.py and scripts/longform_parity.py. Runs in the reference env (PyAV).

    python scripts/make_eval_long.py OUT_DIR --langs ja en --per-lang 20 --seconds 300
"""

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
from make_eval_yodas import ROOT, TO_WHISPER, decode_prefix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--per-lang", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--min-seconds", type=float, default=60.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for lang in args.langs:
        path = out / f"long-{lang}.npz"
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
                    audio = decode_prefix(tf.extractfile(member).read(), args.seconds)[: int(args.seconds * 16000)]
                except Exception as e:  # noqa: BLE001 - undecodable files are skipped
                    print(f"skip {member.name}: {e}", flush=True)
                    continue
                if len(audio) < args.min_seconds * 16000:
                    continue
                waves[f"w{len(meta)}"] = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
                meta.append({"id": Path(member.name).stem, "lang": TO_WHISPER.get(lang, lang)})
        np.savez(path, meta=json.dumps(meta), **waves)
        total = sum(len(w) for w in waves.values()) / 16000
        print(f"{lang}: {len(meta)} recordings, {total / 3600:.2f} h -> {path}", flush=True)


if __name__ == "__main__":
    main()
