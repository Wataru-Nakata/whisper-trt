"""Real-world windows from YODAS v3 (web audio: music, noise, silence, several speakers) for agreement tests.

For each language: recordings in the order of its first audio tar, decoded with PyAV to 16 kHz mono as
faster-whisper decodes files (only as much as the windows need), cut into windows from the start: 30 s long, or (one in three) of a random length
between 0.3 and 30 s, at most --per-recording windows per recording. No VAD, so silence and non-speech stay.
Saved as int16 waves (``w{i}``) with meta (recording id, offset, language) for scripts/ref_ct2.py and
scripts/parity.py. Runs in the reference env (PyAV).

    python scripts/make_eval_yodas.py OUT_DIR --langs ja en de --per-lang 300 [--tar -1]

``--tar``: which of the language's audio tars (in name order) to read: the evaluation sets take the first (0);
-1, the last, gives recordings disjoint from them (FP8 calibration, scripts/fp8_calibrate.py).
"""

import argparse
import io
import json
import random
import tarfile
from pathlib import Path

import av
import numpy as np
from faster_whisper.audio import _group_frames, _ignore_invalid_frames, _resample_frames

ROOT = Path("/groups/gai51740/acc12576tt/datasets/yodas3/data")
TO_WHISPER = {"iw": "he", "jv": "jw", "fil": "tl"}  # YODAS directory codes that Whisper spells differently


def decode_prefix(data: bytes, seconds: float) -> np.ndarray:
    """faster-whisper's ``decode_audio`` (PyAV, s16 mono 16 kHz) of the first ``seconds`` (and a second more,
    so that the resampler's delay does not touch them): the same samples, without decoding hours of video."""
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    out, n, need = [], 0, int((seconds + 1) * 16000)
    with av.open(io.BytesIO(data), mode="r", metadata_errors="ignore") as container:
        frames = _resample_frames(_group_frames(_ignore_invalid_frames(container.decode(audio=0)), 500000),
                                  resampler)
        for frame in frames:
            a = frame.to_ndarray().reshape(-1)
            out.append(a)
            n += len(a)
            if n >= need:
                break
    return (np.concatenate(out) if out else np.zeros(0, np.int16)).astype(np.float32) / 32768.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--per-lang", type=int, default=300)
    ap.add_argument("--per-recording", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tar", type=int, default=0)
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
        tars = sorted((ROOT / lang / "audio").glob("*.tar"))
        tar = tars[args.tar]
        if args.tar != 0 and len(tars) < 2:
            print(f"{lang}: one tar only, skipped", flush=True)
            continue
        with tarfile.open(tar) as tf:
            for member in tf:
                if len(meta) >= args.per_lang:
                    break
                if not member.isfile():
                    continue
                try:
                    audio = decode_prefix(tf.extractfile(member).read(), 30.0 * args.per_recording)
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
