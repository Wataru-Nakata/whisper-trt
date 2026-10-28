"""Reference long-form transcriptions with faster-whisper 1.2.1's WhisperModel.transcribe (in the reference env,
tools/fw_ref): every recording of a long-*.npz with faster-whisper's defaults (beam 5, temperature fallback,
conditioned on the previous text, timestamps), with and without word timestamps; segments with their tokens,
scores and words, and the time taken.

    python scripts/ref_long.py WAVES.npz OUT.json --lang ja [--compute-types float32 float16] [--word-timestamps]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ap = argparse.ArgumentParser()
ap.add_argument("waves")
ap.add_argument("out")
ap.add_argument("--lang", required=True)
ap.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / ".cache" / "fw-large-v3"))
ap.add_argument("--compute-types", nargs="+", default=["float16"])
ap.add_argument("--word-timestamps", action="store_true")
args = ap.parse_args()

data = np.load(args.waves)
meta = json.loads(str(data["meta"]))
waves = [data[f"w{i}"].astype(np.float32) / 32768 for i in range(len(meta))]
audio_s = sum(len(w) for w in waves) / 16000
out = {"meta": meta, "audio_seconds": audio_s, "runs": {}}
for ct in args.compute_types:
    m = WhisperModel(args.model, device="cuda", compute_type=ct, local_files_only=Path(args.model).is_dir())
    t0 = time.perf_counter()
    recs = []
    for w in waves:
        segments, _ = m.transcribe(w, language=args.lang, beam_size=5, word_timestamps=args.word_timestamps)
        recs.append([{"start": s.start, "end": s.end, "text": s.text, "tokens": s.tokens, "seek": s.seek,
                      "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob, "temperature": s.temperature,
                      "words": [[w.start, w.end, w.word, w.probability] for w in s.words] if s.words else None}
                     for s in segments])
    sec = time.perf_counter() - t0
    key = f"{ct}/beam5" + ("/words" if args.word_timestamps else "")
    out["runs"][key] = {"recordings": recs, "seconds": sec, "rtfx": audio_s / sec}
    print(f"{key}: {audio_s / sec:.1f} RTFx, {sum(len(r) for r in recs)} segments", flush=True)
    del m
json.dump(out, open(args.out, "w"), ensure_ascii=False)
