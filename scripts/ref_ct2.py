"""Reference decodes with faster-whisper 1.2.1 / CTranslate2 (run in the reference env, tools/fw_ref).

Each window is decoded as faster-whisper's batched pipeline decodes a clip: its own features
(``feature_extractor(wave)[..., :-1]``, padded to 3000 frames), the prompt sot/lang/transcribe/notimestamps,
``Whisper.generate`` with faster-whisper's options (suppress_tokens=[-1] expanded, suppress_blank, no
penalties, max_length 448).

    python scripts/ref_ct2.py WAVES.npz OUT.json --model DIR --lang en --compute-types float32 float16 \
        --configs 1x1 5x1 5x8
"""

import argparse
import json
import time

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_ctranslate2_storage, get_suppressed_tokens

ap = argparse.ArgumentParser()
ap.add_argument("waves")
ap.add_argument("out")
ap.add_argument("--model", default=".cache/fw-large-v3")
ap.add_argument("--lang", default="en")
ap.add_argument("--compute-types", nargs="+", default=["float32", "float16"])
ap.add_argument("--configs", nargs="+", default=["1x1", "5x1", "5x8"], help="BEAMxBATCH")
ap.add_argument("--save-features", type=int, default=3, help="features/encoder output of the first N")
ap.add_argument("--repeat", type=int, default=2, help="timed passes after one warm-up pass")
args = ap.parse_args()

data = np.load(args.waves)
meta = json.loads(str(data["meta"]))
waves = [data[f"w{i}"] for i in range(len(meta))]
waves = [w.astype(np.float32) / 32768 if w.dtype == np.int16 else w for w in waves]  # (int16 files)
audio_s = sum(len(w) for w in waves) / 16000
out = {"meta": meta, "audio_seconds": audio_s, "runs": {}}
for ct in args.compute_types:
    m = WhisperModel(args.model, device="cuda", compute_type=ct, local_files_only=True)
    tok = Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task="transcribe", language=args.lang)
    prompt = m.get_prompt(tok, [], without_timestamps=True)
    suppress = list(get_suppressed_tokens(tok, [-1]))
    t0 = time.perf_counter()
    feats = np.stack([pad_or_trim(m.feature_extractor(w)[..., :-1]) for w in waves])
    feat_s = time.perf_counter() - t0
    if args.save_features and ct == args.compute_types[0]:
        k = args.save_features
        enc = np.array(m.model.encode(get_ctranslate2_storage(feats[:k]), to_cpu=True))
        np.savez(args.out.replace(".json", "-features.npz"), feats=feats[:k], enc=enc)
    for cfg in args.configs:
        beam, batch = map(int, cfg.split("x"))
        times = []
        for rep in range(args.repeat + 1):
            res, lid = [], []
            t0 = time.perf_counter()
            for i in range(0, len(waves), batch):
                enc = m.model.encode(get_ctranslate2_storage(feats[i : i + batch]))
                if rep == 0:
                    lid += [r[:3] for r in m.model.detect_language(enc)]
                r = m.model.generate(
                    enc, [prompt] * min(batch, len(waves) - i), beam_size=beam, patience=1, length_penalty=1,
                    repetition_penalty=1, no_repeat_ngram_size=0, max_length=448, return_scores=True,
                    return_no_speech_prob=True, suppress_blank=True, suppress_tokens=suppress,
                    max_initial_timestamp_index=50,
                )
                res += r
            times.append(time.perf_counter() - t0)
            if rep == 0:
                rows = [{"tokens": r.sequences_ids[0], "score": r.scores[0], "no_speech_prob": r.no_speech_prob,
                         "text": tok.decode(r.sequences_ids[0]).strip()} for r in res]
                first_lid = lid
        best = min(times[1:]) if len(times) > 1 else times[0]
        key = f"{ct}/beam{beam}/b{batch}"
        out["runs"][key] = {"rows": rows, "seconds": times, "rtfx": audio_s / best, "feature_seconds": feat_s,
                            "lid": first_lid}
        print(f"{key}: {audio_s / best:.1f} RTFx (encode+generate, features {feat_s:.2f}s apart)", flush=True)
    del m
json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
