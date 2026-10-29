"""Language detection of long-*.npz recordings (transcribe's language=None): faster-whisper's
``WhisperModel.detect_language`` on the recording's features, as its ``transcribe`` calls it (``--ref``, in the
reference env) or ours (longform.detect_languages), as JSON
lines; with ``--against REF.jsonl``, ours compared with faster-whisper's (top-1, probability).

    python scripts/lid_long.py --ref --model .cache/fw-large-v3 OUT.jsonl WAVES.npz ...      (reference env)
    python scripts/lid_long.py OUT.jsonl WAVES.npz ... [--against REF.jsonl]
"""

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("waves", nargs="+")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--against")
    args = ap.parse_args()
    rows = []
    if args.ref:
        from faster_whisper import WhisperModel

        model = WhisperModel(args.model, device="cuda", compute_type="float32")
    else:
        import torch

        from whisper_trt import Whisper
        from whisper_trt.longform import _Recording, detect_languages, recording_features

        model = Whisper.from_pretrained(backend="torch", dtype="float32")
    for path in args.waves:
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        for i in range(len(meta)):
            wave = data[f"w{i}"].astype(np.float32) / 32768
            if args.ref:
                # (as transcribe detects: on the whole recording's features, normalized once)
                language, p, _ = model.detect_language(features=model.feature_extractor(wave))
            else:
                rec = _Recording(i, recording_features(model, torch.from_numpy(wave)), [])
                detect_languages(model, [rec])
                language, p = rec.language, rec.language_probability
            rows.append({"set": path.rsplit("/", 1)[-1], "i": i, "language": language, "probability": float(p)})
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if args.against:
        ref = {(r["set"], r["i"]): r for r in map(json.loads, open(args.against))}
        same = [r["language"] == ref[r["set"], r["i"]]["language"] for r in rows]
        diff = [abs(r["probability"] - ref[r["set"], r["i"]]["probability"]) for r in rows]
        print(f"{len(rows)} recordings: top-1 the same {sum(same)}/{len(rows)}, probability within {max(diff):.1e}")


if __name__ == "__main__":
    main()
