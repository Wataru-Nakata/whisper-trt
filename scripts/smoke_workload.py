"""The rest of a faster-whisper clip-transcription workload on clips (scripts/make_clips.py + scripts/ref_ct2.py outputs):
language identification against CTranslate2's, decode_windows with the temperature fallback (sampling), and
(optional) chosen clips decoded by the PyTorch fp16 backend next to TensorRT's.

    python scripts/smoke_workload.py CLIPS.npz REF.json --lang ja [--torch16 30 52]
"""

import argparse
import json
import time

import numpy as np
import torch

from whisper_trt import Whisper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("ref")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--run", default="float16/beam5/b8")
    ap.add_argument("--torch16", nargs="*", type=int, default=[])
    ap.add_argument("--force-fallback", action="store_true",
                    help="log_prob_threshold 0: every window goes through every temperature (best-of-5 sampling)")
    args = ap.parse_args()
    data = np.load(args.clips)
    meta = json.loads(str(data["meta"]))
    waves = [torch.from_numpy(data[f"w{i}"]) for i in range(len(meta))]
    ref = json.load(open(args.ref))["runs"][args.run]
    model = Whisper.from_pretrained(args.model)

    # language identification
    lid = []
    for i in range(0, len(waves), 8):
        lid += model.detect_language(model.encode(model.features(waves[i : i + 8])))
    same, dmax = 0, 0.0
    for ours, theirs in zip(lid, ref["lid"]):
        top = theirs[0][0][2:-2]
        same += ours[0][0] == top
        p = dict(ours)
        dmax = max(dmax, max(abs(p[t[2:-2]] - q) for t, q in theirs))
    print(f"LID: top-1 equal {same}/{len(lid)}, max |prob diff| over CTranslate2's top 3 {dmax:.3g}", flush=True)

    # windows with the temperature fallback
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kw = {"log_prob_threshold": 0.0, "no_speech_threshold": None} if args.force_fallback else {}
    out = model.decode_windows(waves, args.lang, batch_size=8, **kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    fell = [d for d in out if d.temperature > 0]
    temps = sorted({d.temperature for d in fell})
    limit = sum(d.reached_max_length for d in out)
    audio = sum(len(w) for w in waves) / 16000
    print(f"decode_windows: {len(out)} windows in {dt:.2f} s (RTFx {audio / dt:.0f}), {len(fell)} took the fallback "
          f"(final temperatures {temps}), {limit} reached the token limit", flush=True)
    for d in fell[:5]:
        print(f"   T={d.temperature} after {len(d.attempts)} attempts: {d.text[:100]!r}", flush=True)
    if fell:
        by_t = {}
        for d in out:
            for a in d.attempts + [{"temperature": d.temperature, "avg_logprob": d.avg_logprob}]:
                by_t.setdefault(a["temperature"], []).append(a["avg_logprob"])
        print("   mean avg_logprob by temperature:",
              {t: round(sum(v) / len(v), 3) for t, v in sorted(by_t.items())}, flush=True)
    rows = ref["rows"]
    same = sum(d.tokens == r["tokens"] for d, r in zip(out, rows) if d.temperature == 0)
    print(f"   temperature-0 windows identical to CTranslate2's first pass: {same}/{len(out) - len(fell)}", flush=True)

    if args.torch16:
        idx = args.torch16
        t16 = Whisper.from_pretrained(args.model, backend="torch", dtype="float16")
        prompt = model.prompt(args.lang)
        a = model.generate(model.encode(model.features([waves[i] for i in idx])), prompt, beam_size=5)
        b = t16.generate(t16.encode(t16.features([waves[i] for i in idx])), prompt, beam_size=5)
        for i, x, y in zip(idx, a, b):
            r = rows[i]
            print(f"[{i}] tensorrt {len(x.tokens)} tokens (score {x.hypothesis.normalized:.4f}), torch fp16 "
                  f"{len(y.tokens)} ({y.hypothesis.normalized:.4f}), ctranslate2 {len(r['tokens'])} ({r['score']:.4f}); "
                  f"torch16 == ct2: {y.tokens == r['tokens']}, tensorrt == ct2: {x.tokens == r['tokens']}", flush=True)


if __name__ == "__main__":
    main()
