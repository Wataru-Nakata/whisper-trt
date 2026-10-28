"""Scores of saved long-form outputs (scripts/longform_parity.py --out) against faster-whisper's (scripts/ref_long.py),
without a GPU: the same metrics as longform_parity.py, and against the recordings' reference text when the
waves file has one (meta "text", e.g. scripts/make_long_ls.py).

    python scripts/longform_compare.py OURS.json REF.json --lang en [--run b8] [--waves long-ls.npz]
"""

import argparse
import json
import re

import numpy as np

CER_LANGS = {"ja", "zh", "th", "ko", "yue"}


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("ref")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--run", default="b8")
    ap.add_argument("--waves")
    args = ap.parse_args()
    ours = json.load(open(args.ours))[args.run]
    ref = json.load(open(args.ref))["runs"]
    cer = args.lang in CER_LANGS
    units = (lambda t: list(t.replace(" ", ""))) if cer else (lambda t: t.split())  # noqa: E731
    truth = None
    if args.waves:
        meta = json.loads(str(np.load(args.waves)["meta"]))
        if all(m.get("text") for m in meta):
            truth = [re.sub(r"[^A-Z' ]", " ", m["text"].upper()).split() for m in meta]
    runs = {"ours": ours, **{k: v["recordings"] for k, v in ref.items()}}
    if truth:
        for name, recs in runs.items():
            e = sum(edit_distance(re.sub(r"[^A-Z' ]", " ", " ".join(s["text"] for s in segs).upper()).split(), t)
                    for segs, t in zip(recs, truth))
            print(f"{name}: WER against the reference text {e / sum(len(t) for t in truth):.2%}")
    for key, theirs_all in ((k, v["recordings"]) for k, v in ref.items()):
        same = det_same = det_total = det_rec = errs = n = 0
        word_d = []
        for a, b in zip(ours, theirs_all):
            ka = [(s["tokens"], round(s["start"], 3), round(s["end"], 3)) for s in a]
            kb = [(s["tokens"], round(s["start"], 3), round(s["end"], 3)) for s in b]
            same += ka == kb
            k = next((i for i, (x, y) in enumerate(zip(ka, kb)) if x != y), min(len(ka), len(kb)))
            upto = min(next((i for i, s in enumerate(a) if s["temperature"] > 0), len(a)),
                       next((i for i, s in enumerate(b) if s["temperature"] > 0), len(b)))
            det_same += min(k, upto)
            det_total += upto
            det_rec += k >= upto
            ua, ub = units(" ".join(s["text"].strip() for s in a)), units(" ".join(s["text"].strip() for s in b))
            errs += edit_distance(ua, ub)
            n += len(ub)
            for s, t in zip(a[:k], b[:k]):
                for w, v in zip(s.get("words") or [], t.get("words") or []):
                    word_d.append(max(abs(w[0] - v[0]), abs(w[1] - v[1])))
        line = (f"vs {key}: recordings with the same segments {same}/{len(ours)}; before the first sampled window "
                f"segments the same {det_same}/{det_total}, recordings {det_rec}/{len(ours)}; "
                f"{'CER' if cer else 'WER'} against it {errs / max(n, 1):.2%}")
        if word_d:
            d = np.array(word_d)
            line += (f"; words of the same segments {len(d)}: same times {np.mean(d < 1e-6):.1%}, within 0.02 s "
                     f"{np.mean(d <= 0.0200001):.1%}, max diff {d.max():.2f} s")
        print(line, flush=True)


if __name__ == "__main__":
    main()
