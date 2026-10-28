"""Compare our decodes with faster-whisper's (scripts/ref_ct2.py) on the same waveforms.

    python scripts/parity.py WAVES.npz REF.json --backend torch --dtype float32 --out OURS.json

Prints feature/encoder differences (if REF's features file exists), and per configuration: RTFx (encoder +
decoding of all windows, features included), identical transcripts and token sequences against each
reference run of the same search, the largest score / no-speech differences, and WER against the manifest.
"""

import argparse
import json
import re
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from whisper_trt import Whisper


def load_wave(data, i: int) -> np.ndarray:
    w = data[f"w{i}"]
    return w.astype(np.float32) / 32768 if w.dtype == np.int16 else w


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def chars(text: str) -> list[str]:
    return list(re.sub(r"[\W_]+", "", text.lower()))


CER_LANGS = {"ja", "zh", "yue", "th"}  # written without spaces: character error rate
UNITS = words


def wer(hyps: list[str], refs: list[str]) -> float:
    errors = sum(edit_distance(UNITS(h), UNITS(r)) for h, r in zip(hyps, refs))
    return errors / max(1, sum(len(UNITS(r)) for r in refs))


def build_options(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        out[k] = v.lower() == "true" if v.lower() in ("true", "false") else int(v) if v.lstrip("-").isdigit() else v
    return out


def decisions(row: dict, max_new: int = 224) -> dict:
    """faster-whisper's judgments of a window's decode: avg_logprob, compression ratio and the rules."""
    n = len(row["tokens"])
    lp = row["score"] * n / (n + 1) if n else float("-inf")
    data = row["text"].encode()
    cr = len(data) / len(zlib.compress(data)) if data else 0.0
    silence = row["no_speech_prob"] > 0.6 and lp < -1.0
    return {"avg_logprob": lp, "compression_ratio": cr, "silence": silence,
            "fallback": not silence and (cr > 2.4 or lp < -1.0), "limit": n >= max_new}


def compare(rows: list[dict], run: dict, meta: list[dict], rkey: str, have_text: bool, report: dict,
            key: str = "") -> str:
    rr = run["rows"]
    n = len(rr)
    same = [a["tokens"] == b["tokens"] for a, b in zip(rows, rr)]
    both = [(a, b) for a, b, eq in zip(rows, rr, same) if eq]
    ds = max((abs(a["score"] - b["score"]) for a, b in both), default=0.0)
    dn = max(abs(a["no_speech_prob"] - b["no_speech_prob"]) for a, b in zip(rows, rr))
    da = [decisions(a) for a in rows]
    db = [decisions(b) for b in rr]
    line = f"    vs {rkey} (RTFx {run['rtfx']:.1f}"
    if have_text:
        line += f", {'CER' if UNITS is chars else 'WER'} {wer([r['text'] for r in rr], [m['text'] for m in meta]):.2%}"
    line += f"): tokens {sum(same)}/{n}, max |score diff| {ds:.2g}, max |no_speech diff| {dn:.2g}"
    for flag in ("silence", "fallback", "limit"):
        ours, theirs = sum(d[flag] for d in da), sum(d[flag] for d in db)
        agree = sum(x[flag] == y[flag] for x, y in zip(da, db))
        line += f"\n      {flag}: ours {ours}, ref {theirs}, agree {agree}/{n}"
    diff = [i for i in range(n) if not same[i]]
    if diff:  # how different the differing windows are: character error rate of ours against the reference
        err = sum(edit_distance(chars(rows[i]["text"]), chars(rr[i]["text"])) for i in diff)
        tot = sum(len(chars(rr[i]["text"])) for i in diff)
        line += f"\n      differing windows: {len(diff)}, their character difference {err / max(1, tot):.2%}"
    report[f"{key} vs {rkey}"] = [{"i": i, "id": meta[i].get("id"), "ours": rows[i]["text"], "ref": rr[i]["text"],
                     "len": [len(rows[i]["tokens"]), len(rr[i]["tokens"])],
                     "ours_decisions": da[i], "ref_decisions": db[i]} for i in diff]
    return line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("ref")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--build", nargs="*", default=[], help="engine build options KEY=VALUE (build.BuildOptions)")
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--configs", nargs="+", default=["1x1", "5x1", "5x8"], help="BEAMxBATCH")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--stream", action="store_true",
                    help="all windows through refilled slots (Whisper.decode) instead of fixed batches")
    ap.add_argument("--out")
    ap.add_argument("--report", help="every window that differs from the reference, as JSON")
    ap.add_argument("--timestamps", action="store_true", help="decode with timestamps (the reference too)")
    args = ap.parse_args()
    global UNITS
    UNITS = chars if args.lang in CER_LANGS else words
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = [torch.from_numpy(load_wave(data, i)) for i in range(len(meta))]
    audio_s = sum(len(w) for w in waves) / 16000
    ref = json.load(open(args.ref))
    t0 = time.perf_counter()
    opts = build_options(args.build)
    sizes = {k: opts.pop(k) for k in ("max_batch", "max_rows") if k in opts}
    model = Whisper.from_pretrained(args.model, backend=args.backend, dtype=args.dtype, **sizes, **opts)
    print(f"loaded {args.backend}/{args.dtype} in {time.perf_counter() - t0:.1f}s", flush=True)

    feats_path = Path(args.ref.replace(".json", "-features.npz"))
    if feats_path.exists():
        f = np.load(feats_path)
        k = f["feats"].shape[0]
        ours = model.features(waves[:k])
        d = (ours.cpu() - torch.from_numpy(f["feats"])).abs()
        print(f"features: max |diff| {d.max():.3g} mean {d.mean():.3g}")
        if model.model is not None:
            with torch.inference_mode():
                enc = model.model.encoder(ours.to(model.model.encoder.conv1.weight.dtype)).float().cpu()
            e = torch.from_numpy(f["enc"])
            d = (enc - e).abs()
            print(f"encoder: max |diff| {d.max():.3g} mean {d.mean():.3g} (ref max |x| {e.abs().max():.3g})")

    prompt = model.prompt(args.lang, without_timestamps=not args.timestamps)
    out, report = {}, {}
    for cfg in args.configs:
        beam, batch = map(int, cfg.split("x"))
        times = []
        for rep in range(args.repeat + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.stream:
                res = model.decode(waves, prompt, beam_size=beam, batch_size=batch)
            else:
                res = []
                for i in range(0, len(waves), batch):
                    res += model.generate(model.encode(model.features(waves[i : i + batch])), prompt, beam_size=beam)
            rows = [{"tokens": r.tokens, "score": r.hypothesis.normalized, "no_speech_prob": r.no_speech_prob,
                     "text": model.tokenizer.decode(r.tokens).strip()} for r in res]
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        best = min(times[1:]) if len(times) > 1 else times[0]
        key = f"beam{beam}/b{batch}" + ("/stream" if args.stream else "")
        peak = torch.cuda.max_memory_allocated() / 2**30
        out[key] = {"rows": rows, "seconds": times, "rtfx": audio_s / best}
        have_text = any(m["text"] for m in meta)
        line = f"{key}: RTFx {audio_s / best:.1f} (peak GPU memory so far {peak:.1f} GiB)"
        if have_text:
            line += f", {'CER' if UNITS is chars else 'WER'} {wer([r['text'] for r in rows], [m['text'] for m in meta]):.2%}"
        for rkey, run in ref["runs"].items():
            if f"/beam{beam}/" not in rkey:
                continue
            line += "\n" + compare(rows, run, meta, rkey, have_text, report, key)
        print(line, flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    if args.report:
        json.dump(report, open(args.report, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
