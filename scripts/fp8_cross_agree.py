"""Would 8-bit cross-attention keys/values keep the decodes? Emulated: after the encoder, every layer's keys and
values are rounded to float8 e4m3 or int8 (scale per window, head and channel over the 1500 frames) and back,
then decoded as usual; agreement with a reference run and the outputs (for scripts/yodas_captions.py) are
written.

    python scripts/fp8_cross_agree.py WAVES.npz REF.json --lang ja --out OURS.json [--scale channel|head]
        [--format fp8 int8] [--no-baseline]
"""

import argparse
import json

import numpy as np
import torch

from whisper_trt import Whisper


def qdq(cross: torch.Tensor, per: str, fmt: str) -> None:
    """In place: cross [L, 2, W, H, T, D] through float8 e4m3 or int8."""
    dims = (4,) if per == "channel" else (4, 5)
    amax = cross.float().abs().amax(dim=dims, keepdim=True).clamp(min=1e-6)
    if fmt == "fp8":
        scale = amax / 448.0
        q = (cross.float() / scale).to(torch.float8_e4m3fn).float()
    else:
        scale = amax / 127.0
        q = (cross.float() / scale).round().clamp(-127, 127)
    cross.copy_((q * scale).to(cross.dtype))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("ref")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out")
    ap.add_argument("--scale", default="channel", choices=("channel", "head"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--format", nargs="+", default=["fp8"], choices=("fp8", "int8"))
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = []
    for i in range(len(meta)):
        w = data[f"w{i}"]
        waves.append(torch.from_numpy(w.astype(np.float32) / 32768 if w.dtype == np.int16 else w))
    refs = json.load(open(args.ref))["runs"]
    model = Whisper.from_pretrained()
    encode = model.backend.encode

    def quantized(fmt):
        def enc(feats, cross, lane=0):
            encode(feats, cross, lane)
            qdq(cross, args.scale, fmt)

        return enc

    runs = [] if args.no_baseline else [("fp16", encode)]
    runs += [(f"{fmt}-emulated", quantized(fmt)) for fmt in args.format]
    out = {}
    for name, enc in runs:
        model.backend.encode = enc
        res = model.decode(waves, model.prompt(args.lang), beam_size=5, batch_size=args.batch)
        rows = [{"tokens": r.tokens, "score": r.hypothesis.normalized, "no_speech_prob": r.no_speech_prob,
                 "text": model.tokenizer.decode(r.tokens).strip()} for r in res]
        out[f"beam5/b{args.batch}/{name}"] = {"rows": rows}
        line = f"{name}:"
        for rk, run in refs.items():
            same = sum(a["tokens"] == b["tokens"] for a, b in zip(rows, run["rows"]))
            line += f" vs {rk} {same}/{len(rows)}"
        print(line, flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
