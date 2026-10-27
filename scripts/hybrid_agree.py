"""Which half of the TensorRT engines moves decodes away from faster-whisper fp32: the encoder or the decoder?
Decodes the same windows with the encoder and the decoder each from TensorRT or from PyTorch fp16 (cuBLAS,
fp32 accumulation) and counts the windows whose tokens equal a reference run.

    python scripts/hybrid_agree.py WAVES.npz REF.json --lang ja [--ref-run float32/beam5/b8] [--limit 300]
"""

import argparse
import json

import numpy as np
import torch

from whisper_trt import Whisper
from whisper_trt.runtime import TorchBackend


class Hybrid:
    """The encoder of one backend and the decoder calls of another."""

    def __init__(self, enc, dec):
        self.enc, self.dec = enc, dec
        self.name, self.dtype, self.cfg = f"{enc.name}+{dec.name}", dec.dtype, dec.cfg

    def encode(self, feats, cross, lane: int = 0):
        return self.enc.encode(feats, cross, lane) if self.enc.name == "tensorrt" else self.enc.encode(feats, cross)

    def step_fn(self, *args, **kwargs):
        return self.dec.step_fn(*args, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("ref")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--ref-run", default="float32/beam5/b8")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))[: args.limit]
    waves = []
    for i in range(len(meta)):
        w = data[f"w{i}"]
        waves.append(torch.from_numpy(w.astype(np.float32) / 32768 if w.dtype == np.int16 else w))
    ref = json.load(open(args.ref))["runs"][args.ref_run]["rows"]
    trt_model = Whisper.from_pretrained()
    t16 = Whisper.from_pretrained(backend="torch", dtype="float16")
    trt, torch16 = trt_model.backend, TorchBackend(t16.model)
    for enc, dec in ((trt, trt), (trt, torch16), (torch16, trt), (torch16, torch16)):
        model = Whisper(trt_model.cfg, trt_model.tokenizer, Hybrid(enc, dec), t16.model)
        res = model.decode(waves, model.prompt(args.lang), beam_size=5, batch_size=8)
        same = sum(r.tokens == x["tokens"] for r, x in zip(res, ref))
        print(f"encoder {enc.name}, decoder {dec.name}: tokens equal to {args.ref_run} on {same}/{len(res)}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
