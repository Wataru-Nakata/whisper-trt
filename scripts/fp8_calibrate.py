"""Input ranges of the FP8 encoder layers (fp8.py) on calibration audio, as a scale file: the PyTorch fp16
encoder and cross-attention projections run on every window, and each layer's largest absolute input is kept.

    python scripts/fp8_calibrate.py CALIB_DIR [--out src/whisper_trt/fp8_scales/whisper-large-v3.json]

CALIB_DIR: yodas-*.npz from scripts/make_eval_yodas.py --tar -1 (recordings disjoint from the evaluation sets).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from whisper_trt import fp8, hub
from whisper_trt.config import N_FRAMES, load_config
from whisper_trt.features import log_mel
from whisper_trt.modeling import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("calib_dir")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--out", default="src/whisper_trt/fp8_scales/whisper-large-v3.json")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    model_dir = hub.resolve(args.model)
    cfg = load_config(model_dir)
    model = load_model(model_dir, cfg, dtype=torch.float16)
    waves, langs = [], {}
    for path in sorted(Path(args.calib_dir).glob("yodas-*.npz")):
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        for i in range(len(meta)):
            waves.append(torch.from_numpy(data[f"w{i}"].astype(np.float32) / 32768))
        langs[path.stem.removeprefix("yodas-")] = len(meta)
    print(f"{len(waves)} windows ({sum(len(w) for w in waves) / 16000 / 3600:.2f} h): {langs}", flush=True)
    batches = [waves[i : i + args.batch] for i in range(0, len(waves), args.batch)]

    def run(batch):
        with torch.inference_mode():
            model.decoder.cross_kv(model.encoder(log_mel(batch, cfg.n_mels, "cuda", N_FRAMES)))

    amax = fp8.calibrate(model, run, batches)
    sha = fp8.checkpoint_sha256(next(model_dir.glob("model*.safetensors")))
    about = {"audio": "YODAS v3, the last audio tar of each language (the evaluation sets read the first)",
             "windows": langs, "statistic": "max |input|"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp8.save_scales(out, amax, args.model, sha, about)
    top = sorted(amax.items(), key=lambda x: -x[1])[:5]
    print(f"{len(amax)} layers -> {out}; largest ranges: {top}", flush=True)


if __name__ == "__main__":
    main()
