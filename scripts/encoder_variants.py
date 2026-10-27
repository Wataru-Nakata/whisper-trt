"""Precision variants of the TensorRT encoder against the fp32 PyTorch encoder, on a truncated encoder (fast
builds) and real audio.

    python scripts/encoder_variants.py WAVES.npz [--layers 8] [--variants base residual32 softmax32 gelu32 all]

For each variant: export (modeling.Precision flags), build, run, and print the error of the encoder output
against fp32 (and, for scale, PyTorch fp16's own error).
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from whisper_trt import hub
from whisper_trt.build import BuildOptions, _build, _quiet_exporter
from whisper_trt.config import N_FRAMES, load_config
from whisper_trt.features import log_mel
from whisper_trt.modeling import Attention, Precision, load_model
from whisper_trt.runtime import TRTBackend

VARIANTS = {
    "base": {},
    "residual32": {"residual32": True},
    "softmax32": {"softmax32": True},
    "gelu32": {"gelu32": True},
    "all": {"residual32": True, "softmax32": True, "gelu32": True},
}


class EncoderOnly(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, feats):
        return self.encoder(feats)


def err(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    d = (out.float() - ref.float()).abs()
    print(f"  {name}: mean |diff| {d.mean():.4g}, max {d.max():.4g}, rms {d.pow(2).mean().sqrt():.4g} "
          f"(ref rms {ref.float().pow(2).mean().sqrt():.3g})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = [torch.from_numpy(data[f"w{i}"]) for i in range(min(args.n, len(meta)))]
    model_dir = hub.resolve(args.model)
    cfg = load_config(model_dir)
    feats = log_mel(waves, cfg.n_mels, "cuda", N_FRAMES)
    m32 = load_model(model_dir, cfg, dtype=torch.float32)
    m16 = load_model(model_dir, cfg, dtype=torch.float16)
    for m in (m32, m16):
        del m.encoder.layers[args.layers :]
    with torch.inference_mode():
        ref = m32.encoder(feats)
        print(f"{args.layers} layers; torch fp16:")
        err("torch fp16", m16.encoder(feats), ref)
    opts = BuildOptions(max_batch=4)
    for name in args.variants:
        flags = {"residual32": False, "softmax32": False, "gelu32": False, **VARIANTS[name]}
        saved = {k: getattr(Precision, k) for k in flags}
        for k, v in flags.items():
            setattr(Precision, k, v)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                onnx = Path(tmp) / "enc.onnx"
                Attention.explicit = True
                with _quiet_exporter():
                    torch.onnx.export(EncoderOnly(m16.encoder).eval(), (feats[:1].clone(),), onnx,
                                      input_names=["feats"], output_names=["enc"],
                                      dynamic_shapes={"feats": {0: torch.export.Dim("batch", min=1, max=4)}},
                                      dynamo=True, external_data=True)
                Attention.explicit = False
                engine = Path(tmp) / "enc.engine"
                _build(onnx, engine, [{"feats": ((1, cfg.n_mels, N_FRAMES), (2, cfg.n_mels, N_FRAMES),
                                                 (4, cfg.n_mels, N_FRAMES))}], opts)
                be = TRTBackend.__new__(TRTBackend)
                import tensorrt as trt

                from whisper_trt.runtime import trt_logger

                be.trt, be.runtime, be._scratch = trt, trt.Runtime(trt_logger(trt)), {}
                eng = be._load(engine)
                ctx = be._context(eng, 0)
                out = torch.empty(feats.shape[0], cfg.enc_frames, cfg.d_model, dtype=torch.float16, device="cuda")
                be._bind(ctx, "feats", feats.contiguous(), True)
                be._bind(ctx, "enc", out, False)
                be._execute(ctx, "encoder")
                torch.cuda.synchronize()
                print(f"variant {name} {flags}:")
                err("tensorrt", out, ref)
        finally:
            for k, v in saved.items():
                setattr(Precision, k, v)


if __name__ == "__main__":
    main()
