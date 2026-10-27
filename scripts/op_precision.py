"""Single operations of the encoder as TensorRT fp16 engines against PyTorch fp16/fp32, with the checkpoint's
layer-0 weights and realistic inputs (the encoder's own activations), to see where TensorRT loses precision.

    python scripts/op_precision.py WAVES.npz
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from whisper_trt import hub
from whisper_trt.build import BuildOptions, _build, _quiet_exporter
from whisper_trt.config import N_FRAMES, load_config
from whisper_trt.features import log_mel
from whisper_trt.modeling import attend, load_model
from whisper_trt.runtime import TRTBackend, trt_logger


class Op(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def run_trt(fn, x: torch.Tensor, out_shape, opts) -> torch.Tensor:
    import tensorrt as trt

    with tempfile.TemporaryDirectory() as tmp:
        onnx = Path(tmp) / "op.onnx"
        with _quiet_exporter():
            torch.onnx.export(Op(fn).eval(), (x,), onnx, input_names=["x"], output_names=["y"], dynamo=True,
                              external_data=True)
        engine = Path(tmp) / "op.engine"
        _build(onnx, engine, [], opts)
        be = TRTBackend.__new__(TRTBackend)
        be.trt, be.runtime, be._scratch = trt, trt.Runtime(trt_logger(trt)), {}
        ctx = be._context(be._load(engine), 0)
        y = torch.empty(out_shape, dtype=torch.float16, device="cuda")
        be._bind(ctx, "x", x.contiguous(), True)
        be._bind(ctx, "y", y, False)
        be._execute(ctx, "op")
        torch.cuda.synchronize()
        return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = np.load(args.waves)
    model_dir = hub.resolve(args.model)
    cfg = load_config(model_dir)
    feats = log_mel([torch.from_numpy(data["w0"])], cfg.n_mels, "cuda", N_FRAMES)
    m32 = load_model(model_dir, cfg, dtype=torch.float32)
    m16 = load_model(model_dir, cfg, dtype=torch.float16)
    e32, e16 = m32.encoder, m16.encoder
    with torch.inference_mode():
        x0 = F.gelu(e32.conv2(F.gelu(e32.conv1(feats)))).transpose(1, 2) + e32.embed_positions.weight
        h = e32.layers[0].self_attn_layer_norm(x0)  # [1, 1500, d]
        u = e32.layers[0].fc1(e32.layers[0].final_layer_norm(x0))  # [1, 1500, 4d]
    a32, a16 = e32.layers[0].self_attn, e16.layers[0].self_attn
    l32, l16 = e32.layers[0], e16.layers[0]
    inputs = {"conv1+gelu": feats, "fc1 (1280->5120)": h, "fc2 (5120->1280)": F.gelu(u),
              "attention (q,k,v,out)": h, "layer_norm": x0}
    fns = {
        "conv1+gelu": (lambda x: F.gelu(e16.conv1(x)), lambda x: F.gelu(e32.conv1(x)), (1, cfg.d_model, N_FRAMES)),
        "fc1 (1280->5120)": (l16.fc1, l32.fc1, (1, 1500, cfg.enc_ffn)),
        "fc2 (5120->1280)": (l16.fc2, l32.fc2, (1, 1500, cfg.d_model)),
        "attention (q,k,v,out)": (lambda x: a16.merge(attend(a16.heads(a16.q_proj(x)), a16.heads(a16.k_proj(x)),
                                                           a16.heads(a16.v_proj(x)), explicit=True)),
                                  a32, (1, 1500, cfg.d_model)),
        "layer_norm": (l16.self_attn_layer_norm, l32.self_attn_layer_norm, (1, 1500, cfg.d_model)),
    }
    opts = BuildOptions()
    for name, (f16, f32, shape) in fns.items():
        x = inputs[name]
        with torch.inference_mode():
            ref = f32(x.float())
            t16 = f16(x.half())
        y = run_trt(f16, x.half().clone(), shape, opts)
        d_t, d_p = (y.float() - ref).abs(), (t16.float() - ref).abs()
        print(f"{name}: tensorrt mean |diff| {d_t.mean():.3g} max {d_t.max():.3g}; torch fp16 mean {d_p.mean():.3g} "
              f"max {d_p.max():.3g}; ref rms {ref.pow(2).mean().sqrt():.3g}", flush=True)


if __name__ == "__main__":
    main()
