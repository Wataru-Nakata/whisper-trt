"""FP8 (E4M3) linear layers in the encoder engine (opt-in, Hopper or newer; after owsm-trt's).

The encoder's linear layers (and the decoder layers' cross-attention key/value projections, which the encoder
engine computes) get quantize/dequantize pairs on their input (one scale per tensor, from calibration) and
weight (one scale per output channel). Exported to ONNX as QuantizeLinear/DequantizeLinear, they let TensorRT
run those GEMMs in FP8; everything else stays as the fp16 engine computes it.

The input ranges come from a scale file computed once offline on audio disjoint from the evaluation sets
(scripts/fp8_calibrate.py); the package ships one for whisper-large-v3 in ``whisper_trt/fp8_scales``.
"""

import hashlib
import json
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

E4M3_MAX = 448.0
SCALES_DIR = Path(__file__).parent / "fp8_scales"  # scale files shipped with the package
log = logging.getLogger(__name__)


@torch.library.custom_op("whisper_trt::fp8_qdq", mutates_args=())
def fp8_qdq(x: torch.Tensor, scale: torch.Tensor, axis: int) -> torch.Tensor:
    """Round ``x / scale`` to FP8 E4M3 and back. ``scale`` is a scalar, or per slice along ``axis``."""
    if scale.dim() == 1:
        shape = [1] * x.dim()
        shape[axis] = -1
        scale = scale.view(shape)
    return ((x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(x.dtype) * scale).contiguous()


@fp8_qdq.register_fake
def _(x: torch.Tensor, scale: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.empty_like(x)


def _onnx_fp8_qdq(x, scale, axis: int):
    import onnx
    from onnxscript import opset20 as op

    zero = op.Constant(value=onnx.helper.make_tensor("zero", onnx.TensorProto.FLOAT8E4M3FN, [], [0.0]))
    if len(scale.shape) == 1:
        zero = op.Expand(zero, op.Shape(scale))
        q = op.QuantizeLinear(x, scale, zero, axis=axis)
        return op.DequantizeLinear(q, scale, zero, axis=axis)
    return op.DequantizeLinear(op.QuantizeLinear(x, scale, zero), scale, zero)


ONNX_TRANSLATIONS = {torch.ops.whisper_trt.fp8_qdq.default: _onnx_fp8_qdq}


class QLinear(nn.Module):
    """``nn.Linear`` with FP8 input (per tensor) and weight (per output channel)."""

    def __init__(self, lin: nn.Linear, input_amax: float):
        super().__init__()
        self.lin = lin
        w_amax = lin.weight.detach().abs().amax(dim=1).float().clamp(min=1e-8)
        self.register_buffer("w_scale", (w_amax / E4M3_MAX).to(lin.weight.dtype))
        self.register_buffer("x_scale", torch.tensor(max(input_amax, 1e-8) / E4M3_MAX, dtype=lin.weight.dtype))

    @property
    def weight(self) -> torch.Tensor:
        return self.lin.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.lin.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fp8_qdq(self.lin.weight, self.w_scale, 0)
        return F.linear(fp8_qdq(x.to(w.dtype), self.x_scale, 0), w, self.lin.bias)


def quantizable(model: nn.Module) -> dict[str, nn.Linear]:
    """The linear layers the encoder engine computes: every encoder layer's, and the decoder layers'
    cross-attention key/value projections (by their names in ``model``, a modeling.WhisperModel)."""
    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if name.startswith("encoder.") or re.fullmatch(r"decoder\.layers\.\d+\.encoder_attn\.[kv]_proj", name):
            out[name] = m
    return out


@torch.no_grad()
def calibrate(model: nn.Module, run, batches) -> dict[str, float]:
    """Largest absolute input of every quantizable layer of ``model`` while ``run(batch)`` runs."""
    amax: dict[str, float] = {}
    hooks = []
    for name, module in quantizable(model).items():

        def hook(mod, args, name=name):
            a = args[0].detach().abs().amax().float().item()
            amax[name] = max(amax.get(name, 0.0), a)

        hooks.append(module.register_forward_pre_hook(hook))
    try:
        for batch in batches:
            run(batch)
    finally:
        for h in hooks:
            h.remove()
    return amax


@contextmanager
def quantized(model: nn.Module, amax: dict[str, float], skip: str = ""):
    """``model`` with its calibrated layers replaced by their FP8 versions (put back on exit); layers whose
    name matches the regular expression ``skip`` stay fp16."""
    replaced = []
    try:
        for name, a in amax.items():
            if skip and re.search(skip, name):
                continue
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            module = getattr(parent, attr)
            setattr(parent, attr, QLinear(module, a))
            replaced.append((parent, attr, module))
        yield len(replaced)
    finally:
        for parent, attr, module in replaced:
            setattr(parent, attr, module)


# --- scale files ------------------------------------------------------------------------------------


def save_scales(path: str | os.PathLike, amax: dict[str, float], model: str, checkpoint: str, about: dict) -> None:
    """A scale file: the input ranges of the quantizable layers for one checkpoint (its sha256)."""
    data = {"format": 1, "model": model, "checkpoint_sha256": checkpoint, "calibration": about,
            "amax": {k: float(v) for k, v in sorted(amax.items())}}  # fmt: skip
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


def load_scales(path: str | os.PathLike) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != 1 or not isinstance(data.get("amax"), dict):
        raise ValueError(f"{path} is not a whisper_trt FP8 scale file")
    return data


def scales_digest(path: str | os.PathLike) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def packaged_scales(checkpoint: str) -> Path | None:
    """The scale file shipped with the package for the checkpoint with this sha256, if any."""
    for p in sorted(SCALES_DIR.glob("*.json")):
        try:
            if load_scales(p).get("checkpoint_sha256") == checkpoint:
                return p
        except (ValueError, json.JSONDecodeError):
            log.warning("ignoring the malformed FP8 scale file %s", p)
    return None


def checkpoint_sha256(path: str | os.PathLike) -> str:
    """The checkpoint's sha256: the blob name of a Hugging Face cache file, else hashed."""
    real = Path(os.path.realpath(path))
    if real.parent.name == "blobs" and re.fullmatch(r"[0-9a-f]{64}", real.name):
        return real.name
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()
