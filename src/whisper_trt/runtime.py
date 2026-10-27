"""Execution backends: the encoder (log-Mel features -> every decoder layer's cross-attention keys/values) and
decoder calls on caller-owned buffers, the unit :class:`search.Search` captures in CUDA graphs.

* ``encode(feats, cross)``: feats [W, n_mels, 3000] fp32 -> cross [L, 2, W, H, T, d_k]
* ``step_fn(tokens, offset, cache, logits, new_kv, src, cross)``: a closure running one decoder call (see
  ``modeling.Decoder.step``) of tokens [N, S] at ``offset`` [N] against the self-attention cache
  [L, 2, N, H, C, d_k] through the slot table ``src`` [N, C], writing logits [N, S, V] (fp32) and the new
  keys/values [L, 2, N, H, S, d_k].
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import torch

from .config import WhisperConfig
from .modeling import WhisperModel

log = logging.getLogger(__name__)


class TorchBackend:
    name = "torch"

    def __init__(self, model: WhisperModel):
        self.model = model
        self.cfg = model.cfg
        self.dtype = model.decoder.embed_tokens.weight.dtype

    def encode(self, feats: torch.Tensor, cross: torch.Tensor) -> None:
        with torch.inference_mode():
            enc = self.model.encoder(feats)
            cross.copy_(self.model.decoder.cross_kv(enc))

    def step_fn(
        self,
        tokens: torch.Tensor,
        offset: torch.Tensor,
        cache: torch.Tensor,
        logits: torch.Tensor,
        new_kv: torch.Tensor,
        src: torch.Tensor,
        cross: torch.Tensor,
    ) -> Callable[[], None]:
        L = self.cfg.dec_layers
        self_k, self_v = [cache[i, 0] for i in range(L)], [cache[i, 1] for i in range(L)]
        cross_k, cross_v = [cross[i, 0] for i in range(L)], [cross[i, 1] for i in range(L)]
        dec = self.model.decoder

        def run() -> None:
            with torch.inference_mode():
                out, kv = dec.step(tokens, offset, self_k, self_v, cross_k, cross_v, src)
                logits.copy_(out)
                new_kv.copy_(kv)

        return run


def trt_logger(trt):
    """A TensorRT logger for warnings and errors (to whisper_trt's logging)."""

    class Logger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            if severity == trt.ILogger.Severity.WARNING:
                log.warning("TensorRT: %s", msg)
            elif severity in (trt.ILogger.Severity.ERROR, trt.ILogger.Severity.INTERNAL_ERROR):
                log.error("TensorRT: %s", msg)

    return Logger()


class TRTBackend:
    """The engines of a model: encoder (features -> cross-attention keys/values of every layer) and decoder
    (one call of S tokens per row against the static cache; single-token steps and prompt prefill)."""

    name = "tensorrt"

    def __init__(self, cfg: WhisperConfig, engine_dir: Path):
        import tensorrt as trt

        self.trt = trt
        self.cfg = cfg
        self.dtype = torch.float16
        self.logger = trt_logger(trt)
        self.runtime = trt.Runtime(self.logger)
        self.encoder = self._load(engine_dir / "encoder.engine")
        self.decoder = self._load(engine_dir / "decoder.engine")
        self.profiles = json.loads((engine_dir / "build.json").read_text())["profiles"]
        self._scratch: dict[int, torch.Tensor] = {}
        self._enc_ctxs: dict[int, object] = {}

    def _load(self, path: Path):
        engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize {path}; rebuild it with `whisper-trt build`")
        return engine

    def _context(self, engine, profile: int):
        # Every context of an engine shares one scratch buffer: they run on one stream, one at a time.
        ctx = engine.create_execution_context(self.trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        if ctx is None:
            raise RuntimeError(f"TensorRT failed to create an execution context (profile {profile})")
        key = id(engine)
        if key not in self._scratch:
            self._scratch[key] = torch.empty(max(engine.device_memory_size_v2, 1), dtype=torch.uint8,
                                             device="cuda")
        scratch = self._scratch[key]
        ctx.set_device_memory(scratch.data_ptr(), scratch.numel())
        stream = torch.cuda.current_stream()
        ctx.set_optimization_profile_async(profile, stream.cuda_stream)
        stream.synchronize()
        return ctx

    @staticmethod
    def _bind(ctx, name: str, t: torch.Tensor, is_input: bool) -> None:
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if is_input:
            ctx.set_input_shape(name, tuple(t.shape))
        ctx.set_tensor_address(name, t.data_ptr())

    @staticmethod
    def _execute(ctx, what: str) -> None:
        if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(f"{what} engine execution failed")

    def _profile(self, kind: str, rows: int) -> int:
        for i, p in enumerate(self.profiles[kind]):
            if p["batch"][0] <= rows <= p["batch"][2]:
                return i
        raise ValueError(f"no {kind} profile covers {rows} rows")

    def encode(self, feats: torch.Tensor, cross: torch.Tensor) -> None:
        w = feats.shape[0]
        i = self._profile("encoder", w)
        if i not in self._enc_ctxs:
            self._enc_ctxs[i] = self._context(self.encoder, i)
        ctx = self._enc_ctxs[i]
        self._bind(ctx, "feats", feats.contiguous(), True)
        self._bind(ctx, "cross_kv", cross, False)
        self._execute(ctx, "encoder")

    def step_fn(
        self,
        tokens: torch.Tensor,
        offset: torch.Tensor,
        cache: torch.Tensor,
        logits: torch.Tensor,
        new_kv: torch.Tensor,
        src: torch.Tensor,
        cross: torch.Tensor,
    ) -> Callable[[], None]:
        # Each closure has a context of its own, so that a captured CUDA graph never sees its shapes change.
        kind = "decoder"
        ctx = self._context(self.decoder, self._decoder_profile(tokens.shape[0], tokens.shape[1]))
        self._bind(ctx, "tokens", tokens, True)
        self._bind(ctx, "offset", offset, True)
        self._bind(ctx, "src", src, True)
        for i in range(self.cfg.dec_layers):
            self._bind(ctx, f"self_k_{i}", cache[i, 0], True)
            self._bind(ctx, f"self_v_{i}", cache[i, 1], True)
            self._bind(ctx, f"cross_k_{i}", cross[i, 0], True)
            self._bind(ctx, f"cross_v_{i}", cross[i, 1], True)
        self._bind(ctx, "logits", logits, False)
        self._bind(ctx, "new_kv", new_kv, False)

        def run() -> None:
            self._execute(ctx, kind)

        run.ctx = ctx  # keep the context alive as long as the closure
        return run

    def _decoder_profile(self, rows: int, seq: int) -> int:
        for i, p in enumerate(self.profiles["decoder"]):
            if p["batch"][0] <= rows <= p["batch"][2] and p["seq"][0] <= seq <= p["seq"][2]:
                return i
        raise ValueError(f"no decoder profile covers {rows} rows x {seq} tokens")

