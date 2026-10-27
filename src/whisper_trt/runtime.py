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

    def __init__(self, model: WhisperModel, kernels: bool = False):
        """``kernels``: the decoder's attention through the Triton kernels of attn_plugin (fused Q/K/V, the slot
        table read in place, split cross-attention) instead of the PyTorch reference, for speed."""
        self.model = model
        self.cfg = model.cfg
        self.dtype = model.decoder.embed_tokens.weight.dtype
        if kernels:
            from . import attn_plugin

            if attn_plugin.triton is None or not torch.cuda.is_available():
                raise RuntimeError("the decoder kernels need CUDA and Triton")
            for layer in model.decoder.layers:
                layer.self_attn.fuse_qkv()
            model.decoder.plugins = frozenset({"self", "cross", "cross_split", "logits"})
            attn_plugin.RUNTIME_TRITON = True
            self.name = "torch+kernels"

    def encode(self, feats: torch.Tensor, cross: torch.Tensor, lane: int = 0) -> None:
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


class TritonDecoderBackend:
    """The TensorRT encoder with a decoder step of cuBLAS GEMMs and Triton kernels (decoder_kernels, attn_plugin),
    every GEMM accumulating in fp32: per layer, a LayerNorm kernel that also adds the bias of the next residual
    GEMM to the residual stream, which that GEMM then adds its product to (``x.addmm_``); QKV; self-attention of
    one query per (row, head) through the slot table, which also writes the new keys/values into the cache (the
    closure's ``writes_cache``); the output projection; LayerNorm, query, split cross-attention and its output
    projection; LayerNorm, fc1, GELU (fp32) and fc2; then the final LayerNorm and the vocabulary projection.
    2.4 against the TensorRT engine's 3.2 ms a step at 8 windows x 5 beams, 4.8 against 5.9 at 32 x 5 on an
    H200 (scripts/bench_decoder.py), and logits nearer fp32's. The GEMMs need
    ``torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`` off (fp32 split-K sums): this turns it
    off for the process. ``model``: its decoder's weights (on the GPU) are used.

    ``gemm="triton"``: fused Triton LN+GEMM and GEMM+residual kernels instead of cuBLAS (ten kernels a layer;
    slower: Triton's small-M GEMMs re-read the weights), for reference."""

    name = "triton"

    def __init__(self, model: WhisperModel, encoder, gemm: str = "cublas"):
        from . import attn_plugin, decoder_kernels

        if decoder_kernels.triton is None:
            raise RuntimeError("the Triton decoder needs Triton")
        if gemm not in ("triton", "cublas"):
            raise ValueError(f"unknown GEMM kind {gemm!r}")
        self.k, self.attn, self.gemm = decoder_kernels, attn_plugin, gemm
        if gemm == "cublas":
            self.name = "triton+cublas"
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # (fp32 split-K sums)
        self.cfg = model.cfg
        self.dtype = torch.float16
        self.encoder = encoder
        dec = model.decoder
        self.embed, self.pos = dec.embed_tokens.weight, dec.embed_positions.weight
        self.final = (dec.layer_norm.weight, dec.layer_norm.bias)
        self.layers = []
        for layer in dec.layers:
            a, x = layer.self_attn, layer.encoder_attn
            qkv_w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).contiguous()
            qkv_b = torch.cat([a.q_proj.bias, torch.zeros_like(a.q_proj.bias), a.v_proj.bias]).contiguous()
            self.layers.append({
                "ln1": (layer.self_attn_layer_norm.weight, layer.self_attn_layer_norm.bias), "qkv": (qkv_w, qkv_b),
                "out": (a.out_proj.weight, a.out_proj.bias),
                "ln2": (layer.encoder_attn_layer_norm.weight, layer.encoder_attn_layer_norm.bias),
                "q": (x.q_proj.weight, x.q_proj.bias), "xout": (x.out_proj.weight, x.out_proj.bias),
                "ln3": (layer.final_layer_norm.weight, layer.final_layer_norm.bias),
                "fc1": (layer.fc1.weight, layer.fc1.bias), "fc2": (layer.fc2.weight, layer.fc2.bias),
            })

    def encode(self, feats: torch.Tensor, cross: torch.Tensor, lane: int = 0) -> None:
        self.encoder.encode(feats, cross, lane)

    def step_fn(self, tokens, offset, cache, logits, new_kv, src, cross) -> Callable[[], None]:
        if tokens.shape[1] != 1:
            raise ValueError("the Triton decoder runs single-token steps")
        k, attn, cfg = self.k, self.attn, self.cfg
        n, d, H, D = tokens.shape[0], cfg.d_model, cfg.dec_heads, cfg.head_dim
        w = cross.shape[2]

        step_cfg = attn.self_attn_step_config(n)

        def run_cublas() -> None:
            x = (self.embed[tokens[:, 0]] + self.pos[offset]).contiguous()  # [N, d]
            for i, p in enumerate(self.layers):
                h = k.layer_norm_bias(x, *p["ln1"], p["out"][1])
                qkv = torch.addmm(p["qkv"][1], h, p["qkv"][0].t())  # [N, 3d]
                y = attn.self_attn_step_triton(qkv.view(n, 1, 3 * d), cache[i, 0], cache[i, 1], offset, src,
                                               step_cfg, write_cache=True)
                x.addmm_(y.view(n, d), p["out"][0].t())
                h = k.layer_norm_bias(x, *p["ln2"], p["xout"][1])
                qw = torch.addmm(p["q"][1], h, p["q"][0].t()).view(w, n // w, d)
                part = attn.cross_attn_split_triton(qw, cross[i, 0], cross[i, 1])
                a = attn.cross_attn_combine_triton(part, qw)
                x.addmm_(a.view(n, d), p["xout"][0].t())
                h = k.layer_norm_bias(x, *p["ln3"], p["fc2"][1])
                g = torch.nn.functional.gelu(torch.addmm(p["fc1"][1], h, p["fc1"][0].t()))
                x.addmm_(g, p["fc2"][0].t())
            xf = torch.nn.functional.layer_norm(x.float(), (d,), self.final[0].float(), self.final[1].float())
            attn.logits32_triton(xf.half().view(n, 1, d), self.embed, out=logits)

        def run() -> None:
            x = (self.embed[tokens[:, 0]] + self.pos[offset]).contiguous()  # [N, d]
            for i, p in enumerate(self.layers):
                qkv = k.ln_linear(x, *p["ln1"], *p["qkv"])  # [N, 3d]
                new_kv[i].copy_(qkv.view(n, 3, H, 1, D)[:, 1:].permute(1, 0, 2, 3, 4))
                y = attn.self_attn_triton(qkv.view(n, 1, 3 * d), cache[i, 0], cache[i, 1], offset, src)
                k.linear_residual(y.view(n, d), *p["out"], x, out=x)
                q = k.ln_linear(x, *p["ln2"], *p["q"])
                qw = q.view(w, n // w, d)
                part = attn.cross_attn_split_triton(qw, cross[i, 0], cross[i, 1])
                a = attn.cross_attn_combine_triton(part, qw)
                k.linear_residual(a.view(n, d), *p["xout"], x, out=x)
                h = k.ln_linear(x, *p["ln3"], *p["fc1"], gelu=True)
                k.linear_residual(h, *p["fc2"], x, out=x)
            xf = torch.nn.functional.layer_norm(x.float(), (d,), self.final[0].float(), self.final[1].float())
            attn.logits32_triton(xf.half().view(n, 1, d), self.embed, out=logits)

        if self.gemm == "cublas":
            run = run_cublas
            run.writes_cache = True  # (the new keys/values are in the cache at offset, not in new_kv)
        run.keep = (tokens, offset, cache, logits, new_kv, src, cross)
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

    def __init__(self, cfg: WhisperConfig, engine_dir: Path, decoder: bool = True):
        """``decoder=False``: the encoder engine only (another backend decodes)."""
        import tensorrt as trt

        self.trt = trt
        self.cfg = cfg
        self.dtype = torch.float16
        self.logger = trt_logger(trt)
        self.runtime = trt.Runtime(self.logger)
        self.encoder = self._load(engine_dir / "encoder.engine")
        self.decoder = self._load(engine_dir / "decoder.engine") if decoder else None
        meta = json.loads((engine_dir / "build.json").read_text())
        self.profiles = meta["profiles"]
        self._scratch: dict[int, torch.Tensor] = {}
        self._enc_ctxs: dict[int, object] = {}

    def _load(self, path: Path):
        engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize {path}; rebuild it with `whisper-trt build`")
        return engine

    def _context(self, engine, profile: int, lane: int = 0):
        # Every context of an engine and lane shares one scratch buffer: they run on one stream, one at a time.
        # Contexts of another lane may run at the same time (on a stream of their own): scratch of their own.
        ctx = engine.create_execution_context(self.trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        if ctx is None:
            raise RuntimeError(f"TensorRT failed to create an execution context (profile {profile})")
        key = (id(engine), lane) if lane == 0 else (id(engine), lane, profile)
        if key not in self._scratch:  # (another lane: only what its profile needs, e.g. 256 MB for one window)
            size = engine.device_memory_size_v2 if lane == 0 else engine.get_device_memory_size_for_profile_v2(profile)
            self._scratch[key] = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
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

    def encode(self, feats: torch.Tensor, cross: torch.Tensor, lane: int = 0) -> None:
        """``lane``: contexts of their own, for encodes that may run alongside others (see _context)."""
        w = feats.shape[0]
        i = self._profile("encoder", w)
        if (i, lane) not in self._enc_ctxs:
            self._enc_ctxs[i, lane] = self._context(self.encoder, i, lane)
        ctx = self._enc_ctxs[i, lane]
        feats = feats.contiguous()
        self._bind(ctx, "feats", feats, True)
        self._bind(ctx, "cross_kv", cross, False)
        _check_outputs(ctx, {"cross_kv": cross})
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
        if self.decoder is None:
            raise RuntimeError("this TensorRT backend was loaded without its decoder engine")
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
        _check_outputs(ctx, {"logits": logits, "new_kv": new_kv})

        def run() -> None:
            self._execute(ctx, kind)

        # The context holds raw pointers: the closure keeps it and every bound tensor alive (a dropped
        # `offset` once let torch reuse its memory, and every later prefill read garbage offsets).
        run.keep = (ctx, tokens, offset, cache, logits, new_kv, src, cross)
        return run

    def _decoder_profile(self, rows: int, seq: int) -> int:
        for i, p in enumerate(self.profiles["decoder"]):
            if p["batch"][0] <= rows <= p["batch"][2] and p["seq"][0] <= seq <= p["seq"][2]:
                return i
        raise ValueError(f"no decoder profile covers {rows} rows x {seq} tokens")


def _check_outputs(ctx, outputs: dict[str, torch.Tensor]) -> None:
    """The output shapes the context will write, against the buffers bound to them (a bigger output would
    overwrite whatever follows the buffer)."""
    ctx.infer_shapes()
    for name, t in outputs.items():
        shape = tuple(ctx.get_tensor_shape(name))
        if shape != tuple(t.shape):
            raise ValueError(f"the engine writes {name} as {shape}, its buffer is {tuple(t.shape)}")
