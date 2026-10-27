"""Export Whisper to ONNX and build TensorRT engines (once per GPU / TensorRT version; cached).

* ``encoder.engine``: features [W, n_mels, 3000] (fp32) -> every decoder layer's cross-attention keys/values
  [L, 2, W, H, T, d_k] (fp16). Profiles: one window, and 2 .. max_batch windows (kernels tuned for many
  windows waste much of the GPU on one).
* ``decoder.engine``: ``modeling.Decoder.step``: tokens [N, S], offset [N], the slot table src [N, C], the
  per-layer self-attention cache [N, H, C, d_k] and cross-attention keys/values [W, H, T, d_k] (rows grouped
  by window: N = W * K) -> logits [N, S, V] (fp32), new keys/values [L, 2, N, H, S, d_k]. Profiles:
  single-token steps of one row and of 2 .. max_rows rows, and prompts (S up to max_prefill). Self- and
  cross-attention are Triton plugins (attn_plugin) where they can be built.
"""

import hashlib
import json
import logging
import shutil
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from . import attn_plugin
from .config import N_FRAMES, WhisperConfig
from .modeling import Attention, Precision, WhisperModel, load_model
from .runtime import trt_logger
from .search import cache_buckets

log = logging.getLogger(__name__)

# Bump whenever the exported graphs change, so that cached engines of older exports are not reused.
EXPORT_VERSION = 4
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class BuildOptions:
    dtype: str = "float16"
    max_batch: int = 16  # windows per encoder call / per decoding
    max_rows: int = 80  # decoder rows per call (windows x beams)
    capacity: int = 448  # decoder positions (KV-cache slots)
    max_prefill: int = 256  # longest prompt (with previous text) in one call
    workspace_gb: float = 8.0
    opt_level: int = 3
    attn_plugin: bool = True  # decoder attention as Triton plugins (needs Triton at build time only)
    # cross-attention split over the keys (attn_plugin.cross_attn_split): 8 times as many programs
    cross_split: bool = True
    decoder_opt_level: int = 3  # TensorRT builder optimization level of the decoder engine
    # parts of the fp16 graphs computed in fp32 (modeling.Precision)
    ln32: bool = True
    gelu32_encoder: bool = True
    gelu32_decoder: bool = True


class _EncoderGraph(nn.Module):
    def __init__(self, model: WhisperModel):
        super().__init__()
        self.encoder, self.decoder = model.encoder, model.decoder

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.decoder.cross_kv(self.encoder(feats))


class _DecoderGraph(nn.Module):
    def __init__(self, model: WhisperModel):
        super().__init__()
        self.decoder = model.decoder
        self.n = model.cfg.dec_layers

    def forward(self, tokens: torch.Tensor, offset: torch.Tensor, src: torch.Tensor, *caches: torch.Tensor):
        n = self.n
        sk, sv, ck, cv = (list(caches[i * n : (i + 1) * n]) for i in range(4))
        return self.decoder.step(tokens, offset, sk, sv, ck, cv, src, explicit=True)


def decoder_input_names(n_layers: int) -> list[str]:
    names = ["tokens", "offset", "src"]
    for kind in ("self_k", "self_v", "cross_k", "cross_v"):
        names += [f"{kind}_{i}" for i in range(n_layers)]
    return names


@contextmanager
def _quiet_exporter():
    logger = logging.getLogger("torch.onnx")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logger.setLevel(level)


@contextmanager
def _precision(ln32: bool, gelu32: bool):
    saved = Precision.ln32, Precision.gelu32
    Precision.ln32, Precision.gelu32 = ln32, gelu32
    try:
        yield
    finally:
        Precision.ln32, Precision.gelu32 = saved


@contextmanager
def _explicit_attention():
    Attention.explicit = True  # the scale on the query (see modeling.attend)
    try:
        yield
    finally:
        Attention.explicit = False


def export_encoder_onnx(model: WhisperModel, path: Path, opts: BuildOptions) -> Path:
    cfg = model.cfg
    path.parent.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    batch = torch.export.Dim("batch", min=1, max=opts.max_batch)
    feats = torch.randn(2, cfg.n_mels, N_FRAMES, device=dev)
    with _explicit_attention(), _precision(opts.ln32, opts.gelu32_encoder), _quiet_exporter():
        torch.onnx.export(
            _EncoderGraph(model).eval(),
            (feats,),
            path,
            input_names=["feats"],
            output_names=["cross_kv"],
            dynamic_shapes={"feats": {0: batch}},
            dynamo=True,
            verbose=False,
            external_data=True,
        )
    return path


def export_decoder_onnx(model: WhisperModel, path: Path, opts: BuildOptions, plugins: bool = False) -> Path:
    """The decoder call as ONNX; ``plugins``: attention as attn_plugin nodes (registered by the caller)."""
    cfg = model.cfg
    path.parent.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    dtype = DTYPES[opts.dtype]
    H, dk, T = cfg.dec_heads, cfg.head_dim, cfg.enc_frames
    rows = torch.export.Dim("rows", min=1, max=opts.max_rows)
    windows = torch.export.Dim("windows", min=1, max=opts.max_batch)
    seq = torch.export.Dim("seq", min=1, max=opts.max_prefill)
    cache = torch.export.Dim("cache", min=16, max=opts.capacity)
    n, w, s = 4, 2, 3  # sizes > 1 so that the exporter does not specialise them
    tokens = torch.zeros(n, s, dtype=torch.long, device=dev)
    offset = torch.zeros(n, dtype=torch.long, device=dev)
    src = torch.arange(n, dtype=torch.int32, device=dev)[:, None].repeat(1, 32)
    caches = [torch.zeros(n, H, 32, dk, device=dev, dtype=dtype) for _ in range(2 * cfg.dec_layers)]
    caches += [torch.zeros(w, H, T, dk, device=dev, dtype=dtype) for _ in range(2 * cfg.dec_layers)]
    n_self = 2 * cfg.dec_layers
    cache_shapes = tuple({0: rows, 2: cache} if i < n_self else {0: windows} for i in range(len(caches)))
    shapes = {"tokens": {0: rows, 1: seq}, "offset": {0: rows}, "src": {0: rows, 1: cache}, "caches": cache_shapes}
    if plugins:
        for layer in model.decoder.layers:
            layer.self_attn.fuse_qkv()
    model.decoder.plugins = frozenset(
        ({"self", "cross"} | ({"cross_split"} if opts.cross_split else set())) if plugins else ()
    )
    try:
        with _explicit_attention(), _precision(opts.ln32, opts.gelu32_decoder), _quiet_exporter():
            torch.onnx.export(
                _DecoderGraph(model).eval(),
                (tokens, offset, src, *caches),
                path,
                input_names=decoder_input_names(cfg.dec_layers),
                output_names=["logits", "new_kv"],
                dynamic_shapes=shapes,
                dynamo=True,
                verbose=False,
                external_data=True,
                custom_translation_table=attn_plugin.onnx_translations() if plugins else None,
            )
    finally:
        model.decoder.plugins = frozenset()
        for layer in model.decoder.layers:
            if hasattr(layer.self_attn, "qkv_proj"):
                del layer.self_attn.qkv_proj
    return path


def _build(
    onnx_path: Path, engine_path: Path, profiles: list[dict], opts: BuildOptions, opt_level: int | None = None
) -> None:
    import tensorrt as trt

    logger = trt_logger(trt)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    if hasattr(trt.NetworkDefinitionCreationFlag, "PREFER_AOT_PYTHON_PLUGINS"):  # plugins in the engine
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(opts.workspace_gb * (1 << 30)))
    config.builder_optimization_level = opts.opt_level if opt_level is None else opt_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # layer names for profiling; no runtime cost
    dynamic = {
        network.get_input(i).name for i in range(network.num_inputs) if -1 in tuple(network.get_input(i).shape)
    }
    for prof in profiles:
        p = builder.create_optimization_profile()
        for name in dynamic:
            p.set_shape(name, *prof[name])
        config.add_optimization_profile(p)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_path.name}")
    engine_path.write_bytes(serialized)
    log.warning("built %s in %.0fs", engine_path.name, time.time() - t0)


def encoder_profiles(opts: BuildOptions) -> list[dict]:
    out = [{"batch": (1, 1, 1)}]
    if opts.max_batch > 1:
        out.append({"batch": (2, max(2, opts.max_batch // 2), opts.max_batch)})
    return out


def decoder_profiles(opts: BuildOptions) -> list[dict]:
    """(min, opt, max) rows, windows and new tokens per call; the index is the TensorRT profile."""
    R, W = opts.max_rows, opts.max_batch
    return [
        {"kind": "decode", "batch": (1, 1, 1), "windows": (1, 1, 1), "seq": (1, 1, 1)},
        {"kind": "decode", "batch": (2, 40, R), "windows": (1, 8, W), "seq": (1, 1, 1)},
        {"kind": "prefill", "batch": (1, 8, R), "windows": (1, 8, W), "seq": (1, 4, opts.max_prefill)},
    ]


def decoder_profile_shapes(cfg: WhisperConfig, opts: BuildOptions, prof: dict) -> dict:
    H, dk, T = cfg.dec_heads, cfg.head_dim, cfg.enc_frames
    n, w, s = prof["batch"], prof["windows"], prof["seq"]
    smallest = 16
    c = (smallest, cache_buckets(opts.capacity)[0], opts.capacity)
    shapes = {"tokens": tuple(zip(n, s)), "offset": tuple((x,) for x in n), "src": tuple(zip(n, c))}
    for i in range(cfg.dec_layers):
        for kind in ("self_k", "self_v"):
            shapes[f"{kind}_{i}"] = tuple((x, H, y, dk) for x, y in zip(n, c))
        for kind in ("cross_k", "cross_v"):
            shapes[f"{kind}_{i}"] = tuple((x, H, T, dk) for x in w)
    return shapes


def build_engines(model: WhisperModel, engine_dir: Path, opts: BuildOptions, keep_onnx: bool = False) -> None:
    cfg = model.cfg
    engine_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = engine_dir / "onnx"
    t0 = time.time()
    enc_onnx = export_encoder_onnx(model, onnx_dir / "encoder" / "encoder.onnx", opts)
    plugins = opts.attn_plugin and attn_plugin.available(cfg.head_dim)
    if opts.attn_plugin and not plugins:
        log.warning("decoder engine without attention plugins (slower): %s",
                    attn_plugin.unavailable() or f"no kernels for head size {cfg.head_dim}")
    if plugins:
        attn_plugin.register_plugins(cfg.dec_heads, cfg.head_dim)
    dec_onnx = export_decoder_onnx(model, onnx_dir / "decoder" / "decoder.onnx", opts, plugins)
    log.warning("exported ONNX in %.0fs", time.time() - t0)
    enc = encoder_profiles(opts)
    _build(enc_onnx, engine_dir / "encoder.engine",
           [{"feats": tuple((n, cfg.n_mels, N_FRAMES) for n in p["batch"])} for p in enc], opts)
    dec = decoder_profiles(opts)
    _build(dec_onnx, engine_dir / "decoder.engine", [decoder_profile_shapes(cfg, opts, p) for p in dec], opts,
           opts.decoder_opt_level)
    meta = {"options": asdict(opts), "config": asdict(cfg), "profiles": {"encoder": enc, "decoder": dec},
            "attn_plugin": plugins}
    (engine_dir / "build.json").write_text(json.dumps(meta, indent=1))
    if not keep_onnx:
        shutil.rmtree(onnx_dir, ignore_errors=True)


def engine_tag(opts: BuildOptions) -> str:
    """Directory name identifying an engine build: engines are GPU- and TensorRT-version-specific."""
    import tensorrt as trt

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu = props.name.replace(" ", "_").replace("/", "_")
    digest = hashlib.sha1(json.dumps(asdict(opts), sort_keys=True).encode()).hexdigest()[:8]
    return f"v{EXPORT_VERSION}-{gpu}-sm{props.major}{props.minor}-trt{trt.__version__}-{opts.dtype}-{digest}"


def cache_root(cache_dir: str | Path | None = None) -> Path:
    import os

    return Path(cache_dir or os.environ.get("WHISPER_TRT_CACHE") or Path.home() / ".cache" / "whisper_trt")


def model_name(model_dir: Path) -> str:
    """A directory name for a checkpoint: ``org--name`` for a Hugging Face snapshot, else the directory's name
    and a digest of its path."""
    path = Path(model_dir).resolve()
    for part in path.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--") + "-" + path.name[:8]
    return f"{path.name}-{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"


def ensure_engines(
    model_dir: Path, cfg: WhisperConfig, opts: BuildOptions, cache_dir: str | Path | None = None
) -> Path:
    """The engine directory for this checkpoint and options, built if missing."""
    engine_dir = cache_root(cache_dir) / model_name(model_dir) / engine_tag(opts)
    meta = engine_dir / "build.json"
    if meta.exists():
        return engine_dir
    log.warning("building TensorRT engines into %s (first use; this takes several minutes)", engine_dir)
    model = load_model(model_dir, cfg, dtype=DTYPES[opts.dtype])
    tmp = engine_dir.with_name(engine_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    build_engines(model, tmp, opts)
    del model
    torch.cuda.empty_cache()
    tmp.rename(engine_dir)
    return engine_dir
