"""Decoder attention as TensorRT plugins: Triton kernels compiled ahead of time.

(Ported from owsm-trt.) TensorRT's fused attention wants its own layouts: every layer it copies
``cat([cache, new])`` for self-attention and the whole cross-attention keys/values (transposed) before the
attention kernel. These kernels read everything where it is instead:

* :func:`self_attn`: the fused Q/K/V projection [B, S, 3·H·D] and the KV cache [B, H, C, D] of each row up
  to its own ``offset``, causal over the new tokens -> [B, S, H·D]. ``src`` [B, C] (int32) gives, for each
  cache slot of a row, the row whose cache holds it: beam search then never moves the cache. A new beam
  takes its parent's ``src`` row; the token it is fed next is written to its own row (see search.py).
* :func:`cross_attn`: queries [W, S, H·D] and the cross-attention keys/values [W, H, T, D] -> [W, S, H·D]
  (a window's beams are its K * S queries).
* :func:`cross_attn_split` + :func:`cross_attn_combine`: the same attention split over the T keys
  ("flash-decoding"): one program per (window, head) walks 1500 keys one block after another, which leaves
  most of the GPU idle with few windows (20 programs for one window); SPLITS programs per (window, head)
  each take a slice of the keys and write their partial softmax state, which a second kernel combines.

Both are torch custom ops (with a PyTorch reference implementation) that export to ONNX nodes of domain
``whisper_trt``; the TensorRT ONNX parser maps those to the plugins :func:`register_plugins` defines. The
plugins are ahead-of-time: the engine holds the kernels' PTX, so loading it needs neither Triton nor the
registration.
"""

import functools
import math
from typing import List, Tuple, Union

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the plugins cannot be built; the decoder is exported without them
    triton = None

NAMESPACE = "whisper_trt"
# Kernel configurations (BS queries per program -- tl.dot needs >= 16 --, BN keys per step, warps,
# pipeline stages), offered to TensorRT as plugin tactics: it times them and keeps the fastest.
TACTICS = {
    "self_attn": [(16, 64, 4, 2), (16, 32, 4, 2), (16, 128, 4, 2), (16, 64, 4, 3), (32, 64, 4, 2), (16, 64, 8, 2)],
    "cross_attn": [(16, 64, 4, 2), (16, 128, 4, 2), (16, 64, 4, 3), (16, 128, 8, 2), (32, 64, 4, 2), (16, 32, 4, 3)],
}  # fmt: skip
TACTICS["cross_split"] = [(16, 64, 4, 2), (16, 32, 4, 2), (16, 64, 4, 3), (16, 128, 4, 2), (32, 64, 4, 2)]
TACTICS["cross_combine"] = [(16, 0, 4, 1), (16, 0, 2, 1)]
SPLITS = 8  # key slices of the split cross-attention (1500 / 8 = 188 keys each)
TUNE = True  # False: only the first configuration (no tactic timing)
_SHAPE = {"heads": 0, "head_dim": 0}  # of the model being built (set by register_plugins)
_REGISTERED = False


def available(head_dim: int) -> bool:
    """Whether the plugins can be built here (Triton, TensorRT's ahead-of-time Python plugins, an Ampere or
    newer GPU) for this head size."""
    return head_dim in (16, 32, 64, 128) and unavailable() is None


@functools.cache
def unavailable() -> str | None:
    """Why this environment cannot build the plugins (None: it can)."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    if torch.cuda.get_device_capability() < (8, 0):
        return "they need an Ampere or newer GPU"
    if triton is None:
        return "Triton is not installed (it comes with PyTorch on Linux)"
    try:
        import tensorrt
        import tensorrt.plugin as trtp
    except ImportError as e:
        return f"TensorRT's Python plugin API does not import ({e})"
    if not hasattr(trtp, "aot_impl"):
        return f"TensorRT {tensorrt.__version__} has no ahead-of-time Python plugins (whisper-trt needs TensorRT >= 10.16)"
    return None


# --- PyTorch reference implementations, exported as ONNX nodes of domain "whisper_trt" ---


def _merge(x: torch.Tensor) -> torch.Tensor:  # [B, H, S, D] -> [B, S, H·D]
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)


def gather_cache(cache: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """The cache [B, H, C, D] as each row sees it: slot j of row b from row src[b, j]."""
    b, h, c, d = cache.shape
    rows = src.long()[:, None, :, None].expand(b, h, c, d)
    heads = torch.arange(h, device=cache.device)[None, :, None, None].expand(b, h, c, d)
    slots = torch.arange(c, device=cache.device)[None, None, :, None].expand(b, h, c, d)
    dims = torch.arange(d, device=cache.device)[None, None, None, :].expand(b, h, c, d)
    return cache[rows, heads, slots, dims]


@torch.library.custom_op("whisper_trt::self_attn", mutates_args=())
def self_attn(
    qkv: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, offset: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    """Self-attention of S new tokens per row over [cache slots < offset[b] | new tokens (causal)].

    qkv [B, S, 3·H·D] (queries, keys, values; heads in order), k_cache/v_cache [B, H, C, D],
    offset [B] int64, src [B, C] int32 (the row holding each slot) -> [B, S, H·D]"""
    b, s, _ = qkv.shape
    h, c, d = k_cache.shape[1:]
    k_cache, v_cache = gather_cache(k_cache, src), gather_cache(v_cache, src)
    q, k, v = qkv.view(b, s, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)
    new = torch.arange(s, device=qkv.device)
    cache_ok = (torch.arange(c, device=qkv.device) < offset[:, None])[:, None, None, :].expand(b, 1, s, c)
    mask = torch.cat([cache_ok, (new[None, :] <= new[:, None]).expand(b, 1, s, s)], -1)
    scores = (q.float() * d**-0.5) @ torch.cat([k_cache, k], 2).float().transpose(-1, -2)
    p = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    return _merge((p @ torch.cat([v_cache, v], 2).float()).to(qkv.dtype))


@self_attn.register_fake
def _(qkv, k_cache, v_cache, offset, src):
    return qkv.new_empty(qkv.shape[0], qkv.shape[1], k_cache.shape[1] * k_cache.shape[3])


@torch.library.custom_op("whisper_trt::cross_attn", mutates_args=())
def cross_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention of q [B, S, H·D] to k/v [B, H, T, D] -> [B, S, H·D]."""
    b, s, _ = q.shape
    h, d = k.shape[1], k.shape[3]
    qh = q.view(b, s, h, d).transpose(1, 2)
    p = ((qh.float() * d**-0.5) @ k.float().transpose(-1, -2)).softmax(-1)
    return _merge((p @ v.float()).to(q.dtype))


@cross_attn.register_fake
def _(q, k, v):
    return torch.empty_like(q)


def _log2_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    b, s, _ = q.shape
    h, d = k.shape[1], k.shape[3]
    qh = q.view(b, s, h, d).transpose(1, 2).float()  # [B, H, S, D]
    return (qh @ k.float().transpose(-1, -2)) * _qk_scale(d)  # [B, H, S, T], log2 units


@torch.library.custom_op("whisper_trt::cross_attn_split", mutates_args=())
def cross_attn_split(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Partial attention of q [B, S, H·D] over SPLITS slices of the keys k/v [B, H, T, D]: per slice the
    unnormalized output, the running maximum (log2 units) and the sum of exponentials ->
    [B, H, SPLITS, S, D + 2] (fp32)."""
    b, s, _ = q.shape
    h, t, d = k.shape[1:]
    scores = _log2_scores(q, k)
    chunk = -(-t // SPLITS)
    out = q.new_zeros(b, h, SPLITS, s, d + 2, dtype=torch.float32)
    for i in range(SPLITS):
        lo, hi = i * chunk, min((i + 1) * chunk, t)
        if lo >= hi:
            out[:, :, i, :, d] = float("-inf")
            continue
        sc = scores[..., lo:hi]
        m = sc.amax(-1, keepdim=True)
        p = torch.exp2(sc - m)
        out[:, :, i, :, :d] = p @ v[:, :, lo:hi].float()
        out[:, :, i, :, d] = m[..., 0]
        out[:, :, i, :, d + 1] = p.sum(-1)
    return out


@cross_attn_split.register_fake
def _(q, k, v):
    return q.new_empty(q.shape[0], k.shape[1], SPLITS, q.shape[1], k.shape[3] + 2, dtype=torch.float32)


@torch.library.custom_op("whisper_trt::cross_attn_combine", mutates_args=())
def cross_attn_combine(part: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The slices of :func:`cross_attn_split` combined -> [B, S, H·D] in ``q``'s type (``q`` only gives the
    type and shape)."""
    d = part.shape[-1] - 2
    m = part[..., d]  # [B, H, N, S]
    top = m.amax(2, keepdim=True)
    w = torch.exp2(m - top)  # slices without keys: weight 0
    num = (w[..., None] * part[..., :d]).sum(2)  # [B, H, S, D]
    den = (w * part[..., d + 1]).sum(2)
    return _merge((num / den[..., None]).to(q.dtype))


@cross_attn_combine.register_fake
def _(part, q):
    return torch.empty_like(q)


def _onnx_op(name: str, n_in: int, out: int = 0):
    """The ONNX node of a plugin with ``n_in`` inputs; its output has the type of input ``out``, or its own
    type for ``out=-1``."""
    import onnxscript
    from onnx.defs import OpSchema

    types = [f"tensor({t})" for t in ("float16", "bfloat16", "float", "int64", "int32")]
    out_type = "TO" if out < 0 else f"T{out}"
    constraints = [(f"T{i}", types, "") for i in range(n_in)] + ([("TO", types, "")] if out < 0 else [])
    schema = OpSchema(
        name,
        NAMESPACE,
        1,
        inputs=[OpSchema.FormalParameter(f"input{i}", f"T{i}") for i in range(n_in)],
        outputs=[OpSchema.FormalParameter("output", out_type)],
        type_constraints=constraints,
        attributes=[
            OpSchema.Attribute("plugin_namespace", OpSchema.AttrType.STRING, "TensorRT plugin namespace"),
            OpSchema.Attribute("plugin_version", OpSchema.AttrType.STRING, "TensorRT plugin version"),
        ],
    )
    op = onnxscript.values.Op(onnxscript.values.Opset(NAMESPACE, 1), name, schema)

    def translate(*args):
        return op(*args, plugin_namespace=NAMESPACE, plugin_version="1")

    return translate


def onnx_translations() -> dict:
    """``custom_translation_table`` entries for ``torch.onnx.export``."""
    ops = torch.ops.whisper_trt
    return {
        ops.self_attn.default: _onnx_op("self_attn", 5),
        ops.cross_attn.default: _onnx_op("cross_attn", 3),
        ops.cross_attn_split.default: _onnx_op("cross_attn_split", 3, out=-1),
        ops.cross_attn_combine.default: _onnx_op("cross_attn_combine", 2, out=1),
    }


# --- Triton kernels (AOT calling convention: inputs, scalar arguments, outputs, then constexprs) ---

if triton is not None:

    @triton.jit
    def _attend(q, k_ptrs, v_ptrs, valid, ok, m_i, l_i, acc, QK_SCALE: tl.constexpr):
        """One block of keys/values [BN, D] into the running (max, sum, output) of an online softmax.
        ``valid`` [BN, 1]: keys that exist; ``ok`` [BS or 1, BN]: keys each query attends to. Every
        query row must attend to at least one key of the first block it sees."""
        k = tl.load(k_ptrs, mask=valid, other=0.0)
        s = tl.dot(q, tl.trans(k)) * QK_SCALE  # log2 units
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        v = tl.load(v_ptrs, mask=valid, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        return m_new, l_i * alpha + tl.sum(p, 1), acc

    @triton.jit
    def _self_attn_kernel(qkv, kc, vc, offs, src, S, C, y,
                          H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr,
                          QK_SCALE: tl.constexpr):  # fmt: skip
        bh = tl.program_id(0)  # (row, head)
        sb = tl.program_id(1)  # block of queries
        b = bh // H
        h = bh % H
        HD: tl.constexpr = H * D
        # clamped: TensorRT times the plugin on arbitrary input data while building
        off = tl.minimum(tl.maximum(tl.load(offs + b), 0), C).to(tl.int32)
        s = sb * BS + tl.arange(0, BS)
        sm = (s < S)[:, None]
        d = tl.arange(0, D)[None, :]
        q = tl.load(qkv + (b * S + s).to(tl.int64)[:, None] * (3 * HD) + h * D + d, mask=sm, other=0.0)
        m_i = tl.full([BS], float("-inf"), tl.float32)
        l_i = tl.zeros([BS], tl.float32)
        acc = tl.zeros([BS, D], tl.float32)
        for j0 in range(0, off, BN):  # the row's cache: slots < off, each in the row src[b, j] holds it
            j = j0 + tl.arange(0, BN)
            row = tl.load(src + b.to(tl.int64) * C + j, mask=j < off, other=0)
            row = tl.minimum(tl.maximum(row, 0), tl.num_programs(0) // H - 1)  # (clamped, see off)
            ptr = ((row.to(tl.int64) * H + h) * C + j)[:, None] * D + d
            m_i, l_i, acc = _attend(
                q, kc + ptr, vc + ptr, (j < off)[:, None], (j < off)[None, :], m_i, l_i, acc, QK_SCALE
            )
        for j0 in range(0, (sb + 1) * BS, BS):  # the new tokens, causal
            j = j0 + tl.arange(0, BS)
            ok = (j < S)[None, :] & (j[None, :] <= s[:, None])
            new = qkv + (b * S + j).to(tl.int64)[:, None] * (3 * HD) + h * D + d
            m_i, l_i, acc = _attend(q, new + HD, new + 2 * HD, (j < S)[:, None], ok, m_i, l_i, acc, QK_SCALE)
        out = y + (b * S + s).to(tl.int64)[:, None] * HD + h * D + d
        tl.store(out, (acc / l_i[:, None]).to(y.dtype.element_ty), mask=sm)

    @triton.jit
    def _cross_attn_kernel(q_in, kx, vx, S, T, y,
                           H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr,
                           QK_SCALE: tl.constexpr):  # fmt: skip
        bh = tl.program_id(0)
        sb = tl.program_id(1)
        b = bh // H
        h = bh % H
        HD: tl.constexpr = H * D
        s = sb * BS + tl.arange(0, BS)
        sm = (s < S)[:, None]
        d = tl.arange(0, D)[None, :]
        rows = (b * S + s).to(tl.int64)[:, None] * HD + h * D + d
        q = tl.load(q_in + rows, mask=sm, other=0.0)
        m_i = tl.full([BS], float("-inf"), tl.float32)
        l_i = tl.zeros([BS], tl.float32)
        acc = tl.zeros([BS, D], tl.float32)
        base = (b * H + h).to(tl.int64) * T * D
        for j0 in range(0, T, BN):
            j = j0 + tl.arange(0, BN)
            ptr = base + j[:, None] * D + d
            m_i, l_i, acc = _attend(
                q, kx + ptr, vx + ptr, (j < T)[:, None], (j < T)[None, :], m_i, l_i, acc, QK_SCALE
            )
        tl.store(y + rows, (acc / l_i[:, None]).to(y.dtype.element_ty), mask=sm)

    @triton.jit
    def _cross_split_kernel(q_in, kx, vx, S, T, part,
                            H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr,
                            NS: tl.constexpr, QK_SCALE: tl.constexpr):  # fmt: skip
        bh = tl.program_id(0)
        sb = tl.program_id(1)
        sp = tl.program_id(2)  # key slice
        b = bh // H
        h = bh % H
        HD: tl.constexpr = H * D
        s = sb * BS + tl.arange(0, BS)
        sm = (s < S)[:, None]
        d = tl.arange(0, D)[None, :]
        q = tl.load(q_in + (b * S + s).to(tl.int64)[:, None] * HD + h * D + d, mask=sm, other=0.0)
        m_i = tl.full([BS], float("-inf"), tl.float32)
        l_i = tl.zeros([BS], tl.float32)
        acc = tl.zeros([BS, D], tl.float32)
        chunk = tl.cdiv(T, NS)
        lo = sp * chunk
        hi = tl.minimum(lo + chunk, T)
        base = (b * H + h).to(tl.int64) * T * D
        for j0 in range(lo, hi, BN):
            j = j0 + tl.arange(0, BN)
            ptr = base + j[:, None] * D + d
            m_i, l_i, acc = _attend(
                q, kx + ptr, vx + ptr, (j < hi)[:, None], (j < hi)[None, :], m_i, l_i, acc, QK_SCALE
            )
        # part [B, H, NS, S, D + 2]: acc, then the maximum and the sum
        row = (((b * H + h) * NS + sp) * S + s).to(tl.int64) * (D + 2)
        tl.store(part + row[:, None] + d, acc, mask=sm)
        tl.store(part + row + D, m_i, mask=s < S)
        tl.store(part + row + D + 1, l_i, mask=s < S)

    @triton.jit
    def _cross_combine_kernel(part, q_in, S, y,
                              H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, NS: tl.constexpr):  # fmt: skip
        bh = tl.program_id(0)
        sb = tl.program_id(1)
        b = bh // H
        h = bh % H
        HD: tl.constexpr = H * D
        s = sb * BS + tl.arange(0, BS)
        smask = s < S
        d = tl.arange(0, D)[None, :]
        top = tl.full([BS], float("-inf"), tl.float32)
        for sp in tl.static_range(NS):
            row = (((b * H + h) * NS + sp) * S + s).to(tl.int64) * (D + 2)
            top = tl.maximum(top, tl.load(part + row + D, mask=smask, other=float("-inf")))
        num = tl.zeros([BS, D], tl.float32)
        den = tl.zeros([BS], tl.float32)
        for sp in tl.static_range(NS):
            row = (((b * H + h) * NS + sp) * S + s).to(tl.int64) * (D + 2)
            m = tl.load(part + row + D, mask=smask, other=float("-inf"))
            w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - top))
            num += w[:, None] * tl.load(part + row[:, None] + d, mask=smask[:, None], other=0.0)
            den += w * tl.load(part + row + D + 1, mask=smask, other=0.0)
        out = y + (b * S + s).to(tl.int64)[:, None] * HD + h * D + d
        tl.store(out, (num / den[:, None]).to(y.dtype.element_ty), mask=smask[:, None])


def _qk_scale(d: int) -> float:
    return d**-0.5 * math.log2(math.e)


def self_attn_triton(
    qkv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    offset: torch.Tensor,
    src: torch.Tensor,
    config: tuple[int, int, int, int] = TACTICS["self_attn"][0],
) -> torch.Tensor:
    """The :func:`self_attn` kernel launched directly, for tests and timing."""
    b, s, _ = qkv.shape
    h, c, d = k_cache.shape[1:]
    bs, bn, warps, stages = config
    y = qkv.new_empty(b, s, h * d)
    _self_attn_kernel[(b * h, triton.cdiv(s, bs))](
        qkv, k_cache, v_cache, offset, src, s, c, y, H=h, D=d, BS=bs, BN=bn, QK_SCALE=_qk_scale(d),
        num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return y


def cross_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: tuple[int, int, int, int] = TACTICS["cross_attn"][0],
) -> torch.Tensor:
    """The :func:`cross_attn` kernel launched directly, for tests and timing."""
    b, s, _ = q.shape
    h, t, d = k.shape[1:]
    bs, bn, warps, stages = config
    y = torch.empty_like(q)
    _cross_attn_kernel[(b * h, triton.cdiv(s, bs))](
        q, k, v, s, t, y, H=h, D=d, BS=bs, BN=bn, QK_SCALE=_qk_scale(d), num_warps=warps, num_stages=stages
    )
    return y


def cross_attn_split_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    config: tuple[int, int, int, int] = TACTICS["cross_split"][0],
) -> torch.Tensor:
    """The :func:`cross_attn_split` kernel launched directly, for tests and timing."""
    b, s, _ = q.shape
    h, t, d = k.shape[1:]
    bs, bn, warps, stages = config
    part = q.new_empty(b, h, SPLITS, s, d + 2, dtype=torch.float32)
    _cross_split_kernel[(b * h, triton.cdiv(s, bs), SPLITS)](
        q, k, v, s, t, part, H=h, D=d, BS=bs, BN=bn, NS=SPLITS, QK_SCALE=_qk_scale(d), num_warps=warps,
        num_stages=stages,
    )  # fmt: skip
    return part


def cross_attn_combine_triton(
    part: torch.Tensor, q: torch.Tensor, config: tuple[int, int, int, int] = TACTICS["cross_combine"][0]
) -> torch.Tensor:
    b, h, _, s, d2 = part.shape
    bs, _, warps, stages = config
    y = torch.empty_like(q)
    _cross_combine_kernel[(b * h, triton.cdiv(s, bs))](
        part, q, s, y, H=h, D=d2 - 2, BS=bs, NS=SPLITS, num_warps=warps, num_stages=stages
    )
    return y


# --- TensorRT plugins ---

_COMPILED: dict = {}
_KERNELS = {  # plugin -> kernel, its arguments (inputs, scalars, outputs)
    "self_attn": ("_self_attn_kernel", ["qkv", "kc", "vc", "offs", "src", "S", "C", "y"]),
    "cross_attn": ("_cross_attn_kernel", ["q_in", "kx", "vx", "S", "T", "y"]),
    "cross_split": ("_cross_split_kernel", ["q_in", "kx", "vx", "S", "T", "part"]),
    "cross_combine": ("_cross_combine_kernel", ["part", "q_in", "S", "y"]),
}


def _config(kind: str, tactic: int) -> tuple[int, int, int, int]:
    configs = TACTICS[kind]
    return configs[max(tactic - 1, 0) % len(configs)]  # tactic 0: no autotuning


def _compile(kind: str, tactic: int, dtype: str):
    """(name, PTX, warps, shared memory, queries per program) of a kernel for the model being built."""
    h, d = _SHAPE["heads"], _SHAPE["head_dim"]
    bs, bn, warps, stages = _config(kind, tactic)
    key = (kind, h, d, dtype, bs, bn, warps, stages)
    if key not in _COMPILED:
        name, args = _KERNELS[kind]
        types = {"offs": "*i64", "src": "*i32", "part": "*fp32", "S": "i32", "C": "i32", "T": "i32"}
        consts = {"H": h, "D": d, "BS": bs, "BN": bn, "QK_SCALE": _qk_scale(d)}
        if kind == "cross_split":
            consts["NS"] = SPLITS
        elif kind == "cross_combine":
            consts = {"H": h, "D": d, "BS": bs, "NS": SPLITS}
        signature = {a: types.get(a, f"*{dtype}") for a in args} | dict.fromkeys(consts, "constexpr")
        # TensorRT and torch buffers are 16-byte aligned: lets loads/stores be vectorized
        aligned = {(i,): [["tt.divisibility", 16]] for i, a in enumerate(args) if signature[a][0] == "*"}
        src = triton.compiler.ASTSource(
            fn=globals()[name], signature=signature, constexprs=consts, attrs=aligned
        )
        k = triton.compile(src, options={"num_warps": warps, "num_stages": stages})
        # a name of its own per configuration (entry and parameter names start with it)
        unique = f"{k.metadata.name}_{bs}_{bn}_{warps}_{stages}"
        ptx = k.asm["ptx"].replace(k.metadata.name, unique)
        _COMPILED[key] = (unique, ptx, k.metadata.num_warps, k.metadata.shared, bs)
    return _COMPILED[key]


def register_plugins(heads: int, head_dim: int) -> None:
    """Define the TensorRT plugins (once per process) for a model with ``heads`` x ``head_dim``."""
    global _REGISTERED
    _SHAPE.update(heads=heads, head_dim=head_dim)
    if _REGISTERED:
        return
    import tensorrt as trt
    import tensorrt.plugin as trtp

    Desc = trtp.TensorDesc
    AOT = Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]
    Tune = List[trtp.AutoTuneCombination]
    names = {trt.float16: ("FP16", "fp16"), trt.bfloat16: ("BF16", "bf16")}

    def launch(kind: str, x: Desc, tactic: int, extras: list, grid_z: int = 1):
        """Grid (rows of ``x`` x heads, blocks of its tokens, ``grid_z``)."""
        rows, tokens = x.shape_expr[0], x.shape_expr[1]
        name, ptx, warps, shared, bs = _compile(kind, tactic, names[x.dtype][1])
        params = trtp.KernelLaunchParams()
        params.grid_x = rows * _SHAPE["heads"]
        params.grid_y = trtp.cdiv(tokens, bs)
        if grid_z > 1:
            params.grid_z = grid_z
        params.block_x = warps * 32
        params.shared_mem = shared
        args = trtp.SymIntExprs(len(extras))
        for i, e in enumerate(extras):
            args[i] = trtp.SymInt32(e)
        return name, ptx, params, args

    def tune(kind: str, x: Desc, other_types: dict | None = None) -> Tune:
        """Every input/output in ``x``'s type, except those given by index in ``other_types``."""
        args = [a for a in _KERNELS[kind][1] if a not in ("S", "C", "T")]  # inputs + outputs
        types = ", ".join((other_types or {}).get(i, names[x.dtype][0]) for i in range(len(args)))
        tactics = range(1, len(TACTICS[kind]) + 1) if TUNE else [1]
        return [trtp.AutoTuneCombination(types, "LINEAR", list(tactics))]

    @trtp.register(f"{NAMESPACE}::self_attn")
    def _self_desc(qkv: Desc, k_cache: Desc, v_cache: Desc, offset: Desc, src: Desc) -> Tuple[Desc]:
        b, s = qkv.shape_expr[0], qkv.shape_expr[1]
        return (trtp.from_shape_expr((b, s, k_cache.shape_expr[1] * k_cache.shape_expr[3]), qkv.dtype),)

    @trtp.autotune(f"{NAMESPACE}::self_attn")
    def _self_tune(
        qkv: Desc, k_cache: Desc, v_cache: Desc, offset: Desc, src: Desc, outputs: Tuple[Desc]
    ) -> Tune:
        return tune("self_attn", qkv, {3: "INT64", 4: "INT32"})

    @trtp.aot_impl(f"{NAMESPACE}::self_attn")
    def _self_aot(
        qkv: Desc, k_cache: Desc, v_cache: Desc, offset: Desc, src: Desc, outputs: Tuple[Desc], tactic: int
    ) -> AOT:
        return launch("self_attn", qkv, tactic, [qkv.shape_expr[1], k_cache.shape_expr[2]])

    @trtp.register(f"{NAMESPACE}::cross_attn")
    def _cross_desc(q: Desc, k: Desc, v: Desc) -> Tuple[Desc]:
        return (q.like(),)

    @trtp.autotune(f"{NAMESPACE}::cross_attn")
    def _cross_tune(q: Desc, k: Desc, v: Desc, outputs: Tuple[Desc]) -> Tune:
        return tune("cross_attn", q)

    @trtp.aot_impl(f"{NAMESPACE}::cross_attn")
    def _cross_aot(q: Desc, k: Desc, v: Desc, outputs: Tuple[Desc], tactic: int) -> AOT:
        return launch("cross_attn", q, tactic, [q.shape_expr[1], k.shape_expr[2]])

    @trtp.register(f"{NAMESPACE}::cross_attn_split")
    def _split_desc(q: Desc, k: Desc, v: Desc) -> Tuple[Desc]:
        h, d = _SHAPE["heads"], _SHAPE["head_dim"]
        return (trtp.from_shape_expr((q.shape_expr[0], h, SPLITS, q.shape_expr[1], d + 2), trt.float32),)

    @trtp.autotune(f"{NAMESPACE}::cross_attn_split")
    def _split_tune(q: Desc, k: Desc, v: Desc, outputs: Tuple[Desc]) -> Tune:
        return tune("cross_split", q, {3: "FP32"})

    @trtp.aot_impl(f"{NAMESPACE}::cross_attn_split")
    def _split_aot(q: Desc, k: Desc, v: Desc, outputs: Tuple[Desc], tactic: int) -> AOT:
        return launch("cross_split", q, tactic, [q.shape_expr[1], k.shape_expr[2]], grid_z=SPLITS)

    @trtp.register(f"{NAMESPACE}::cross_attn_combine")
    def _combine_desc(part: Desc, q: Desc) -> Tuple[Desc]:
        return (q.like(),)

    @trtp.autotune(f"{NAMESPACE}::cross_attn_combine")
    def _combine_tune(part: Desc, q: Desc, outputs: Tuple[Desc]) -> Tune:
        return tune("cross_combine", q, {0: "FP32"})

    @trtp.aot_impl(f"{NAMESPACE}::cross_attn_combine")
    def _combine_aot(part: Desc, q: Desc, outputs: Tuple[Desc], tactic: int) -> AOT:
        return launch("cross_combine", q, tactic, [q.shape_expr[1]])

    _REGISTERED = True
