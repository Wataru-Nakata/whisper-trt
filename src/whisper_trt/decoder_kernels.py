"""The decoder step's linear layers as fused Triton kernels (fp16 in, fp32 accumulation, fp16 out).

TensorRT runs every decoder GEMM of a step as a kernel of its own with few blocks for the 1-160 rows of a step
(0.6 TB/s of weights on an H200), and LayerNorm, GELU and residual additions as kernels between them. Here:

* :func:`ln_linear`: ``LN(x) @ w^T + b`` (optionally GELU): each program normalizes its rows itself (fp32
  statistics over the whole row; the normalized row rounded to fp16, as ``modeling.LayerNorm`` gives it), so
  the LayerNorm needs no kernel of its own;
* :func:`linear_residual`: ``res + (x @ w^T + b)``;
* :func:`layer_norm_bias`: ``LN(x)`` (fp32 statistics, rounded to fp16) and, in place, ``x += bias``: the bias
  of the GEMM whose result is added to ``x`` next (``x.addmm_(h, w.t())``, cuBLAS), so that GEMM needs no
  epilogue of its own.

The numerics follow PyTorch's (cuBLAS): fp32 accumulation, the GEMM's result rounded to fp16 before the
activation (computed in fp32, ``Precision.gelu32``) or the residual addition.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # (no Triton: the TensorRT decoder is used)
    triton = None


def _configs(split_m: bool):
    out = []
    for bm in (16, 32, 64, 128) if split_m else (16, 32, 64, 128, 256):
        for bn in (16, 32, 64):
            for bk in (64, 128):
                for stages in (3, 4):
                    if (bm + bn) * bk * 2 * stages <= 160 * 1024:
                        out.append(triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=4, num_stages=stages))
    return out


if triton is not None:

    @triton.jit
    def _gelu(x):
        return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    @triton.autotune(configs=_configs(False), key=["M", "N", "K"])
    @triton.jit
    def _ln_linear_kernel(x, gamma, beta, w, b, y, M, N, K: tl.constexpr, EPS: tl.constexpr, GELU: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):  # fmt: skip
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm = rm < M
        row = rm.to(tl.int64)[:, None] * K
        # the rows' statistics (two passes, fp32)
        s = tl.zeros([BM], tl.float32)
        for k0 in range(0, K, BK):
            s += tl.sum(tl.load(x + row + (k0 + rk)[None, :], mask=mm[:, None], other=0.0).to(tl.float32), 1)
        mean = s / K
        v = tl.zeros([BM], tl.float32)
        for k0 in range(0, K, BK):
            d = tl.load(x + row + (k0 + rk)[None, :], mask=mm[:, None], other=0.0).to(tl.float32) - mean[:, None]
            v += tl.sum(d * d, 1)
        rstd = 1.0 / tl.sqrt(v / K + EPS)
        acc = tl.zeros([BM, BN], tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + rk
            xv = tl.load(x + row + k[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
            g = tl.load(gamma + k).to(tl.float32)
            bb = tl.load(beta + k).to(tl.float32)
            xn = ((xv - mean[:, None]) * rstd[:, None] * g[None, :] + bb[None, :]).to(tl.float16)
            wv = tl.load(w + rn.to(tl.int64)[:, None] * K + k[None, :], mask=(rn < N)[:, None], other=0.0)
            acc += tl.dot(xn, tl.trans(wv))
        out = acc + tl.load(b + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
        if GELU:
            out = _gelu(out.to(tl.float16).to(tl.float32))
        tl.store(y + rm.to(tl.int64)[:, None] * N + rn[None, :], out.to(tl.float16), mask=mm[:, None] & (rn < N)[None, :])

    @triton.autotune(configs=_configs(False), key=["M", "N", "K"])
    @triton.jit
    def _linear_residual_kernel(x, w, b, res, y, M, N, K: tl.constexpr,
                                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):  # fmt: skip
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mm = rm < M
        acc = tl.zeros([BM, BN], tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + rk
            xv = tl.load(x + rm.to(tl.int64)[:, None] * K + k[None, :], mask=mm[:, None], other=0.0)
            wv = tl.load(w + rn.to(tl.int64)[:, None] * K + k[None, :], mask=(rn < N)[:, None], other=0.0)
            acc += tl.dot(xv, tl.trans(wv))
        out = (acc + tl.load(b + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]).to(tl.float16)
        idx = rm.to(tl.int64)[:, None] * N + rn[None, :]
        m2 = mm[:, None] & (rn < N)[None, :]
        r = tl.load(res + idx, mask=m2, other=0.0)
        tl.store(y + idx, (out.to(tl.float32) + r.to(tl.float32)).to(tl.float16), mask=m2)


if triton is not None:

    @triton.jit
    def _layer_norm_bias_kernel(x, gamma, beta, bias, y, K: tl.constexpr, EPS: tl.constexpr, BK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64) * K
        k = tl.arange(0, BK)
        m = k < K
        xv = tl.load(x + row + k, mask=m, other=0.0)
        xf = xv.to(tl.float32)
        mean = tl.sum(xf, 0) / K
        d = tl.where(m, xf - mean, 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(d * d, 0) / K + EPS)
        g = tl.load(gamma + k, mask=m, other=0.0).to(tl.float32)
        b = tl.load(beta + k, mask=m, other=0.0).to(tl.float32)
        tl.store(y + row + k, (d * rstd * g + b).to(tl.float16), mask=m)
        tl.store(x + row + k, (xf + tl.load(bias + k, mask=m, other=0.0).to(tl.float32)).to(tl.float16), mask=m)


def layer_norm_bias(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, bias: torch.Tensor,
                    eps: float = 1e-5) -> torch.Tensor:
    """``LN(x)`` [M, K] fp16 (fp32 statistics), and ``x += bias`` in place (after reading it)."""
    m, k = x.shape
    y = torch.empty_like(x)
    _layer_norm_bias_kernel[(m,)](x, gamma, beta, bias, y, K=k, EPS=eps, BK=triton.next_power_of_2(k),
                                  num_warps=4)
    return y


def ln_linear(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
              gelu: bool = False, eps: float = 1e-5, out: torch.Tensor | None = None) -> torch.Tensor:
    """``LN(x) @ w^T + b`` (GELU'd if ``gelu``): x [M, K] fp16, w [N, K] -> [M, N] fp16."""
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty(m, n, dtype=x.dtype, device=x.device) if out is None else out
    grid = lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))  # noqa: E731
    _ln_linear_kernel[grid](x, gamma, beta, w, b, y, m, n, K=k, EPS=eps, GELU=gelu)
    return y


def linear_residual(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, res: torch.Tensor,
                    out: torch.Tensor | None = None) -> torch.Tensor:
    """``res + (x @ w^T + b)``: x [M, K] fp16, w [N, K], res [M, N] -> [M, N] fp16 (``out`` may be ``res``)."""
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty_like(res) if out is None else out
    grid = lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))  # noqa: E731
    _linear_residual_kernel[grid](x, w, b, res, y, m, n, K=k)
    return y
