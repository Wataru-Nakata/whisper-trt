"""One decoder step's GEMMs (32 layers x QKV, out, cross query, cross out, fc1, fc2) as a chain captured in a CUDA
graph: cuBLAS against Triton GEMM kernels with and without programmatic dependent launch (PDL: the next kernel
launches while the previous one runs, and waits for its results only before reading them); and cuBLAS with the
next GEMM's weights read into L2 on a side stream while each GEMM runs (--prefetch-only: that alone).

    python scripts/bench_gemm_chain.py [--rows 40 160] [--prefetch-only]
"""

import argparse

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

SHAPES = [(3840, 1280), (1280, 1280), (1280, 1280), (1280, 1280), (5120, 1280), (1280, 5120)]  # (N, K)


@triton.jit
def _gemm(x, w, b, y, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          PDL: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    if PDL:
        gdc_launch_dependents()
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    if PDL:
        gdc_wait()
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + rk
        xv = tl.load(x + rm.to(tl.int64)[:, None] * K + k[None, :], mask=(rm < M)[:, None], other=0.0)
        wv = tl.load(w + rn.to(tl.int64)[:, None] * K + k[None, :])
        acc += tl.dot(xv, tl.trans(wv))
    out = acc + tl.load(b + rn).to(tl.float32)[None, :]
    tl.store(y + rm.to(tl.int64)[:, None] * N + rn[None, :], out.to(tl.float16), mask=(rm < M)[:, None])


def graph_time(body, reps: int = 50) -> float:
    for _ in range(3):
        body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        body()
    a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return a.elapsed_time(e) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", type=int, default=[40, 160])
    ap.add_argument("--prefetch-only", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    weights = [[(torch.randn(n, k, device="cuda").half() * 0.02, torch.randn(n, device="cuda").half())
                for n, k in SHAPES] for _ in range(32)]
    nbytes = sum(w.numel() * 2 for layer in weights for w, _ in layer)
    for m in args.rows:
        xs = {k: torch.randn(m, k, device="cuda").half() for k in (1280, 5120)}
        outs = {n: torch.empty(m, n, device="cuda", dtype=torch.float16) for n in (1280, 3840, 5120)}

        def cublas():
            for layer in weights:
                for (w, b), (n, k) in zip(layer, SHAPES):
                    torch.addmm(b, xs[k], w.t(), out=outs[n])

        t = graph_time(cublas)
        print(f"rows {m}: cuBLAS {t:.3f} ms ({nbytes / t / 1e9:.2f} TB/s of weights)", flush=True)
        flat = [(w, b, n, k) for layer in weights for (w, b), (n, k) in zip(layer, SHAPES)]
        side = torch.cuda.Stream()
        sink = torch.zeros((), device="cuda")
        for ahead in (1, 2):
            def prefetched():
                main = torch.cuda.current_stream()
                for i, (w, b, n, k) in enumerate(flat):
                    if i + ahead < len(flat):  # the weights of the GEMM `ahead` later, into L2
                        side.wait_stream(main)
                        with torch.cuda.stream(side):
                            sink.add_(flat[i + ahead][0].sum(dtype=torch.float32))
                    torch.addmm(b, xs[k], w.t(), out=outs[n])
                main.wait_stream(side)

            t2 = graph_time(prefetched)
            print(f"rows {m}: cuBLAS, weights {ahead} GEMM(s) ahead read into L2 on a side stream: {t2:.3f} ms "
                  f"({nbytes / t2 / 1e9:.2f} TB/s)", flush=True)
        if args.prefetch_only:
            continue
        for bm, bn, bk, warps, stages in ((64, 16, 128, 4, 4), (64, 32, 128, 4, 4), (64, 64, 64, 4, 3),
                                          (32, 32, 128, 4, 4), (16, 64, 128, 4, 4), (128, 16, 128, 4, 3),
                                          (256, 16, 64, 8, 3)):
            if m > bm and bm < 128:
                pass
            for pdl in (False, True):
                def tri():
                    for layer in weights:
                        for (w, b), (n, k) in zip(layer, SHAPES):
                            _gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                                xs[k], w, b, outs[n], m, N=n, K=k, BM=bm, BN=bn, BK=bk, PDL=pdl,
                                num_warps=warps, num_stages=stages, launch_pdl=pdl)

                try:
                    t = graph_time(tri)
                    print(f"rows {m}: triton BM {bm} BN {bn} BK {bk} w{warps} s{stages} pdl={pdl}: {t:.3f} ms "
                          f"({nbytes / t / 1e9:.2f} TB/s)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"rows {m}: BM {bm} BN {bn} pdl={pdl} failed: {e}"[:200], flush=True)


if __name__ == "__main__":
    main()
