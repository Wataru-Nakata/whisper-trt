"""One decoder step's GEMMs (32 layers x QKV, out, cross query, cross out, fc1, fc2) as a chain in a CUDA graph:
cuBLAS against a split-K Triton GEMM in one kernel (each program one K slice of an output tile; the last
program of a tile, counted by an atomic, adds the slices and the bias), at several rows; and its error against
cuBLAS.

    python scripts/bench_splitk.py [--rows 40 160]
"""

import argparse

import torch
import triton
import triton.language as tl

SHAPES = [(3840, 1280), (1280, 1280), (1280, 1280), (1280, 1280), (5120, 1280), (1280, 5120)]  # (N, K)


@triton.jit
def _splitk_kernel(x, w, b, y, ws, cnt, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, SPLIT: tl.constexpr):  # fmt: skip
    pn = tl.program_id(0)
    pm = tl.program_id(1)
    ps = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    KS: tl.constexpr = K // SPLIT
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(ps * KS, (ps + 1) * KS, BK):
        k = k0 + rk
        xv = tl.load(x + rm.to(tl.int64)[:, None] * K + k[None, :], mask=(rm < M)[:, None], other=0.0)
        wv = tl.load(w + rn.to(tl.int64)[:, None] * K + k[None, :])
        acc += tl.dot(xv, tl.trans(wv))
    tile = pm * tl.num_programs(0) + pn
    part = ws + ((ps * tl.num_programs(0) * tl.num_programs(1) + tile) * BM + tl.arange(0, BM)[:, None]) * BN \
        + tl.arange(0, BN)[None, :]
    tl.store(part, acc)
    if tl.atomic_add(cnt + tile, 1) == SPLIT - 1:  # the tile's last slice: add them up
        tot = tl.zeros([BM, BN], tl.float32)
        for s in range(SPLIT):
            q = ws + ((s * tl.num_programs(0) * tl.num_programs(1) + tile) * BM + tl.arange(0, BM)[:, None]) * BN \
                + tl.arange(0, BN)[None, :]
            tot += tl.load(q, cache_modifier=".cg")
        out = tot + tl.load(b + rn).to(tl.float32)[None, :]
        tl.store(y + rm.to(tl.int64)[:, None] * N + rn[None, :], out.to(tl.float16), mask=(rm < M)[:, None])
        tl.atomic_xchg(cnt + tile, 0)


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
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    weights = [[(torch.randn(n, k, device="cuda").half() * 0.02, torch.randn(n, device="cuda").half())
                for n, k in SHAPES] for _ in range(32)]
    nbytes = sum(w.numel() * 2 for layer in weights for w, _ in layer)
    ws = torch.empty(64 * 1024 * 1024, device="cuda")  # fp32 partial tiles
    cnt = torch.zeros(4096, dtype=torch.int32, device="cuda")
    for m in args.rows:
        xs = {k: torch.randn(m, k, device="cuda").half() for k in (1280, 5120)}
        outs = {n: torch.empty(m, n, device="cuda", dtype=torch.float16) for n in (1280, 3840, 5120)}

        def cublas():
            for layer in weights:
                for (w, b), (n, k) in zip(layer, SHAPES):
                    torch.addmm(b, xs[k], w.t(), out=outs[n])

        t = graph_time(cublas)
        ref = {n: outs[n].float().clone() for n in outs}
        print(f"rows {m}: cuBLAS {t:.3f} ms ({nbytes / t / 1e9:.2f} TB/s of weights)", flush=True)
        for bm in (64, 128):
            if bm < m and bm == 64 and m > 128:
                pass
            for bn in (32, 64):
                for bk in (64, 128):
                    for split in (2, 4, 8):
                        for warps in (4,):
                            def tri():
                                for layer in weights:
                                    for (w, b), (n, k) in zip(layer, SHAPES):
                                        if (k // split) % bk:
                                            raise ValueError("K slice not a multiple of BK")
                                        grid = (n // bn, triton.cdiv(m, bm), split)
                                        _splitk_kernel[grid](xs[k], w, b, outs[n], ws, cnt, m, N=n, K=k, BM=bm, BN=bn,
                                                             BK=bk, SPLIT=split, num_warps=warps, num_stages=3)

                            try:
                                t2 = graph_time(tri)
                                err = max((outs[n].float() - ref[n]).abs().max().item() for n in outs)
                                print(f"rows {m}: split-K BM {bm} BN {bn} BK {bk} split {split}: {t2:.3f} ms "
                                      f"({nbytes / t2 / 1e9:.2f} TB/s; max |diff| vs cuBLAS {err:.3g})", flush=True)
                            except Exception as e:  # noqa: BLE001
                                print(f"rows {m}: BM {bm} BN {bn} BK {bk} split {split} failed: {str(e)[:120]}",
                                      flush=True)


if __name__ == "__main__":
    main()
