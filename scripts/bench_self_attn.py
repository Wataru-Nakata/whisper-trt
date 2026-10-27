"""Self-attention of single-token steps (32 layers' calls in a CUDA graph): attn_plugin's kernel (its first
tactic) against its single-token kernel (self_attn_step_triton: one query per (row, head), no 16-row tile, fp32
probabilities) with several block sizes and warps, at several rows and cache fills; and the outputs' difference
from fp32 attention.

    python scripts/bench_self_attn.py [--rows 40 160] [--offsets 20 60 120 200]
"""

import argparse

import torch

from whisper_trt import attn_plugin as ap_

H, D, C, L = 20, 64, 228, 32


def decode(qkv, kc, vc, offs, src, bn, warps):
    return ap_.self_attn_step_triton(qkv, kc, vc, offs, src, (bn, warps))


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


def reference(qkv, kc, vc, offs, src):
    """fp32 attention of each row's query over its cache slots < offset (rows from src) and its new token."""
    n = qkv.shape[0]
    out = torch.empty(n, H * D, device="cuda")
    q, k_new, v_new = qkv.float().view(n, 3, H, D).unbind(1)
    for b in range(n):
        o = int(offs[b])
        rows = src[b, :o].long()
        k = torch.cat([kc[rows, :, torch.arange(o)].float(), k_new[b][None]], 0)  # [o + 1, H, D]
        v = torch.cat([vc[rows, :, torch.arange(o)].float(), v_new[b][None]], 0)
        w = torch.einsum("hd,thd->ht", q[b], k) / D**0.5
        out[b] = torch.einsum("ht,thd->hd", w.softmax(-1), v).reshape(-1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", type=int, default=[40, 160])
    ap.add_argument("--offsets", nargs="+", type=int, default=[20, 60, 120, 200])
    args = ap.parse_args()
    for n in args.rows:
        caches = [(torch.randn(n, H, C, D, device="cuda").half(), torch.randn(n, H, C, D, device="cuda").half())
                  for _ in range(L)]
        qkv = torch.randn(n, 1, 3 * H * D, device="cuda").half()
        # beams: each row's slots in rows of its window (5 rows a window), as beam search leaves them
        src = (torch.arange(n, device="cuda")[:, None] // 5 * 5
               + torch.randint(0, 5, (n, C), device="cuda")).int()
        for o in args.offsets:
            offs = torch.full((n,), o, dtype=torch.long, device="cuda")
            nbytes = L * n * H * o * D * 2 * 2
            ref = reference(qkv[:, 0], *caches[0], offs, src)
            line = f"rows {n} offset {o}:"
            for cfg in ap_.TACTICS["self_attn"][:1]:
                t = graph_time(lambda: [ap_.self_attn_triton(qkv, k, v, offs, src, cfg) for k, v in caches])
                err = (ap_.self_attn_triton(qkv, *caches[0], offs, src, cfg).float()[:, 0] - ref).abs().max().item()
                line += f" plugin{cfg} {t:.3f} ms ({nbytes / t / 1e9:.1f} TB/s, err {err:.2g});"
            for bn in (16, 32, 64):
                for warps in (1, 2, 4):
                    t = graph_time(lambda: [decode(qkv, k, v, offs, src, bn, warps) for k, v in caches])
                    err = (decode(qkv, *caches[0], offs, src, bn, warps).float()[:, 0] - ref).abs().max().item()
                    mark = " (default)" if (bn, warps) == ap_.self_attn_step_config(n) else ""
                    line += f" step(BN {bn} w{warps}){mark} {t:.3f} ms ({nbytes / t / 1e9:.1f} TB/s, err {err:.2g});"
            print(line.replace(";", "\n  "), flush=True)


if __name__ == "__main__":
    main()
