"""Which work runs on a green-context stream (torch.cuda.GreenContext): a torch matmul, the TensorRT encoder as
is, the TensorRT encoder with an execution context made while the green context is current, and the encoder
captured in a CUDA graph on a normal stream then launched on the green stream. Each in a child process (a crash
ends only that check).

    python scripts/green_check.py [--sms 66]
"""

import argparse
import faulthandler
import multiprocessing as mp
import sys


def check(kind: str, sms: int) -> None:
    faulthandler.enable()
    import torch
    from torch.cuda.green_contexts import GreenContext

    from whisper_trt import Whisper
    from whisper_trt.config import N_FRAMES

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    ref = x @ x
    gc = GreenContext(num_sms=sms)
    gs = gc.Stream()
    if kind == "matmul":
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            y = x @ x
        torch.cuda.synchronize()
        print(f"{kind}: ok, equal {torch.equal(y, ref)}", flush=True)
        return
    model = Whisper.from_pretrained()
    be = model.backend.encoder
    feats = torch.randn(1, model.cfg.n_mels, N_FRAMES, device="cuda")
    c = model.cfg
    out = torch.empty(c.dec_layers, 2, 1, c.dec_heads, c.enc_frames, c.head_dim, dtype=torch.float16, device="cuda")
    want = torch.empty_like(out)
    be.encode(feats, want, lane=1)
    torch.cuda.synchronize()
    gs.wait_stream(torch.cuda.current_stream())
    if kind == "trt":
        with torch.cuda.stream(gs):
            be.encode(feats, out, lane=1)
    elif kind == "trt-ctx":
        gc.set_context()
        try:
            be._enc_ctxs.clear()
            be._scratch.clear()
            with torch.cuda.stream(gs):
                be.encode(feats, out, lane=1)
        finally:
            gc.pop_context()
    elif kind == "graph":
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            be.encode(feats, out, lane=1)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            be.encode(feats, out, lane=1)
        out.zero_()
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            g.replay()
    torch.cuda.synchronize()
    print(f"{kind}: ok, output equal {torch.equal(out, want)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sms", type=int, default=66)
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    for kind in ("matmul", "trt", "trt-ctx", "graph"):
        p = ctx.Process(target=check, args=(kind, args.sms))
        p.start()
        p.join()
        if p.exitcode != 0:
            print(f"{kind}: FAILED (exit code {p.exitcode})", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
