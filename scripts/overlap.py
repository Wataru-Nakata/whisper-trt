"""How well the encoder (on a side stream) overlaps the decoder steps: D captured step graphs (32 windows x 5
beams) and E single-window encodes, alone and together, with the encoder's stream as Whisper.decode makes it
(priority 0), with priority -1, and captured in a CUDA graph; overlap = the time saved against running them one
after the other, as a share of the shorter. (An encoder profile built for a CUDA green context of fewer SMs,
tried as a build option, overlapped more but took longer together: 1413-1535 against 1323 ms.)

    python scripts/overlap.py WAVES.npz --lang ja [--steps 120] [--encodes 100] [--build KEY=VALUE]
"""

import argparse
import json
import time

import numpy as np
import torch

from whisper_trt import Whisper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--encodes", type=int, default=100)
    ap.add_argument("--build", nargs="*", default=[], help="engine build options KEY=VALUE (build.BuildOptions)")
    args = ap.parse_args()
    from profile_decode import build_options

    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = []
    for i in range(min(len(meta), 2 * args.batch)):
        w = data[f"w{i}"]
        waves.append(torch.from_numpy(w.astype(np.float32) / 32768 if w.dtype == np.int16 else w))
    model = Whisper.from_pretrained(**build_options(args.build))
    prompt = model.prompt(args.lang)
    model.decode(waves, prompt, beam_size=5, batch_size=args.batch)  # (captures the step graph)
    search = model._search("beam", args.batch, 5, len(prompt))
    main_stream = model.stream
    with torch.cuda.stream(main_stream):
        feats = model.features(waves[:1])

    def reset() -> None:  # every slot decoding from 80 cached positions, far from the token limit
        with torch.cuda.stream(main_stream):
            search.pos.fill_(80)
            search.t.fill_(0)
            search.done.fill_(False)
            search.h_count.zero_()

    c = model.cfg
    buf = torch.empty(c.dec_layers, 2, 1, c.dec_heads, c.enc_frames, c.head_dim, dtype=torch.float16, device="cuda")
    reference = torch.empty_like(buf)
    with torch.cuda.stream(main_stream):
        model.backend.encode(feats, reference, lane=1)
    torch.cuda.synchronize()

    def run(steps: int, encodes: int, side) -> float:
        reset()
        side.wait_stream(main_stream)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(max(steps, encodes)):
            if i < steps:
                with torch.cuda.stream(main_stream):
                    search._graph.replay()
            if i < encodes:
                with torch.cuda.stream(side):
                    model.backend.encode(feats, buf, lane=1)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    sides = [("the model's encoder stream", model.side_stream), ("priority -1", torch.cuda.Stream(priority=-1))]
    # the encoder captured in a CUDA graph (one launch instead of TensorRT's ~300)
    gside = torch.cuda.Stream(priority=0)
    gside.wait_stream(main_stream)
    with torch.cuda.stream(gside):
        model.backend.encode(feats, buf, lane=1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=gside):
        model.backend.encode(feats, buf, lane=1)
    encode_eager = model.backend.encode
    t_dec = run(args.steps, 0, sides[0][1])
    print(f"{args.steps} steps alone: {t_dec * 1000:.0f} ms ({t_dec / args.steps * 1000:.2f} ms each)", flush=True)
    sides.append(("graph on a priority-0 stream", gside))
    for name, side in sides:
        if side is gside:
            model.backend.encode = lambda *a, **k: graph.replay()
        t_enc = run(0, args.encodes, side)
        t_both = run(args.steps, args.encodes, side)
        model.backend.encode = encode_eager
        ok = torch.equal(buf, reference)
        saved = (t_dec + t_enc - t_both) / min(t_dec, t_enc)
        print(f"{name}: {args.encodes} encodes alone {t_enc * 1000:.0f} ms ({t_enc / args.encodes * 1000:.2f} ms "
              f"each); together {t_both * 1000:.0f} ms (one after the other: {(t_dec + t_enc) * 1000:.0f}); "
              f"overlap {saved:.0%}; encoder output {'identical' if ok else 'DIFFERENT'}", flush=True)


if __name__ == "__main__":
    main()
