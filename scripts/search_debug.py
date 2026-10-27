"""search.Search on TensorRT against a manual decoder loop fed the same tokens: the first step whose logits
differ shows where Search's state (cache, slot table, offsets, buffers) goes wrong.

    python scripts/search_debug.py WAVES.npz --items 15 17 [--graphs 0|1] [--sync 0|1]
"""

import argparse

import numpy as np
import torch

from whisper_trt import Whisper
from whisper_trt.search import Search


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--items", nargs="+", type=int, default=[15, 17])
    ap.add_argument("--graphs", type=int, default=0)
    ap.add_argument("--sync", type=int, default=0, help="synchronize after every Search step")
    ap.add_argument("--reuse", type=int, default=0, help="one Search for all items (as Whisper.generate)")
    ap.add_argument("--fresh-prefill", type=int, default=0, help="drop the cached prefill closures before each item")
    ap.add_argument("--prefill-twice", type=int, default=0, help="run the prefill once before the search")
    args = ap.parse_args()
    data = np.load(args.waves)
    trt = Whisper.from_pretrained(args.model, backend="tensorrt")
    cfg = trt.cfg
    L, H, dk, V = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size
    prompt = trt.prompt("en", without_timestamps=True)
    P = len(prompt)
    suppress = list(trt.tokenizer.suppressed_tokens((-1,)))
    begin = list(cfg.begin_suppress_tokens)
    with torch.cuda.stream(trt.stream):
        shared = None
        for i in args.items:
            trt.encode(trt.features([torch.from_numpy(data[f"w{i}"])]))  # into trt.cross(1)
            if shared is None or not args.reuse:
                search = Search(cfg, lambda *a: trt.backend.step_fn(*a, cross=trt.cross(1)), 1, 1, P, "greedy",
                                graphs=bool(args.graphs))
                choose = search._choose
                shared = search
            seen = []
            if args.fresh_prefill:
                trt._prefills.clear()
            if args.prefill_twice:
                c0 = search.cache(search.buckets[0])
                s0 = search.src(search.buckets[0])
                s0.copy_(search.row_ids.expand_as(s0))
                l1 = trt._prefill(1, torch.tensor([prompt], device="cuda"), c0, s0).clone()
                l2 = trt._prefill(1, torch.tensor([prompt], device="cuda"), c0, s0).clone()
                print(f"    prefill twice: max |diff| {float((l1 - l2).abs().max()):.3g}", flush=True)

            def record(logits, c, first, choose=choose, search=search, seen=seen):
                seen.append((logits.clone(), search.pos.clone(), c))  # (no host sync)
                if args.sync:
                    torch.cuda.synchronize()
                choose(logits, c, first)

            search._choose = record
            res = search.run(torch.tensor([prompt]), lambda r, c, s: trt._prefill(1, r, c, s), suppress, begin)[0]
            toks = res.tokens
            print(f"[{i}] search: {trt.tokenizer.decode(toks)!r} ({len(seen)} steps recorded)", flush=True)
            # manual: the same tokens, one call each
            C = 256
            cache = torch.zeros(L, 2, 1, H, C, dk, dtype=torch.float16, device="cuda")
            src = torch.zeros(1, C, dtype=torch.int32, device="cuda")
            manual = [trt._prefill(1, torch.tensor([prompt], device="cuda"), cache, src)[0, -1].clone()]
            tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
            off = torch.zeros(1, dtype=torch.long, device="cuda")
            logits = torch.zeros(1, 1, V, device="cuda")
            new_kv = torch.zeros(L, 2, 1, H, 1, dk, dtype=torch.float16, device="cuda")
            step = trt.backend.step_fn(tok, off, cache, logits, new_kv, src, cross=trt.cross(1))
            for j, t in enumerate(toks):
                tok.fill_(t)
                off.fill_(P + j)
                step()
                cache[..., P + j : P + j + 1, :] = new_kv
                manual.append(logits[0, 0].clone())
            for j, (lg, pos, c) in enumerate(seen[: len(manual)]):
                pos = int(pos)
                d = float((lg[0] - manual[j]).abs().max())
                if d > 0.05:
                    if j:
                        print(f"    step {j - 1} matched: max |diff| {float((seen[j - 1][0][0] - manual[j - 1]).abs().max()):.3g}")
                    fed = toks[j - 1] if j else None
                    print(f"    step {j} (pos {pos}, bucket {c}, fed {fed}): max |logit diff| {d:.3g}; search argmax "
                          f"{int(lg[0].argmax())} manual {int(manual[j].argmax())}", flush=True)
                    break
            else:
                print("    every step equal", flush=True)


if __name__ == "__main__":
    main()
