"""Teacher-forced decoder comparison: the reference transcript's tokens fed one by one through the TensorRT
decoder (as search.Search feeds them: prefill, then one call per token, cache written at its slot), against
the PyTorch fp32 decoder on the same cross-attention keys/values (the TensorRT encoder's), so that only the
decoder differs. Prints, per utterance, the first step whose argmax differs and the logit gap there.

    python scripts/teacher_force.py WAVES.npz REF.json --items 9 11 15 17
"""

import argparse
import json

import numpy as np
import torch

from whisper_trt import Whisper, hub
from whisper_trt.modeling import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("ref")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--items", nargs="+", type=int, default=[9, 11, 15, 17])
    ap.add_argument("--run", default="float32/beam1/b1")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    data = np.load(args.waves)
    ref = json.load(open(args.ref))["runs"][args.run]["rows"]
    trt = Whisper.from_pretrained(args.model, backend="tensorrt", decoder="tensorrt")
    m32 = load_model(hub.resolve(args.model), trt.cfg, dtype=torch.float32)
    with torch.cuda.stream(trt.stream):  # (the model works on its own stream)
        for i in args.items:
            run_item(args, trt, m32, data, ref, i)


def run_item(args, trt, m32, data, ref, i) -> None:
    cfg = trt.cfg
    L, H, dk, V = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size
    prompt = trt.prompt("en", without_timestamps=True)
    P = len(prompt)
    if True:
        wave = torch.from_numpy(data[f"w{i}"])
        cross = trt.encode(trt.features([wave]))  # [L, 2, 1, H, T, dk]
        toks = ref[i]["tokens"] + [cfg.eot]
        seq = prompt + toks
        with torch.inference_mode():
            full, _ = m32.decoder(torch.tensor([seq], device="cuda"), cross.float())
        want = full[0, P - 1 : P - 1 + len(toks)]  # logits predicting each reference token
        C = 256
        cache = torch.zeros(L, 2, 1, H, C, dk, dtype=torch.float16, device="cuda")
        src = torch.zeros(1, C, dtype=torch.int32, device="cuda")
        got = [trt._prefill(1, torch.tensor([prompt], device="cuda"), cache, src)[0, -1].clone()]
        tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        off = torch.zeros(1, dtype=torch.long, device="cuda")
        logits = torch.zeros(1, 1, V, device="cuda")
        new_kv = torch.zeros(L, 2, 1, H, 1, dk, dtype=torch.float16, device="cuda")
        step = trt.backend.step_fn(tok, off, cache, logits, new_kv, src, cross=trt.cross(1))
        for j, t in enumerate(toks[:-1]):
            tok.fill_(t)
            off.fill_(P + j)
            step()
            cache[..., P + j : P + j + 1, :] = new_kv
            got.append(logits[0, 0].clone())
        got = torch.stack(got)
        a, b = got.argmax(-1), want.argmax(-1)
        diff = (a != b).nonzero()
        first = int(diff[0, 0]) if len(diff) else None
        d = (got - want).abs().max(-1).values
        line = f"[{i}] {len(toks)} tokens, max |logit diff| {d.max():.3g} (mean over steps {d.mean():.3g})"
        if first is None:
            line += ", argmax equal at every step"
        else:
            top2 = want[first].topk(2).values
            line += (f", first argmax difference at step {first}: ours {int(a[first])} ({trt.tokenizer.decode([int(a[first])])!r})"
                     f" ref {int(b[first])} ({trt.tokenizer.decode([int(b[first])])!r}), ref top-2 gap {float(top2[0] - top2[1]):.3g}"
                     f", {len(diff)} steps differ; per-step diff {[round(float(x), 2) for x in d[: first + 3]]}")
        print(line, flush=True)
        # free-running greedy: Search (as generate runs it) against manual loops with cache sizes 64 and 256
        res = trt.generate(cross, prompt, beam_size=1)[0].tokens
        print(f"    search greedy: {trt.tokenizer.decode(res)!r}", flush=True)
        for C in (64, 256):
            cache = torch.zeros(L, 2, 1, H, C, dk, dtype=torch.float16, device="cuda")
            src = torch.zeros(1, C, dtype=torch.int32, device="cuda")
            nxt = int(trt._prefill(1, torch.tensor([prompt], device="cuda"), cache, src)[0, -1].argmax())
            step = trt.backend.step_fn(tok, off, cache, logits, new_kv, src, cross=trt.cross(1))
            out = []
            for j in range(C - P - 1):
                if nxt == cfg.eot:
                    break
                out.append(nxt)
                tok.fill_(nxt)
                off.fill_(P + j)
                step()
                cache[..., P + j : P + j + 1, :] = new_kv
                nxt = int(logits[0, 0].argmax())
            print(f"    manual greedy C={C}: {trt.tokenizer.decode(out)!r}", flush=True)


if __name__ == "__main__":
    main()
