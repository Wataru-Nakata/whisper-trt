"""TensorRT engines against the PyTorch model (fp32), component by component, on real audio.

    python scripts/check_engines.py WAVES.npz [--n 3]

Prints the largest differences of the cross-attention keys/values, of the prompt's logits, and of one decode
step's logits through a slot table that sends some slots to other rows (as beam search does).
"""

import argparse
import json

import numpy as np
import torch

from whisper_trt import Whisper
from whisper_trt.modeling import load_model


def report(name: str, ours: torch.Tensor, ref: torch.Tensor) -> None:
    d = (ours.float() - ref.float()).abs()
    rel = d.max() / ref.float().abs().max().clamp(min=1e-9)
    print(f"{name}: max |diff| {d.max():.4g} (relative {rel:.3g}), mean {d.mean():.3g}, ref max {ref.abs().max():.3g}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = [torch.from_numpy(data[f"w{i}"]) for i in range(min(args.n, len(meta)))]

    trt = Whisper.from_pretrained(args.model, backend="tensorrt")
    from whisper_trt import hub

    ref = load_model(hub.resolve(args.model), trt.cfg, dtype=torch.float32)
    cfg = trt.cfg
    feats = trt.features(waves)
    w = feats.shape[0]
    cross = trt.encode(feats).clone()
    with torch.inference_mode():
        enc = ref.encoder(feats)
        cross_ref = ref.decoder.cross_kv(enc)
    report("cross_kv", cross, cross_ref)
    with torch.inference_mode():
        h16 = load_model(hub.resolve(args.model), cfg, dtype=torch.float16)
        cross16 = h16.decoder.cross_kv(h16.encoder(feats))
        del h16
    report("cross_kv (torch fp16 vs fp32)", cross16, cross_ref)

    # prompt, then one step: rows 0..w-1 with a 2-row-per-window layout to exercise the slot table
    K = 2
    n = w * K
    prompt = trt.prompt("en", without_timestamps=True)
    P = len(prompt)
    C = 64
    L, H, dk = cfg.dec_layers, cfg.dec_heads, cfg.head_dim
    rows = torch.tensor([prompt] * n, device="cuda")
    for name, model_cross, run in (("tensorrt", cross, "trt"), ("torch32", cross_ref, "ref")):
        cache = torch.zeros(L, 2, n, H, C, dk, device="cuda",
                            dtype=torch.float16 if run == "trt" else torch.float32)
        src = torch.arange(n, dtype=torch.int32, device="cuda")[:, None].repeat(1, C)
        if run == "trt":
            trt._cross_flat[: cross.numel()].copy_(cross.view(-1))
            logits = trt._prefill(w, rows, cache, src).clone()
        else:
            with torch.inference_mode():
                lg, kv = ref.decoder.step(rows, torch.zeros(n, dtype=torch.long, device="cuda"),
                                          [cache[i, 0] for i in range(L)], [cache[i, 1] for i in range(L)],
                                          [model_cross[i, 0] for i in range(L)], [model_cross[i, 1] for i in range(L)])
                cache[..., :P, :] = kv
                logits = lg
        if run == "trt":
            out_trt_prefill, cache_trt = logits, cache
        else:
            out_ref_prefill, cache_ref = logits, cache
    report("prefill logits", out_trt_prefill, out_ref_prefill)
    # a decode step after writing different tokens per row at slot P, with row r reading slot P from row r^1
    toks = torch.tensor([[220 + r] for r in range(n)], device="cuda")
    for run in ("trt", "ref"):
        cache = cache_trt if run == "trt" else cache_ref
        src = torch.arange(n, dtype=torch.int32, device="cuda")[:, None].repeat(1, C)
        src[:, P] = torch.arange(n, dtype=torch.int32, device="cuda") ^ 1
        # slot P of each row holds the KV of the row's own previous token (computed by a first step)
        first = torch.full((n, 1), 1000, device="cuda")
        if run == "trt":
            logits = torch.zeros(n, 1, cfg.vocab_size, device="cuda")
            new_kv = torch.zeros(L, 2, n, H, 1, dk, device="cuda", dtype=torch.float16)
            for tok, o, table in ((first + toks, P, None), (toks, P + 1, src)):
                t = tok.contiguous()
                offset = torch.full((n,), o, dtype=torch.long, device="cuda")
                tab = table if table is not None else torch.arange(n, dtype=torch.int32, device="cuda")[:, None].repeat(1, C)
                fn = trt.backend.step_fn(t, offset, cache, logits, new_kv, tab.contiguous(), cross=trt.cross(w))
                fn()
                cache[..., o : o + 1, :] = new_kv
            step_trt = logits.clone()
        else:
            with torch.inference_mode():
                for tok, o, table in ((first + toks, P, None), (toks, P + 1, src)):
                    offset = torch.full((n,), o, dtype=torch.long, device="cuda")
                    lg, kv = ref.decoder.step(tok, offset, [cache[i, 0] for i in range(L)],
                                              [cache[i, 1] for i in range(L)], [cross_ref[i, 0] for i in range(L)],
                                              [cross_ref[i, 1] for i in range(L)], table)
                    cache[..., o : o + 1, :] = kv
                step_ref = lg
    report("decode step logits (shuffled slot table)", step_trt, step_ref)
    top_t, top_r = step_trt[:, 0].argmax(-1), step_ref[:, 0].argmax(-1)
    print("argmax equal:", (top_t == top_r).tolist())


if __name__ == "__main__":
    main()
