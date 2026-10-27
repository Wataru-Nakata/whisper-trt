"""Do TensorRT contexts on one optimization profile disturb each other when bound to different shapes?

A decoder prefill context A (cache length 64) runs, then B (same profile, cache length 256), then A again on
the same inputs: A's logits should not change. Then the same with A's input shapes set again before its
second run.

    python scripts/context_clash.py WAVES.npz
"""

import argparse

import numpy as np
import torch

from whisper_trt import Whisper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    args = ap.parse_args()
    data = np.load(args.waves)
    trt = Whisper.from_pretrained(args.model, backend="tensorrt", decoder="tensorrt")
    cfg = trt.cfg
    L, H, dk, V = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size
    be = trt.backend
    prompt = torch.tensor([trt.prompt("en")], device="cuda")
    with torch.cuda.stream(trt.stream):
        trt.encode(trt.features([torch.from_numpy(data["w0"])]))

        def make(C, n=1, s=prompt.shape[1]):
            cache = torch.zeros(L, 2, n, H, C, dk, dtype=torch.float16, device="cuda")
            src = torch.zeros(n, C, dtype=torch.int32, device="cuda")
            tok = prompt.repeat(n, 1)[:, :s].contiguous()
            off = torch.zeros(n, dtype=torch.long, device="cuda")
            logits = torch.zeros(n, s, V, device="cuda")
            new_kv = torch.zeros(L, 2, n, H, s, dk, dtype=torch.float16, device="cuda")
            fn = be.step_fn(tok, off, cache, logits, new_kv, src, cross=trt.cross(1))
            fn.inputs = {"tokens": tok, "offset": off, "cache": cache, "src": src, "cross": trt.cross(1)}
            return fn, logits

        a, la = make(64)
        before = {k: v.clone() for k, v in a.inputs.items()}
        a()
        torch.cuda.synchronize()
        for k, v in a.inputs.items():
            changed = (v != before[k]).sum().item()
            print(f"input {k}: {changed} of {v.numel()} elements changed by a run", flush=True)
            if changed and v.numel() < 64:
                print("   before", before[k].flatten().tolist(), "after", v.flatten().tolist())
        first = la.clone()
        a.inputs["tokens"].copy_(before["tokens"])
        a.inputs["offset"].copy_(before["offset"])
        a()
        print("A twice, tokens/offset refilled:", float((la - first).abs().max()), flush=True)
        a()
        print("A twice:", float((la - first).abs().max()), flush=True)
        b, lb = make(256)
        b()
        print("B (C=256) vs A:", float((lb - first).abs().max()), flush=True)
        a()
        print("A after B:", float((la - first).abs().max()), flush=True)
        c, lc = make(64, s=1)  # another shape on the decode profile, then A again
        c()
        a()
        print("A after a one-token context:", float((la - first).abs().max()), flush=True)
        b()
        # A with its shapes set again
        a_ctx = a.ctx
        for i in range(a_ctx.engine.num_io_tensors):
            name = a_ctx.engine.get_tensor_name(i)
            if a_ctx.engine.get_tensor_mode(name).name == "INPUT":
                a_ctx.set_input_shape(name, tuple(a_ctx.get_tensor_shape(name)))
        a()
        print("A after B, shapes set again:", float((la - first).abs().max()), flush=True)
        d, ld = make(64)  # a fresh context like A
        d()
        print("fresh context like A:", float((ld - first).abs().max()), flush=True)


if __name__ == "__main__":
    main()
