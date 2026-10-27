"""Stream ordering around TensorRT context creation: are tensors initialized on the model's stream visible to
the next kernel on that stream, before and after creating a decoder context?

    python scripts/stream_order.py
"""

import torch

from whisper_trt import Whisper


def main() -> None:
    trt = Whisper.from_pretrained("openai/whisper-large-v3", backend="tensorrt", decoder="tensorrt")
    be, cfg = trt.backend, trt.cfg
    L, H, dk, V = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size
    with torch.cuda.stream(trt.stream):
        print("current stream is the model's:", torch.cuda.current_stream() == trt.stream, flush=True)
        for trial, make_ctx in ((0, False), (1, True), (2, True)):
            src = torch.zeros(1, 64, dtype=torch.int32, device="cuda")
            tok = torch.full((1, 4), 7, dtype=torch.long, device="cuda")
            if make_ctx:
                cache = torch.zeros(L, 2, 1, H, 64, dk, dtype=torch.float16, device="cuda")
                off = torch.zeros(1, dtype=torch.long, device="cuda")
                logits = torch.zeros(1, 4, V, device="cuda")
                new_kv = torch.zeros(L, 2, 1, H, 4, dk, dtype=torch.float16, device="cuda")
                be.step_fn(tok, off, cache, logits, new_kv, src, cross=trt.cross(1))
            a, b = src.clone(), tok.clone()
            torch.cuda.synchronize()
            print(f"trial {trial} (context created: {make_ctx}): src clone all zero {bool((a == 0).all())}, "
                  f"tok clone all 7 {bool((b == 7).all())}; src now zero {bool((src == 0).all())}", flush=True)


if __name__ == "__main__":
    main()
