"""Does a TensorRT decoder context write past the scratch memory it is given?

Prints the engine's device memory sizes (overall and per profile), then runs a prefill context with scratch of
the reported size followed by a guard region, and reports whether the guard changed.

    python scripts/scratch_guard.py WAVES.npz
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
    trt = Whisper.from_pretrained(args.model, backend="tensorrt")
    be, cfg = trt.backend, trt.cfg
    L, H, dk, V = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size
    for name, eng in (("encoder", be.encoder), ("decoder", be.decoder)):
        sizes = [eng.get_device_memory_size_for_profile_v2(i) for i in range(eng.num_optimization_profiles)]
        print(f"{name}: device_memory_size_v2 {eng.device_memory_size_v2 / 2**20:.1f} MiB; per profile "
              f"{[round(s / 2**20, 1) for s in sizes]} MiB", flush=True)
    import tensorrt

    eng = be.decoder
    with torch.cuda.stream(trt.stream):
        trt.encode(trt.features([torch.from_numpy(data["w0"])]))
        for profile, (n, s) in ((2, (1, 4)), (0, (1, 1)), (2, (8, 4)), (1, (8, 1))):
            size = eng.device_memory_size_v2
            guard = 256 << 20
            buf = torch.full((size + guard,), 0xAB, dtype=torch.uint8, device="cuda")
            ctx = eng.create_execution_context(tensorrt.ExecutionContextAllocationStrategy.USER_MANAGED)
            ctx.set_device_memory(buf.data_ptr(), size)
            ctx.set_optimization_profile_async(profile, torch.cuda.current_stream().cuda_stream)
            C = 64
            cache = torch.zeros(L, 2, n, H, C, dk, dtype=torch.float16, device="cuda")
            src = torch.zeros(n, C, dtype=torch.int32, device="cuda")
            tok = torch.full((n, s), 50258, dtype=torch.long, device="cuda")
            off = torch.zeros(n, dtype=torch.long, device="cuda")
            g = 1 << 20  # guard elements after each output
            logits_buf = torch.full((n * s * V + g,), 7.0, device="cuda")
            kv_buf = torch.full((L * 2 * n * H * s * dk + g,), 7.0, dtype=torch.float16, device="cuda")
            logits = logits_buf[: n * s * V].view(n, s, V)
            new_kv = kv_buf[: L * 2 * n * H * s * dk].view(L, 2, n, H, s, dk)
            cross = trt.cross(1) if n == 1 else trt.cross(1).expand(-1, -1, n, -1, -1, -1).contiguous()
            be._bind(ctx, "tokens", tok, True)
            be._bind(ctx, "offset", off, True)
            be._bind(ctx, "src", src, True)
            for i in range(L):
                be._bind(ctx, f"self_k_{i}", cache[i, 0], True)
                be._bind(ctx, f"self_v_{i}", cache[i, 1], True)
                be._bind(ctx, f"cross_k_{i}", cross[i, 0], True)
                be._bind(ctx, f"cross_v_{i}", cross[i, 1], True)
            be._bind(ctx, "logits", logits, False)
            be._bind(ctx, "new_kv", new_kv, False)
            be._execute(ctx, "decoder")
            torch.cuda.synchronize()
            touched = (buf[size:] != 0xAB).nonzero()
            reach = int(touched[-1]) + 1 if len(touched) else 0
            lg = (logits_buf[n * s * V :] != 7.0).nonzero()
            kv = (kv_buf[L * 2 * n * H * s * dk :] != 7.0).nonzero()
            print(f"profile {profile} rows {n} tokens {s}: scratch {size / 2**20:.1f} MiB, "
                  f"profile needs {eng.get_device_memory_size_for_profile_v2(profile) / 2**20:.1f} MiB, "
                  f"wrote {reach / 2**20:.2f} MiB past the scratch; logits: {len(lg)} elements past the end "
                  f"(last {int(lg[-1]) + 1 if len(lg) else 0}); new_kv: {len(kv)} past the end "
                  f"(last {int(kv[-1]) + 1 if len(kv) else 0})", flush=True)
            del ctx


if __name__ == "__main__":
    main()
