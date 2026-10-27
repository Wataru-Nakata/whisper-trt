"""How a streamed decode's step splits: the decoder call against the rest of Search._body (cache writes, no-speech,
token choice, beam bookkeeping), each captured in a CUDA graph and timed alone in the state a decode left; and
the rest's kernels (PyTorch profiler, eager).

    python scripts/profile_body.py WAVES.npz --lang ja [--batch 8 32] [--decoder tensorrt|triton]
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from whisper_trt import Whisper
from whisper_trt.search import _capture


def replay_ms(graph, reps: int = 50) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    a.record()
    for _ in range(reps):
        graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--batch", nargs="+", type=int, default=[8, 32])
    ap.add_argument("--decoder", default="tensorrt", choices=("tensorrt", "triton"))
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    waves = []
    for i in range(len(meta)):
        w = data[f"w{i}"]
        waves.append(torch.from_numpy(w.astype(np.float32) / 32768 if w.dtype == np.int16 else w))
    model = Whisper.from_pretrained(decoder=args.decoder)
    prompt = model.prompt(args.lang)
    for batch in args.batch:
        model.decode(waves[: batch * 3], prompt, beam_size=5, batch_size=batch)
        search = model._search("beam", batch, 5, len(prompt))
        with torch.cuda.stream(model.stream):
            search.pos.fill_(80)  # (a typical fill: the decoder reads 80 cached positions per row)
            search.done.fill_(False)
            body = replay_ms(search._graph)
            step = replay_ms(_capture(search._step))
            fn = search._step
            search._step = lambda: None
            rest = replay_ms(_capture(search._body))
            search._body()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    search._body()
                torch.cuda.synchronize()
            search._step = fn
        by = defaultdict(lambda: [0.0, 0])
        for e in prof.events():
            if e.device_type.name == "CUDA":
                by[e.name][0] += e.device_time_total / 10
                by[e.name][1] += 1
        print(f"batch {batch} ({args.decoder}): step graph {body:.3f} ms = decoder {step:.3f} ms + rest "
              f"{rest:.3f} ms (alone; {body - step:.3f} ms in the step)", flush=True)
        for name, (us, cnt) in sorted(by.items(), key=lambda x: -x[1][0])[:14]:
            print(f"   {us / 1000:.3f} ms  {cnt // 10:3d}x  {name[:110]}", flush=True)


if __name__ == "__main__":
    main()
