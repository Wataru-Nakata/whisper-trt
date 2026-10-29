"""Where word alignment's time goes, on the first 30 s window of long-*.npz recordings (decoded with timestamps):
done on the host as before (the window's encode, the PyTorch decoder call op by op, the scores' normalization
and filter with the copy to the host, the dynamic time warping in numpy, faster-whisper's word rules), each part
timed alone (synchronized); and alignment.Aligner's (the decoder call in a CUDA graph, the time warping in a
Triton kernel), the same words?

    python scripts/profile_align.py WAVES.npz --lang ja [--n 20]
"""

import argparse
import json
import time

import numpy as np
import torch

from whisper_trt import Whisper
from whisper_trt.alignment import (
    add_word_timestamps,
    find_alignment,
    first_frames,
    median_filter,
    negative_dtw,
)
from whisper_trt.longform import recording_features
from whisper_trt.segments import split_by_timestamps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("waves")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    data = np.load(args.waves)
    meta = json.loads(str(data["meta"]))
    model = Whisper.from_pretrained()
    c, tok = model.cfg, model.tokenizer
    feats = []
    for i in range(min(args.n, len(meta))):
        wave = torch.from_numpy(data[f"w{i}"].astype(np.float32) / 32768)
        f = recording_features(model, wave)
        feats.append(f[:, :3000])
    out = model.decode(feats, model.prompt(args.lang, without_timestamps=False), batch_size=8)
    decoder = model._torch_decoder()
    aligner = model._aligner(decoder)
    sot_sequence = tok.sot_sequence(args.lang, "transcribe")
    times = {k: [] for k in ("encode", "decoder", "normalize+copy", "dtw", "words")}
    new_times, same = [], []
    n_tokens = []

    def tick() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    for _ in range(2):  # (the first pass warms up)
        for k in times:
            times[k].clear()
        n_tokens.clear()
        new_times.clear()
        same.clear()
        for f, d in zip(feats, out):
            segs, _, _ = split_by_timestamps(d.tokens, c.timestamp_begin, 0.0, 3000, 30.0, 0)
            text = [t for s in segs for t in s["tokens"] if t < tok.eot]
            if not text:
                continue
            n_tokens.append(len(text))
            t0 = tick()
            cross = model.encode(f[None].to(model.device))[:, :, 0]
            t1 = tick()
            with torch.inference_mode():
                tokens = list(sot_sequence) + [c.no_timestamps] + text + [c.eot]
                logits, scores = decoder(torch.tensor([tokens], device=model.device), cross[:, :, None],
                                         heads=c.alignment_heads)
                t2 = tick()
                probs = logits[0, :, : c.eot].float().softmax(-1)
                start = len(sot_sequence)
                idx = torch.arange(len(text), device=model.device)
                text_probs = probs[start + idx, torch.tensor(text, device=model.device)].tolist()
                w = scores[0, :, :, :1500].float().softmax(-1)
                mean = w.mean(-2, keepdim=True)
                w = (w - mean) / ((w - mean) ** 2).mean(-2, keepdim=True).sqrt()
                w = median_filter(w, 7).mean(0)
                matrix = w[start : start + len(text) + 1].cpu().numpy()
            t3 = tick()
            first = first_frames(negative_dtw(matrix), matrix.shape[0])
            t4 = tick()
            alignment = find_alignment(tok, args.lang, text, first, text_probs)
            add_word_timestamps(segs, tok, args.lang, alignment, "\"'“¿([{-", "\"'.。,，!！?？:：”)]}、", 0.0)
            t5 = tick()
            for k, a, b in zip(times, (t0, t1, t2, t3, t4), (t1, t2, t3, t4, t5)):
                times[k].append((b - a) * 1000)
            t6 = tick()
            handle = aligner.submit(cross, None, sot_sequence, text, 3000)
            first2, probs2 = aligner.result(handle)
            new_times.append((tick() - t6) * 1000)
            same.append(first2.tolist() == first.tolist())
    print(f"{len(n_tokens)} windows, text tokens {np.mean(n_tokens):.0f} on average (max {max(n_tokens)})")
    for k, v in times.items():
        print(f"  {k:16s} {np.mean(v):7.2f} ms a window (median {np.median(v):.2f}, max {max(v):.2f})")
    print(f"  total            {sum(np.mean(v) for v in times.values()):7.2f} ms a window")
    print(f"  Aligner          {np.mean(new_times):7.2f} ms a window (on the GPU, the host not waiting in transcribe); "
          f"the same first frames on {sum(same)}/{len(same)} windows")


if __name__ == "__main__":
    main()
