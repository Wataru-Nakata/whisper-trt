"""whisper-trt build | transcribe | detect-language"""

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="whisper-trt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build (or find) the TensorRT engines")
    t = sub.add_parser("transcribe", help="transcribe up-to-30 s recordings, each on its own")
    d = sub.add_parser("detect-language", help="language probabilities of each recording's first 30 s")
    for p in (b, t, d):
        p.add_argument("--model", default="openai/whisper-large-v3")
        p.add_argument("--backend", default="tensorrt", choices=("tensorrt", "torch"))
        p.add_argument("--max-batch", type=int, default=16)
        p.add_argument("--decoder", default=None, choices=("triton", "tensorrt"),
                       help="triton: cuBLAS GEMMs + Triton kernels (the default for float16); tensorrt: the TensorRT "
                            "decoder engine")
        p.add_argument("--fp8", action="store_true",
                       help="FP8 GEMMs in the encoder's MLPs (Hopper+; faster, different decodes: see the README)")
    for p in (t, d):
        p.add_argument("audio", nargs="+")
        p.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--language", required=True)
    t.add_argument("--task", default="transcribe", choices=("transcribe", "translate"))
    t.add_argument("--beam-size", type=int, default=5)
    t.add_argument("--json", action="store_true", help="every value per recording, as JSON lines")
    t.add_argument("--timestamps", action="store_true", help="decode with timestamps: segments per recording")
    args = ap.parse_args(argv)

    from . import Whisper

    model = Whisper.from_pretrained(args.model, backend=args.backend, max_batch=args.max_batch, decoder=args.decoder,
                                    fp8=args.fp8)
    if args.cmd == "build":
        print("engines ready", file=sys.stderr)
    elif args.cmd == "transcribe":
        out = model.decode_windows(args.audio, args.language, args.task, beam_size=args.beam_size,
                                   batch_size=args.batch_size, without_timestamps=not args.timestamps)
        for path, d in zip(args.audio, out):
            if args.json:
                row = {"audio": path, "text": d.text, "avg_logprob": d.avg_logprob, "no_speech_prob": d.no_speech_prob,
                       "compression_ratio": d.compression_ratio, "temperature": d.temperature,
                       "reached_max_length": d.reached_max_length}
                if args.timestamps:
                    row["segments"] = [{k: seg[k] for k in ("start", "end", "text")} for seg in d.segments]
                print(json.dumps(row, ensure_ascii=False))
            elif args.timestamps:
                for seg in d.segments:
                    print(f"{path}\t[{seg['start']:.2f} -> {seg['end']:.2f}]\t{seg['text'].strip()}")
            else:
                print(f"{path}\t{d.text}")
    else:
        for i in range(0, len(args.audio), args.batch_size):
            paths = args.audio[i : i + args.batch_size]
            probs = model.detect_language(model.encode(model.features(paths)))
            for path, p in zip(paths, probs):
                print(json.dumps({"audio": path, "top": p[:5]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
