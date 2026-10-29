"""whisper-trt build | transcribe | detect-language"""

import argparse
import dataclasses
import json
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="whisper-trt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build (or find) the TensorRT engines")
    t = sub.add_parser("transcribe", help="transcribe up-to-30 s recordings, each on its own (--long: of any length)")
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
    t.add_argument("--language", help="the language (required but with --long: detected per recording)")
    t.add_argument("--task", default="transcribe", choices=("transcribe", "translate"))
    t.add_argument("--beam-size", type=int, default=5)
    t.add_argument("--json", action="store_true", help="every value per recording, as JSON lines")
    t.add_argument("--timestamps", action="store_true", help="decode with timestamps: segments per recording")
    t.add_argument("--long", action="store_true",
                   help="recordings of any length, as faster-whisper's WhisperModel.transcribe (segments)")
    t.add_argument("--word-timestamps", action="store_true", help="with --long: each segment's words")
    args = ap.parse_args(argv)
    if args.cmd == "transcribe" and args.word_timestamps and not args.long:
        ap.error("--word-timestamps needs --long")
    if args.cmd == "transcribe" and not args.long and not args.language:
        ap.error("--language is required (without --long)")

    from . import Whisper

    model = Whisper.from_pretrained(args.model, backend=args.backend, max_batch=args.max_batch, decoder=args.decoder,
                                    fp8=args.fp8)
    if args.cmd == "build":
        print("engines ready", file=sys.stderr)
    elif args.cmd == "transcribe" and args.long:
        out = model.transcribe(args.audio, args.language, args.task, beam_size=args.beam_size,
                               batch_size=args.batch_size, word_timestamps=args.word_timestamps)
        for path, segments in zip(args.audio, out):
            if args.json:
                print(json.dumps({"audio": path, "language": segments.language,
                                  "language_probability": segments.language_probability,
                                  "segments": [dataclasses.asdict(s) for s in segments]}, ensure_ascii=False))
                continue
            if not args.language:
                print(f"{path}\tlanguage {segments.language} ({segments.language_probability:.2f})")
            for s in segments:
                print(f"{path}\t[{s.start:.2f} -> {s.end:.2f}]\t{s.text.strip()}")
                for w in s.words or []:
                    print(f"{path}\t  [{w.start:.2f} -> {w.end:.2f}]\t{w.word.strip()}")
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
