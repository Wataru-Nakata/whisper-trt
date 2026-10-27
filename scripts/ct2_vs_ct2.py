"""Calibration: how often faster-whisper disagrees with itself across precisions (fp16 vs fp32 runs of one
reference file from scripts/ref_ct2.py), the noise floor for our fp16 agreement.

    python scripts/ct2_vs_ct2.py REF.json [--a float16/beam5/b8 --b float32/beam5/b8]
"""

import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("refs", nargs="+")
ap.add_argument("--a", default="float16/beam5/b8")
ap.add_argument("--b", default="float32/beam5/b8")
args = ap.parse_args()
for path in args.refs:
    runs = json.load(open(path))["runs"]
    if args.a not in runs or args.b not in runs:
        continue
    ra, rb = runs[args.a]["rows"], runs[args.b]["rows"]
    same = sum(x["tokens"] == y["tokens"] for x, y in zip(ra, rb))
    print(f"{path.split('/')[-1]}: {args.a} vs {args.b}: tokens {same}/{len(ra)}")
