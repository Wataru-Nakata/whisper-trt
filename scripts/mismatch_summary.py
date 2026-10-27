"""Summarize parity.py --report files: how the differing windows differ.

    python scripts/mismatch_summary.py outputs/job-N/report-*.json [--run float16/beam5/b8] [--examples 3]

Per report: differing windows by kind -- "edit" (character difference under 10%: a near-tie of punctuation,
casing or a word), "length" (one side at least 30% longer: an early end or a repetition loop), "other" --,
decision flips (silence rule, fallback, token limit), and the windows that differ most.
"""

import argparse
import json
import re
from pathlib import Path


def chars(text: str) -> list[str]:
    return list(re.sub(r"[\W_]+", "", text.lower()))


def distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--run", default="beam5/b8/stream vs float16/beam5/b8", help="report key (ours vs reference)")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    for path in args.reports:
        rows = json.load(open(path)).get(args.run)
        if rows is None:
            print(f"{path}: no {args.run!r}; keys {list(json.load(open(path)))}")
            continue
        kinds = {"edit": 0, "length": 0, "other": 0}
        flips = {"silence": 0, "fallback": 0, "limit": 0}
        scored = []
        for r in rows:
            a, b = chars(r["ours"]), chars(r["ref"])
            d = distance(a, b) / max(1, len(b), len(a))
            la, lb = r["len"]
            if max(la, lb) >= 1.3 * max(1, min(la, lb)) and abs(la - lb) >= 5:
                kinds["length"] += 1
            elif d < 0.1:
                kinds["edit"] += 1
            else:
                kinds["other"] += 1
            for f in flips:
                flips[f] += r["ours_decisions"][f] != r["ref_decisions"][f]
            scored.append((d, r))
        name = Path(path).stem.removeprefix("report-")
        print(f"{name}: {len(rows)} differing windows: {kinds}; decision flips {flips}")
        for d, r in sorted(scored, key=lambda x: -x[0])[: args.examples]:
            print(f"   [{r['i']}] {r['id']} tokens ours/ref {r['len']}, difference {d:.0%}")
            print(f"      ours: {r['ours'][:160]!r}")
            print(f"      ref:  {r['ref'][:160]!r}")


if __name__ == "__main__":
    main()
