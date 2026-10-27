"""Quality on YODAS windows against the videos' own captions (approximate references): the caption segments whose
midpoint falls inside a window, joined. Compares the character error rate of our decodes and faster-whisper's.

    python scripts/yodas_captions.py --eval-dir .cache/eval --ours outputs/job-N/ours-yodas-{lang}.json \
        --langs ja en ... [--run beam5/b8/stream] [--ref-run float16/beam5/b8]
"""

import argparse
import json
import re
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/groups/gai51740/acc12576tt/datasets/yodas3/data")
FROM_WHISPER = {"he": "iw", "jw": "jv", "tl": "fil"}
NO_SPACE = {"ja", "zh", "th", "yue", "ko"}


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


def lengths(npz: Path) -> list[float]:
    """Seconds of each window, from the archive's member sizes (int16 arrays, without loading them)."""
    with zipfile.ZipFile(npz) as z:
        sizes = {i.filename[:-4]: i.file_size for i in z.infolist() if i.filename.startswith("w")}
    return [(sizes[f"w{i}"] - 128) / 2 / 16000 for i in range(len(sizes))]


def captions(lang: str, ids: set[str]) -> dict[str, list[dict]]:
    out = {}
    for f in sorted((ROOT / FROM_WHISPER.get(lang, lang) / "metadata").glob("*.parquet"))[:1]:
        t = pq.read_table(f, columns=["id", "transcript"]).to_pylist()
        for row in t:
            if row["id"] in ids and row["transcript"]:
                out[row["id"]] = json.loads(row["transcript"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=".cache/eval")
    ap.add_argument("--ours", required=True, help="path with {lang}")
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--run", default="beam5/b8/stream")
    ap.add_argument("--ref-run", default="float16/beam5/b8")
    args = ap.parse_args()
    E = Path(args.eval_dir)
    total = {"ours": [0, 0], "ref": [0, 0]}
    for lang in args.langs:
        npz = E / f"yodas-{lang}.npz"
        with zipfile.ZipFile(npz) as z:
            meta = json.loads(str(__import__("numpy").load(z.open("meta.npy"), allow_pickle=False)))
        secs = lengths(npz)
        rec = {m["id"].split("@")[0] for m in meta}
        caps = captions(lang, rec)
        ours = json.load(open(args.ours.format(lang=lang)))[args.run]["rows"]
        ref = json.load(open(E / f"ref-yodas-{lang}.json"))["runs"][args.ref_run]["rows"]
        err = {"ours": [0, 0], "ref": [0, 0]}
        used = 0
        for m, n, a, b in zip(meta, secs, ours, ref):
            vid, start = m["id"].split("@")
            start = float(start)
            segs = caps.get(vid)
            if not segs:
                continue
            sep = "" if lang in NO_SPACE else " "
            text = sep.join(s["text"] for s in segs if start <= (s["start"] + s["duration"] / 2) / 1000 < start + n)
            target = chars(text)
            if not target:
                continue
            used += 1
            for k, row in (("ours", a), ("ref", b)):
                err[k][0] += distance(chars(row["text"]), target)
                err[k][1] += len(target)
        for k in total:
            total[k][0] += err[k][0]
            total[k][1] += err[k][1]
        print(f"{lang}: {used} windows with captions; CER vs captions: ours {err['ours'][0] / max(1, err['ours'][1]):.2%}, "
              f"faster-whisper {err['ref'][0] / max(1, err['ref'][1]):.2%}", flush=True)
    print(f"all: ours {total['ours'][0] / max(1, total['ours'][1]):.2%}, "
          f"faster-whisper {total['ref'][0] / max(1, total['ref'][1]):.2%}")


if __name__ == "__main__":
    main()
