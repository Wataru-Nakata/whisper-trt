# (development record) Long-form fp32 one recording at a time: the same decodes as many recordings together?
set -uo pipefail
E=$REPO/.cache/eval
$PY scripts/longform_parity.py $E/long-ja.npz $E/lref-ja.json --lang ja --batch 8 --single --backend torch \
  --dtype float32 --out $OUT/ours32-1-ja.json 2>&1 | grep -E "RTFx|vs |Error|Trace"
