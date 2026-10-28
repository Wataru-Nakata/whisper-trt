# (development record) Long-form fp32 (PyTorch backend) outputs for the comparison with faster-whisper fp32.
set -uo pipefail
E=$REPO/.cache/eval
$PY scripts/longform_parity.py $E/long-ja.npz $E/lref-ja.json --lang ja --batch 8 --backend torch --dtype float32 \
  --out $OUT/ours32-ja.json 2>&1 | grep -E "RTFx|vs |Error|Trace"
