# (development record) Language detection of the long recordings (transcribe's language=None), fp32, against
# faster-whisper's detect_language.
set -uo pipefail
E=$REPO/.cache/eval
$REF_PY scripts/lid_long.py --ref --model $REPO/.cache/fw-large-v3 $OUT/lid-ref.jsonl $E/long-ja.npz $E/long-en.npz \
  $E/long-ls.npz 2>&1 | grep -E "Error|Trace"
$PY scripts/lid_long.py $OUT/lid-ours.jsonl $E/long-ja.npz $E/long-en.npz $E/long-ls.npz --against $OUT/lid-ref.jsonl \
  2>&1 | grep -E "recordings|Error|Trace"
