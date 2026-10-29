# (development record) Where word alignment's time goes.
set -uo pipefail
for s in ja:ja en:en; do
  echo "=== ${s%%:*}"
  $PY scripts/profile_align.py $REPO/.cache/eval/long-${s%%:*}.npz --lang ${s#*:} --n 20 2>&1 | grep -E "windows|ms a window|Error|Trace"
done
