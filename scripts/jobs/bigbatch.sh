# (development record) More slots: engines for 32 windows x 5 beams, streamed beam 5 on clips at 16/24/32.
set -uo pipefail
P=/groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2778704
for lang in en ja; do
  echo "=== clips $lang"
  $PY scripts/parity.py $P/clips-$lang.npz $P/ref-$lang.json --lang $lang --configs 5x16 5x24 5x32 --repeat 2 --stream \
    --build max_batch=32 max_rows=160 2>&1 | grep -E "^beam|b8 \(|Traceback|Error|memory"
done
