# (development record) Refill: unit tests, then LibriSpeech dev-clean fixed batches vs streamed, and clips.
set -uo pipefail
P=/groups/gai51740/acc12576tt/github.com/whisper_trt/outputs
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -4
echo "=== fixed"; $PY scripts/parity.py $P/job-2777982/waves.npz $P/job-2777982/ref.json --configs 1x1 5x8 --repeat 1 2>&1 | grep -E "^beam|tokens|Traceback|Error"
echo "=== stream"; $PY scripts/parity.py $P/job-2777982/waves.npz $P/job-2777982/ref.json --configs 1x1 5x8 5x16 --repeat 1 --stream 2>&1 | grep -E "^beam|tokens|Traceback|Error"
for lang in en ja; do
  echo "=== clips $lang stream"; $PY scripts/parity.py $P/job-2778704/clips-$lang.npz $P/job-2778704/ref-$lang.json --lang $lang --configs 5x8 5x16 --repeat 1 --stream 2>&1 | grep -E "^beam|tokens|Traceback|Error"
done
