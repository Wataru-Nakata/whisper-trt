# (development record) Encoder ahead on a side stream: tests, LibriSpeech dev-clean and en/ja clips, streamed.
set -uo pipefail
P=/groups/gai51740/acc12576tt/github.com/whisper_trt/outputs
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -2
echo "=== stream"; $PY scripts/parity.py $P/job-2777982/waves.npz $P/job-2777982/ref.json --configs 5x8 5x16 --repeat 2 --stream 2>&1 | grep -E "^beam|float32/beam5/b8|Traceback|Error"
for lang in en ja; do
  echo "=== clips $lang stream"; $PY scripts/parity.py $P/job-2778704/clips-$lang.npz $P/job-2778704/ref-$lang.json --lang $lang --configs 5x8 5x16 --repeat 2 --stream 2>&1 | grep -E "^beam|b8 \(|Traceback|Error"
done
