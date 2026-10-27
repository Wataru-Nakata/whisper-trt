# (development record) Refill with batched prefill: tests, LibriSpeech dev-clean fixed vs streamed, en clips.
set -uo pipefail
P=/groups/gai51740/acc12576tt/github.com/whisper_trt/outputs
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -3
echo "=== fixed"; $PY scripts/parity.py $P/job-2777982/waves.npz $P/job-2777982/ref.json --configs 1x1 5x1 5x8 --repeat 2 2>&1 | grep -E "^beam|float32/beam1/b1|float32/beam5/b8|Traceback|Error"
echo "=== stream"; $PY scripts/parity.py $P/job-2777982/waves.npz $P/job-2777982/ref.json --configs 5x8 5x16 --repeat 2 --stream 2>&1 | grep -E "^beam|float32/beam5/b8|Traceback|Error"
echo "=== clips en stream"; $PY scripts/parity.py $P/job-2778704/clips-en.npz $P/job-2778704/ref-en.json --lang en --configs 5x8 5x16 --repeat 2 --stream 2>&1 | grep -E "^beam|b8 \(|Traceback|Error"
