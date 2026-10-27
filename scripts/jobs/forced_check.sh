# (development record) Prompts as in-graph steps: tests, LibriSpeech dev-clean parity, streamed time breakdown.
set -uo pipefail
P=/groups/gai51740/acc12576tt/github.com/whisper_trt/outputs
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -3
echo "=== parity"; $PY scripts/parity.py $P/job-2777982/waves.npz $P/job-2777982/ref.json --configs 1x1 5x8 --repeat 1 2>&1 | grep -E "^beam|float32/beam1/b1|float32/beam5/b8|Traceback|Error"
$PY scripts/parity.py $P/job-2777982/waves.npz $P/job-2777982/ref.json --configs 5x8 --repeat 1 --stream 2>&1 | grep -E "^beam|float32/beam5/b8|Traceback|Error"
for spec in yodas-ja:ja yodas-en:en ls-test-clean:en; do
  echo "=== ${spec%%:*}"; $PY scripts/profile_stream.py $E/${spec%%:*}.npz --lang ${spec#*:} --batch 8 16 32 2>&1 | grep -E "^batch|Traceback|Error"
done
echo "=== yodas-ja agreement"; $PY scripts/parity.py $E/yodas-ja.npz $E/ref32-yodas-ja.json --lang ja --configs 5x8 5x32 --repeat 0 --stream 2>&1 | grep -E "^beam|tokens|silence|fallback|limit|Traceback|Error"
