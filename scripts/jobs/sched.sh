# (development record) Lagged asynchronous checks in Search.stream: unit tests, then wall time and results of
# check schedules (check_every, ahead) on YODAS ja / en and LibriSpeech test-clean.
set -uo pipefail
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q tests/test_search.py tests/test_sampling.py 2>&1 | tail -3
for spec in yodas-ja:ja yodas-en:en ls-test-clean:en; do
  echo "=== ${spec%%:*}"
  $PY scripts/profile_stream.py $E/${spec%%:*}.npz --lang ${spec#*:} --batch 8 32 --schedules 4,0 1,2 1,1 2,1 1,3 2>&1 | grep -E "^batch|Traceback|Error"
done
