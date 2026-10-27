# (development record) Lagged asynchronous checks without host syncs in Search._start, and the Triton
# decoder (cuBLAS GEMMs), streamed: tests, remaining host syncs, wall time and results on YODAS ja / en and
# LibriSpeech test-clean.
set -uo pipefail
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q tests/test_search.py tests/test_sampling.py 2>&1 | tail -3
echo "=== syncs"; $PY scripts/sync_check.py $E/yodas-ja.npz --lang ja 2>&1 | grep -E "^ +[0-9]+x|Traceback|Error" | cut -c1-200
for spec in yodas-ja:ja yodas-en:en ls-test-clean:en; do
  for dec in tensorrt triton; do
    echo "=== ${spec%%:*} decoder $dec"
    sch="1,1"; [ $dec = tensorrt ] && sch="4,0 4,0 1,1 1,2"
    $PY scripts/profile_stream.py $E/${spec%%:*}.npz --lang ${spec#*:} --batch 8 32 --decoder $dec --schedules $sch 2>&1 | grep -E "^batch|Traceback|Error"
  done
done
