# (development record) Logits plugin (fp32 accumulation): tests, decoder step time, agreement with fp32.
set -uo pipefail
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -3
echo "=== build [default: logits plugin]"
$PY scripts/bench_decoder.py 2>&1 | grep -E "^rows|Traceback|Error|error"
for lang in ja ta en; do
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --configs 5x8 --repeat 0 --stream 2>&1 | grep -E "vs float32|Traceback|Error" | sed "s/^/  $lang /"
done
