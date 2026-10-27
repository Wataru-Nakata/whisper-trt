# (development record) fp32 vocabulary projection: decoder step time and agreement with faster-whisper fp32.
set -uo pipefail
E=$REPO/.cache/eval
for b in "" "logits32=true"; do
  echo "=== build [$b]"
  $PY scripts/bench_decoder.py --build $b 2>&1 | grep -E "^rows|Traceback|Error"
  for lang in ja ta en; do
    $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --configs 5x8 --repeat 0 --stream --build $b 2>&1 | grep -E "vs float32" | sed "s/^/  $lang /"
  done
done
