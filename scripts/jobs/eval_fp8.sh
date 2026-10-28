# (development record) The FP8 encoder (fp8=True: the MLPs in FP8) at scale: every clean set and the 18 YODAS
# languages against faster-whisper fp16 (streamed, 8 and 32 slots), the six YODAS languages with fp32
# references, caption CER, and timing against the fp16 encoder (best of two passes).
set -uo pipefail
E=$REPO/.cache/eval
X="--build fp8=true"
B=$X
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -2
for spec in ls-test-clean:en jvs-test:ja fleurs-de:de fleurs-zh:zh fleurs-es:es fleurs-en:en fleurs-fr:fr fleurs-ja:ja fleurs-ru:ru; do
  name=${spec%%:*}; lang=${spec#*:}
  echo "=== $name"
  $PY scripts/parity.py $E/$name.npz $E/ref-$name.json --lang $lang --configs 5x8 5x32 --repeat 0 --stream $B \
    --out $OUT/ours-$name.json --report $OUT/report-$name.json 2>&1 | grep -E "^beam|tokens|silence|fallback|limit|Traceback|Error"
done
for lang in ja en de zh ko hi ta el ar ru es fr pt vi th it id tr; do
  echo "=== yodas $lang"
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref-yodas-$lang.json --lang $lang --configs 5x8 5x32 --repeat 0 --stream $B \
    --out $OUT/ours-yodas-$lang.json --report $OUT/report-yodas-$lang.json 2>&1 | grep -E "^beam|tokens|silence|fallback|limit|Traceback|Error"
done
for lang in ja en de hi ta el; do
  echo "=== yodas $lang vs fp32"
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --configs 5x8 --repeat 0 --stream $B \
    2>&1 | grep -E "^beam|vs float|Traceback|Error"
done
echo "=== captions"
uv run --no-project --with pyarrow python scripts/yodas_captions.py --eval-dir $E --ours "$OUT/ours-yodas-{lang}.json" --langs ja en de zh ko hi ta el ar ru es fr pt vi th it id tr 2>&1 | tail -25
echo "=== timing"
for spec in ls-test-clean:en jvs-test:ja yodas-ja:ja yodas-en:en yodas-hi:hi; do
  name=${spec%%:*}; lang=${spec#*:}
  for enc in fp16 fp8; do
    b=""; [ $enc = fp8 ] && b="--build fp8=true"
    $PY scripts/parity.py $E/$name.npz $E/ref-$name.json --lang $lang --configs 5x8 5x16 5x32 --repeat 1 --stream \
      $b 2>&1 | grep -E "^beam|memory" | sed "s/^/$name $enc /"
  done
done
