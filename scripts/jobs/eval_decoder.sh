# (development record) A decoder's agreement at scale (DEC=tensorrt|triton): every clean set and the 18 YODAS
# languages against faster-whisper fp16 (streamed, 8 and 32 slots), the six YODAS languages with fp32
# references, caption CER, and timing of both decoders (best of two passes).
set -uo pipefail
E=$REPO/.cache/eval
DEC=${DEC:-triton}
B="--build decoder=$DEC"
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
$PY scripts/yodas_captions.py --eval-dir $E --ours "$OUT/ours-yodas-{lang}.json" --langs ja en de zh ko hi ta el ar ru es fr pt vi th it id tr 2>&1 | tail -25
echo "=== timing"
for spec in ls-test-clean:en jvs-test:ja yodas-ja:ja yodas-en:en yodas-hi:hi; do
  name=${spec%%:*}; lang=${spec#*:}
  for dec in tensorrt triton; do
    $PY scripts/parity.py $E/$name.npz $E/ref-$name.json --lang $lang --configs 5x8 5x16 5x32 --repeat 1 --stream \
      --build decoder=$dec 2>&1 | grep -E "^beam|memory" | sed "s/^/$name $dec /"
  done
done
