# (development record) Decoding with timestamps at scale: faster-whisper references with timestamps (fp16 for every
# set, fp32 for six YODAS languages), then our decodes against them (streamed, 8 and 32 slots) and the fp32
# PyTorch backend against faster-whisper fp32.
set -uo pipefail
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -2
for spec in ls-test-clean:en jvs-test:ja fleurs-de:de fleurs-zh:zh fleurs-es:es fleurs-en:en fleurs-fr:fr fleurs-ja:ja fleurs-ru:ru \
            yodas-ja:ja yodas-en:en yodas-de:de yodas-zh:zh yodas-ko:ko yodas-hi:hi yodas-ta:ta yodas-el:el yodas-ar:ar \
            yodas-ru:ru yodas-es:es yodas-fr:fr yodas-pt:pt yodas-vi:vi yodas-th:th yodas-it:it yodas-id:id yodas-tr:tr; do
  name=${spec%%:*}; lang=${spec#*:}
  types=float16; case $name in yodas-ja|yodas-en|yodas-de|yodas-hi|yodas-ta|yodas-el) types="float32 float16";; esac
  [ -f $E/tsref-$name.json ] || $REF_PY scripts/ref_ct2.py $E/$name.npz $E/tsref-$name.json --lang $lang --timestamps \
    --compute-types $types --configs 5x8 --save-features 0 --repeat 0 2>&1 | tail -2
  echo "=== $name"
  $PY scripts/parity.py $E/$name.npz $E/tsref-$name.json --lang $lang --configs 5x8 5x32 --repeat 0 --stream --timestamps \
    --out $OUT/ours-$name.json --report $OUT/report-$name.json 2>&1 | grep -E "^beam|tokens|silence|fallback|limit|Traceback|Error"
done
for lang in ja en de hi ta el; do
  echo "=== yodas $lang fp32"
  $PY scripts/parity.py $E/yodas-$lang.npz $E/tsref-yodas-$lang.json --lang $lang --backend torch --dtype float32 \
    --configs 5x8 --repeat 0 --stream --timestamps 2>&1 | grep -E "vs float32|Traceback|Error"
done
