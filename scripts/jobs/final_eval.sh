# (development record) Final numbers: agreement on every set (streamed, 8 and 32 slots, one pass) and warm
# timing (best of two passes) on the main sets.
set -uo pipefail
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -2
for spec in ls-test-clean:en jvs-test:ja fleurs-de:de fleurs-zh:zh fleurs-es:es fleurs-en:en fleurs-fr:fr fleurs-ja:ja fleurs-ru:ru; do
  name=${spec%%:*}; lang=${spec#*:}
  echo "=== $name"
  $PY scripts/parity.py $E/$name.npz $E/ref-$name.json --lang $lang --configs 5x8 5x32 --repeat 0 --stream \
    --out $OUT/ours-$name.json --report $OUT/report-$name.json 2>&1 | grep -E "^beam|tokens|silence|fallback|limit|Traceback|Error"
done
for lang in ja en de zh ko hi ta el ar ru es fr pt vi th it id tr; do
  echo "=== yodas $lang"
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref-yodas-$lang.json --lang $lang --configs 5x8 5x32 --repeat 0 --stream \
    --out $OUT/ours-yodas-$lang.json --report $OUT/report-yodas-$lang.json 2>&1 | grep -E "^beam|tokens|silence|fallback|limit|Traceback|Error"
done
echo "=== timing"
for spec in ls-test-clean:en jvs-test:ja yodas-ja:ja yodas-en:en yodas-hi:hi; do
  name=${spec%%:*}; lang=${spec#*:}
  [ -f $E/$name.npz ] || continue
  $PY scripts/parity.py $E/$name.npz $E/ref-$name.json --lang $lang --configs 5x8 5x16 5x32 --repeat 1 --stream 2>&1 | grep -E "^beam" | sed "s/^/$name /"
done
