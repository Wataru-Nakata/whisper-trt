# Large real-world evaluation: YODAS windows (scripts/make_eval_yodas.py) -> faster-whisper fp16 vs ours.
set -uo pipefail
E=$REPO/.cache/eval
LANGS=${LANGS:-ja en de zh ko hi ta el ar ru es fr pt vi th it id tr}
$REF_PY scripts/make_eval_yodas.py $E --langs $LANGS --per-lang ${PER_LANG:-300}
for lang in $LANGS; do
  echo "=== yodas $lang"
  [ -f $E/ref-yodas-$lang.json ] || $REF_PY scripts/ref_ct2.py $E/yodas-$lang.npz $E/ref-yodas-$lang.json --lang $lang \
    --compute-types float16 --configs 5x8 --save-features 0 --repeat 0 | tail -1
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref-yodas-$lang.json --lang $lang --configs 5x8 5x16 --repeat 0 --stream \
    --out $OUT/ours-yodas-$lang.json --report $OUT/report-yodas-$lang.json 2>&1 | grep -v "hidden args\|^loaded\|^features"
done
