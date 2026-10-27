# (development record) Semantic check at scale: our PyTorch fp32 against faster-whisper fp32 on YODAS windows
# (should be identical); faster-whisper fp16 vs fp32 (its own noise floor); our TensorRT fp16 against both.
set -uo pipefail
E=$REPO/.cache/eval
for lang in ${LANGS:-ja en de hi ta el ar}; do
  echo "=== yodas $lang"
  [ -f $E/ref32-yodas-$lang.json ] || $REF_PY scripts/ref_ct2.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang \
    --compute-types float16 float32 --configs 5x8 --save-features 0 --repeat 0 | tail -2
  $PY scripts/ct2_vs_ct2.py $E/ref32-yodas-$lang.json
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --backend torch --dtype float32 \
    --configs 5x8 --repeat 0 --stream --report $OUT/report32-yodas-$lang.json 2>&1 | grep -E "^beam|tokens|silence|fallback|limit|differing|Traceback|Error"
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --configs 5x8 --repeat 0 --stream \
    --report $OUT/report16-yodas-$lang.json 2>&1 | grep -E "^beam|tokens|silence|fallback|limit|differing|Traceback|Error"
done
