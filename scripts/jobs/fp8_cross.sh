# (development record) FP8 cross-attention keys/values, emulated: agreement with faster-whisper fp32/fp16.
set -uo pipefail
E=$REPO/.cache/eval
for lang in ja en de ta hi; do
  echo "=== yodas $lang"
  $PY scripts/fp8_cross_agree.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --format ${FORMATS:-fp8} ${EXTRA:-} --out $OUT/fp8-yodas-$lang.json 2>&1 | grep -E "^(fp|int)|Traceback|Error"
done
