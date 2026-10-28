# (development record) The rest of eval_ts.sh: YODAS tr, and the fp32 PyTorch backend with timestamps against
# faster-whisper fp32.
set -uo pipefail
E=$REPO/.cache/eval
echo "=== yodas-tr"
$PY scripts/parity.py $E/yodas-tr.npz $E/tsref-yodas-tr.json --lang tr --configs 5x8 5x32 --repeat 0 --stream --timestamps \
  --out $OUT/ours-yodas-tr.json 2>&1 | grep -E "^beam|tokens|silence|fallback|limit|Traceback|Error"
for lang in ja en de hi ta el; do
  echo "=== yodas $lang fp32"
  $PY scripts/parity.py $E/yodas-$lang.npz $E/tsref-yodas-$lang.json --lang $lang --backend torch --dtype float32 \
    --configs 5x8 --repeat 0 --stream --timestamps 2>&1 | grep -E "vs float32|Traceback|Error"
done
