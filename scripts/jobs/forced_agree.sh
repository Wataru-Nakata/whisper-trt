# (development record) In-graph prompts: TensorRT fp16 and PyTorch fp32 against faster-whisper fp32 (YODAS).
set -uo pipefail
E=$REPO/.cache/eval
for lang in ja en de hi ta; do
  echo "=== yodas $lang"
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --configs 5x8 --repeat 0 --stream 2>&1 | grep -E "vs float32" | sed 's/^/  trt16 /'
done
for lang in ja ta; do
  echo "=== torch32 yodas $lang"
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --backend torch --dtype float32 --configs 5x8 --repeat 0 --stream 2>&1 | grep -E "vs float32|Traceback|Error"
done
