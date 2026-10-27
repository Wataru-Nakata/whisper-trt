# (development record) Encoder vs decoder precision: TensorRT / PyTorch fp16 halves against faster-whisper fp32.
set -uo pipefail
E=$REPO/.cache/eval
for lang in ${LANGS:-ja ta}; do
  echo "=== yodas $lang"; $PY scripts/hybrid_agree.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang 2>&1 | grep -E "^encoder|Traceback|Error"
done
