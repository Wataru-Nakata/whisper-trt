# (development record) Timestamps: tests, then the PyTorch rules against the fused kernel on YODAS ja.
set -uo pipefail
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q -x --tb=short tests/test_timestamps.py tests/test_search.py 2>&1 | tail -3
for fused in 0 1; do
  echo "=== fused $fused"
  WHISPER_TRT_FUSED_BEAM=$fused $PY scripts/parity.py $E/yodas-ja.npz $E/tsref-yodas-ja.json --lang ja --configs 5x8 --repeat 0 --stream --timestamps \
    --out $OUT/ours-fused$fused.json 2>&1 | grep -E "vs float|Traceback|Error"
done
