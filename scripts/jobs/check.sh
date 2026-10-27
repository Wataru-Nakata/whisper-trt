# Unit tests, then (PREV=outputs/job-N with waves.npz/ref.json): torch fp32 parity, engine check (builds the
# engines if needed), TensorRT fp16 parity.
set -uo pipefail
PREV=${PREV:?set PREV to the output dir of a parity job}
echo "=== pytest"; $PY -m pytest -q -x tests 2>&1 | tail -15
for step in ${STEPS:-torch32 check trt16}; do
  echo "=== $step"
  case $step in
    torch32) $PY scripts/parity.py $PREV/waves.npz $PREV/ref.json --backend torch --dtype float32 --out $OUT/torch32.json ;;
    check) $PY scripts/check_engines.py $PREV/waves.npz ;;
    trt16) $PY scripts/parity.py $PREV/waves.npz $PREV/ref.json --backend tensorrt --dtype float16 --out $OUT/trt16.json ;;
  esac || echo "!!! $step failed"
done
