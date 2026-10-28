# (development record) FP8 encoder: calibration (scale file into the repository), speed, first agreement.
set -uo pipefail
E=$REPO/.cache/eval
S=$REPO/src/whisper_trt/fp8_scales/whisper-large-v3.json
echo "=== calibrate"; $PY scripts/fp8_calibrate.py $REPO/.cache/calib --out $S 2>&1 | grep -E "windows|layers|Error|Trace"
echo "=== speed"; $PY scripts/bench_encoder.py --batches 1 8 --build fp8=$S 2>&1 | grep -E "^batch|Error|Trace|built"
for lang in ja en ta; do
  echo "=== yodas $lang"
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --configs 5x8 5x32 --repeat 0 --stream \
    --build fp8=$S 2>&1 | grep -E "^beam|vs float|Traceback|Error"
done
