# Our side only, against a previous job's waveforms/reference (PREV=outputs/job-N): torch fp32, then TensorRT fp16
# (engines built on first use).
set -uo pipefail
PREV=${PREV:?set PREV to the output dir of a parity job}
for spec in ${SPECS:-torch/float32 tensorrt/float16}; do
  echo "=== $spec"
  $PY scripts/parity.py $PREV/waves.npz $PREV/ref.json --backend ${spec%/*} --dtype ${spec#*/} \
    --out $OUT/${spec//\//-}.json || echo "!!! $spec failed"
done
