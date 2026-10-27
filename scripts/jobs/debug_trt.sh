# Locate the TensorRT decoding error (PREV: a parity job's outputs).
set -uo pipefail
PREV=${PREV:?set PREV to the output dir of a parity job}
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -15
echo "=== teacher"; $PY scripts/teacher_force.py $PREV/waves.npz $PREV/ref.json || echo "!!! teacher failed"
echo "=== nographs"; WHISPER_TRT_GRAPHS=0 $PY scripts/parity.py $PREV/waves.npz $PREV/ref.json --backend tensorrt --configs 1x1 --repeat 0 || echo "!!! nographs failed"
echo "=== variants"; $PY scripts/encoder_variants.py $PREV/waves.npz --layers 8 || echo "!!! variants failed"
