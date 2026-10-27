# (development record) PyTorch TunableOp (GEMM algorithms timed per shape) on the decoder's GEMMs: the GEMM
# chain and the Triton decoder step, without and with it.
set -uo pipefail
echo "=== default"
$PY scripts/bench_gemm_chain.py --rows 40 160 2>&1 | grep -E "cuBLAS|Error|Trace"
echo "=== tunableop"
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=$OUT/tunableop.csv \
  $PY scripts/bench_gemm_chain.py --rows 40 160 2>&1 | grep -E "cuBLAS|Error|Trace"
echo "=== step default"
$PY scripts/bench_decoder.py --cases 40x8 160x32 2>&1 | grep -E "^rows|Error|Trace" | sed 's/, triton GEMMs.*//'
echo "=== step tunableop"
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=$OUT/tunableop-step.csv \
  $PY scripts/bench_decoder.py --cases 40x8 160x32 2>&1 | grep -E "^rows|Error|Trace" | sed 's/, triton GEMMs.*//'
ls $OUT/*.csv 2>/dev/null && head -5 $OUT/tunableop*.csv
