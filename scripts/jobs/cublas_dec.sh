# (development record) Decoder step with cuBLAS GEMMs + Triton LN/attention against TensorRT: time, logits error,
# per-kernel profile.
set -uo pipefail
$PY scripts/bench_decoder.py --cases 1x1 40x8 80x16 160x32 2>&1 | grep -E "^rows|Traceback|Error"
$PY scripts/profile_triton_decoder.py --gemm cublas --cases 40x8 160x32 2>&1 | grep -vE "hidden args|SymIntExpr|ShapeExpr|both arguments|USDT"
