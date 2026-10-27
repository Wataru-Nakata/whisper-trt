# (development record) TensorRT layer profile of a decoder step at 40/80/160 rows.
set -uo pipefail
$PY scripts/trt_profile.py $REPO/.cache/eval/yodas-ja.npz --rows ${ROWS:-40 80 160} --windows ${WINDOWS:-8 16 32} --cache 228 --offset 80 2>&1 | grep -v "hidden args\|SymIntExpr\|ShapeExpr\|both arguments"
