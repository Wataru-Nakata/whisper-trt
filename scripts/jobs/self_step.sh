# (development record) The single-token self-attention kernel: tests, block sizes by rows, the Triton decoder's step.
set -uo pipefail
echo "=== pytest"; $PY -m pytest -q tests/test_attn_plugin.py tests/test_search.py 2>&1 | tail -3
echo "=== self-attention"; $PY scripts/bench_self_attn.py --rows 5 20 40 80 120 160 --offsets 20 80 160 2>&1 | grep -vE "hidden args|SymIntExpr|ShapeExpr|both arguments|USDT"
echo "=== step"; $PY scripts/bench_decoder.py --cases 5x1 40x8 80x16 160x32 2>&1 | grep -E "^rows|Traceback|Error"
