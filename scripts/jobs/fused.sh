# (development record) Beam bookkeeping in Triton kernels: tests, the step's split, real windows against the
# PyTorch bookkeeping.
set -uo pipefail
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q -x --tb=short tests/test_search.py 2>&1 | tail -15
echo "=== body"; $PY scripts/profile_body.py $E/yodas-ja.npz --lang ja --decoder triton 2>&1 | grep -E "^batch|Traceback|Error"
for spec in yodas-ja:ja yodas-en:en ls-test-clean:en; do
  echo "=== ${spec%%:*}"; $PY scripts/fused_check.py $E/${spec%%:*}.npz --lang ${spec#*:} 2>&1 | grep -E "^batch|Traceback|Error"
done
