# (development record) The encoder engine at builder optimization levels 3 and 5: time per window.
set -uo pipefail
for lvl in 3 5; do
  echo "=== opt_level $lvl"; $PY scripts/bench_encoder.py --batches 1 8 --build opt_level=$lvl 2>&1 | grep -E "^batch|Error|Trace"
done
