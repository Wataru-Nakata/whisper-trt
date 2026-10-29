# (development record) The streaming search: unit tests, then long-form smoke runs (words; detected language).
set -uo pipefail
E=$REPO/.cache/eval
$PY -m pytest -q -x --tb=short tests 2>&1 | tail -30
echo "=== smoke ja words"
$PY scripts/longform_smoke.py $E/long-ja.npz --lang ja --n 3 --word-timestamps 2>&1 | grep -vE "hidden args|SymIntExpr|ShapeExpr|both arguments|USDT|default stream|AOT Plugin" | head -40
echo "=== smoke en detected"
$PY scripts/longform_smoke.py $E/long-en.npz --n 3 2>&1 | grep -vE "hidden args|SymIntExpr|ShapeExpr|both arguments|USDT|default stream|AOT Plugin" | head -30
