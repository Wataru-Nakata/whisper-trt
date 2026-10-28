# (development record) Long-form: the long recordings (reference env), unit tests, a first transcription.
set -uo pipefail
E=$REPO/.cache/eval
(cd $REPO/scripts && $REF_PY make_eval_long.py $E --langs ja en --per-lang 20 --seconds 300) 2>&1 | grep -vE "^skip"
echo "=== pytest"; $PY -m pytest -q -x --tb=short tests/test_alignment.py tests/test_search.py tests/test_timestamps.py 2>&1 | tail -4
for wt in "" "--word-timestamps"; do
  echo "=== smoke ja $wt"; $PY scripts/longform_smoke.py $E/long-ja.npz --lang ja --n 3 $wt 2>&1 | grep -vE "hidden args|SymIntExpr|ShapeExpr|both arguments|USDT" | tail -40
done
