# (development record) FP8 calibration audio: YODAS windows from each language's last audio tar (the evaluation
# sets read the first), in the reference env (PyAV).
set -uo pipefail
$REF_PY scripts/make_eval_yodas.py $REPO/.cache/calib --langs ja en de zh hi ta es fr ru ar ko th vi --per-lang 60 --tar -1 --seed 1 2>&1 | grep -vE "^skip"
