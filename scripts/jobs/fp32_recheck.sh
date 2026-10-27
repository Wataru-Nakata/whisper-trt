# (development record) fp32 exactness on the current code for the YODAS languages checked before.
set -uo pipefail
E=$REPO/.cache/eval
for lang in ${LANGS:-ja en de hi ta el}; do
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --backend torch --dtype float32 \
    --configs 5x8 --repeat 0 --stream 2>&1 | grep -E "vs float32|Traceback|Error" | sed "s/^/$lang /"
done
