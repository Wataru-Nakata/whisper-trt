# (development record) Throughput at 8/16/32 slots on the large sets (references from the eval jobs).
set -uo pipefail
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -2
for spec in ls-test-clean:en jvs-test:ja yodas-ja:ja yodas-en:en; do
  name=${spec%%:*}; lang=${spec#*:}
  echo "=== $name"
  $PY scripts/parity.py $E/$name.npz $E/ref-$name.json --lang $lang --configs 5x8 5x16 5x32 --repeat 1 --stream 2>&1 \
    | grep -E "^beam|tokens|silence|fallback|limit|Traceback|Error"
done
