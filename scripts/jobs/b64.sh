# (development record) 64 slots (engines for up to 64 windows x 5 beams; the encoder up to 32) against 32,
# streamed, with the results of 32 and 64 slots compared.
set -uo pipefail
E=$REPO/.cache/eval
for spec in ls-test-clean:en yodas-ja:ja yodas-en:en; do
  echo "=== ${spec%%:*}"
  $PY scripts/parity.py $E/${spec%%:*}.npz $E/ref-${spec%%:*}.json --lang ${spec#*:} --configs 5x32 5x48 5x64 --repeat 1 --stream \
    --build max_batch=64 max_rows=320 2>&1 | grep -E "^beam|tokens|Traceback|Error"
done
