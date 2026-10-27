set -uo pipefail
PREV=${PREV:?set PREV to the output dir of a parity job}
for opt in "--fresh-prefill 1" "--prefill-twice 1"; do
  echo "=== reuse=1 $opt"
  $PY scripts/search_debug.py $PREV/waves.npz --items 0 1 2 4 5 7 9 --graphs 0 --reuse 1 $opt 2>&1 | grep -v "hidden args"
done
