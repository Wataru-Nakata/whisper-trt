set -uo pipefail
PREV=${PREV:?set PREV to the output dir of a parity job}
$PY scripts/teacher_force.py $PREV/waves.npz $PREV/ref.json --items ${ITEMS:-9 11 15 17}
