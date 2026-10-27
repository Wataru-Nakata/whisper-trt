# (development record) Decoder step times, each backend twice in turn (order effects).
set -uo pipefail
$PY scripts/bench_decoder.py --cases 5x1 40x8 80x16 160x32 2>&1 | grep -E "^rows|Traceback|Error"
