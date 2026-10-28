# (development record) Decoding with timestamps: unit tests against CTranslate2's loops with its timestamp rules.
set -uo pipefail
$PY -m pytest -q -x --tb=short tests/test_timestamps.py tests/test_search.py 2>&1 | tail -25
