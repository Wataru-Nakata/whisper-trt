# (development record) Search with prefilled prompts of any length: unit tests.
set -uo pipefail
$PY -m pytest -q -x --tb=short tests/test_search.py tests/test_timestamps.py 2>&1 | tail -25
