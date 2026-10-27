# (development record) attn_plugin tests with failure details.
set -uo pipefail
$PY -m pytest -q -x --tb=short tests/test_attn_plugin.py 2>&1 | tail -30
