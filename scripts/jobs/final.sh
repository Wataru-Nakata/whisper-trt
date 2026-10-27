set -uo pipefail
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -3
P=/groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2778453
echo "=== smoke ja (forced fallback)"; $PY scripts/smoke_workload.py $P/clips-ja.npz $P/ref-ja.json --lang ja --force-fallback 2>&1 | grep -v "hidden args"
CONFIGS="5x8 5x16" REF_TYPES=float16 PROFILE_LANGS="" bash scripts/jobs/clips.sh
