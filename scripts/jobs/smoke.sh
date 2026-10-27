set -uo pipefail
echo "=== profile"; $PY scripts/trt_profile.py /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2778453/clips-en.npz 2>&1 | grep -v "hidden args"
echo "=== smoke ja"; $PY scripts/smoke_workload.py /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2778453/clips-ja.npz /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2778453/ref-ja.json --lang ja
echo "=== smoke en"; $PY scripts/smoke_workload.py /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2778453/clips-en.npz /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2778453/ref-en.json --lang en --torch16 30 52
