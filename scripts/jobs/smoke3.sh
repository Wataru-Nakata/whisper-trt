# (development record) decode_windows with every window through the fallback, and the CLI, on the final code.
set -uo pipefail
P=/groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2778453
D=/groups/gai51740/acc12576tt/github.com/owsm_trt/.cache/testdata/librispeech_dummy
$PY scripts/smoke_workload.py $P/clips-ja.npz $P/ref-ja.json --lang ja --force-fallback 2>&1 | grep -v "hidden args"
$PY scripts/smoke_workload.py $P/clips-en.npz $P/ref-en.json --lang en 2>&1 | grep -v "hidden args"
$PY -m whisper_trt.cli transcribe --language en --json $D/1272-128104-0000.flac 2>&1 | grep -v "hidden args"
