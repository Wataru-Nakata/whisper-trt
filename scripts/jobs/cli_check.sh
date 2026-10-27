# (development record) Unit tests, the CLI, and LibriSpeech parity after the stream fence.
set -uo pipefail
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -2
echo "=== cli transcribe"; $PY -m whisper_trt.cli transcribe --language en --json /groups/gai51740/acc12576tt/github.com/owsm_trt/.cache/testdata/librispeech_dummy/1272-128104-0000.flac /groups/gai51740/acc12576tt/github.com/owsm_trt/.cache/testdata/librispeech_dummy/1272-128104-0001.flac 2>&1 | grep -v "hidden args"
echo "=== cli detect-language"; $PY -m whisper_trt.cli detect-language /groups/gai51740/acc12576tt/github.com/owsm_trt/.cache/testdata/librispeech_dummy/1272-128104-0000.flac 2>&1 | grep -v "hidden args"
echo "=== parity"; $PY scripts/parity.py /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2777982/waves.npz /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2777982/ref.json --configs 1x1 5x1 5x8 --repeat 1 2>&1 | grep -E "^beam|vs float32"
