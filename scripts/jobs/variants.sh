# Engine variants: transcript agreement (LibriSpeech 73 vs faster-whisper) and decoder step time.
set -uo pipefail
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -5
for v in "" "decoder_opt_level=5" "gelu32_decoder=false" "ln32=false"; do
  echo "=== variant [$v]"
  $PY scripts/parity.py /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2777982/waves.npz /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2777982/ref.json --repeat 1 --build $v 2>&1 | grep -E "^beam|vs float32/beam1/b1|vs float32/beam5/b8|loaded"
  $PY scripts/trt_profile.py /groups/gai51740/acc12576tt/github.com/whisper_trt/outputs/job-2778453/clips-en.npz --build $v 2>&1 | grep -E "^rows|^   "
done
