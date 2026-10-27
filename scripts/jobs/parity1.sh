# Waveforms -> faster-whisper reference (fp32/fp16) -> our torch fp32 parity.
set -euo pipefail
D=/groups/gai51740/acc12576tt/github.com/owsm_trt/.cache/testdata/librispeech_dummy/manifest.json
$PY scripts/prepare_waves.py $D $OUT/waves.npz
$REF_PY scripts/ref_ct2.py $OUT/waves.npz $OUT/ref.json --lang en --compute-types float32 float16 --configs 1x1 5x1 5x8
$PY scripts/parity.py $OUT/waves.npz $OUT/ref.json --backend torch --dtype float32 --out $OUT/torch32.json
