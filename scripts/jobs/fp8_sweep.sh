# (development record) FP8 encoder with some layers kept fp16 (fp8_skip): encoder time, agreement on YODAS ja/en
# (vs faster-whisper fp16 and fp32) and LibriSpeech test-clean (WER).
set -uo pipefail
E=$REPO/.cache/eval
S=$REPO/src/whisper_trt/fp8_scales/whisper-large-v3.json
for skip in '^decoder\.' '^decoder\.|self_attn\.[qk]_proj' '^decoder\.|self_attn' '^decoder\.|self_attn|fc2' '^decoder\.|self_attn|fc1'; do
  echo "=== skip $skip"
  $PY scripts/bench_encoder.py --batches 1 8 --build fp8=$S "fp8_skip=$skip" 2>&1 | grep -E "^batch|Error|Trace"
  for spec in yodas-ja:ja:ref32-yodas-ja yodas-en:en:ref32-yodas-en ls-test-clean:en:ref-ls-test-clean; do
    IFS=: read name lang ref <<< "$spec"
    $PY scripts/parity.py $E/$name.npz $E/$ref.json --lang $lang --configs 5x8 --repeat 0 --stream \
      --build fp8=$S "fp8_skip=$skip" 2>&1 | grep -E "^beam|vs float|Traceback|Error" | sed "s/^/$name /"
  done
done
