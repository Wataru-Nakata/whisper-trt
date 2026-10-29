# (development record) The CLI's long-form without --language (detected), its errors; the PyTorch rules of the
# mixed search (WHISPER_TRT_FUSED_BEAM=0) in the step graph on the real model.
set -uo pipefail
E=$REPO/.cache/eval
F="hidden args|SymInt|ShapeExpr|USDT|default stream|AOT|both arguments"
$PY - <<PYEOF
import numpy as np, soundfile as sf
for name, i in (("ls", 0), ("ja", 1)):
    d = np.load(f"$E/long-{name}.npz")
    sf.write(f"$OUT/{name}.wav", d[f"w{i}"][: 70 * 16000], 16000)
PYEOF
$PY -m whisper_trt.cli transcribe --long $OUT/ls.wav $OUT/ja.wav 2>&1 | grep -vE "$F" | grep -E "language|\[" | head -8
$PY -m whisper_trt.cli transcribe --long --json $OUT/ja.wav 2>/dev/null | head -c 300; echo
$PY -m whisper_trt.cli transcribe $OUT/ls.wav 2>&1 | tail -1
echo "=== PyTorch rules (WHISPER_TRT_FUSED_BEAM=0), YODAS ja"
WHISPER_TRT_FUSED_BEAM=0 $PY scripts/longform_parity.py $E/long-ja.npz $E/lref-ja.json --lang ja --batch 8 \
  --out $OUT/ours-ja-torchrules.json 2>&1 | grep -E "RTFx|vs float32|Error|Trace" | sed -E 's/its fallback segments [0-9]+; //'
