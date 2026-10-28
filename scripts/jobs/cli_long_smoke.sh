# (development record) The CLI's long-form transcription on two LibriSpeech chapters (their first 70 s).
set -uo pipefail
$PY - <<PYEOF
import json, numpy as np, soundfile as sf
d = np.load("$REPO/.cache/eval/long-ls.npz")
for i in range(2):
    sf.write(f"$OUT/ch{i}.wav", d[f"w{i}"][: 70 * 16000], 16000)
PYEOF
$PY -m whisper_trt.cli transcribe --long --language en $OUT/ch0.wav $OUT/ch1.wav 2>&1 | grep -vE "hidden args|SymInt|ShapeExpr|USDT|default stream|AOT"
$PY -m whisper_trt.cli transcribe --long --word-timestamps --language en $OUT/ch0.wav 2>&1 | grep -P "\t" | head -12
$PY -m whisper_trt.cli transcribe --long --word-timestamps --json --language en $OUT/ch1.wav 2>/dev/null | head -c 600; echo
$PY -m whisper_trt.cli transcribe --word-timestamps --language en $OUT/ch0.wav 2>&1 | tail -1
