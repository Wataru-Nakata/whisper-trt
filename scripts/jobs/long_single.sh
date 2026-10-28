# (development record) Long-form one recording at a time (as faster-whisper transcribes); the TensorRT decoder
# engine's refusal of long prompts; the unit tests.
set -uo pipefail
E=$REPO/.cache/eval
for spec in ja:ja en:en ls:en; do
  set=${spec%%:*}; lang=${spec#*:}
  for wt in "" "--word-timestamps"; do
    tag=${wt:+-words}
    echo "=== $set $tag fp16, one recording at a time"
    $PY scripts/longform_parity.py $E/long-$set.npz $E/lref-$set$tag.json --lang $lang $wt --batch 8 --single \
      --out $OUT/ours1-$set$tag.json 2>&1 | grep -E "RTFx|Error|Trace"
  done
done
echo "=== decoder=tensorrt"
$PY - <<PYEOF 2>&1 | grep -vE "hidden args|SymInt|ShapeExpr|USDT|default stream|AOT"
import numpy as np, torch
from whisper_trt import Whisper
d = np.load("$E/long-ls.npz")
m = Whisper.from_pretrained(decoder="tensorrt")
try:
    m.transcribe([torch.from_numpy(d["w0"].astype(np.float32) / 32768)], "en")
    print("no error")
except NotImplementedError as e:
    print("NotImplementedError:", e)
PYEOF
echo "=== tests"
$PY -m pytest -q --tb=short tests 2>&1 | tail -5
