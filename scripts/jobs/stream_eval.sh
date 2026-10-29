# (development record) Long-form on the streaming search (Search.serve): fp16 at 8 and 32 slots against
# faster-whisper's transcribe, with and without word timestamps; a second run at 8 slots (the same output?).
set -uo pipefail
E=$REPO/.cache/eval
for spec in ja:ja en:en ls:en; do
  set=${spec%%:*}; lang=${spec#*:}
  for wt in "" "--word-timestamps"; do
    tag=${wt:+-words}
    echo "=== $set $tag fp16"
    $PY scripts/longform_parity.py $E/long-$set.npz $E/lref-$set$tag.json --lang $lang $wt --batch 8 32 \
      --out $OUT/ours-$set$tag.json 2>&1 | grep -E "RTFx|vs |Error|Trace"
    $PY scripts/longform_parity.py $E/long-$set.npz $E/lref-$set$tag.json --lang $lang $wt --batch 8 \
      --out $OUT/ours-$set$tag-again.json 2>&1 | grep -E "RTFx|Error|Trace"
    $PY - <<PYEOF
import json
a = json.load(open("$OUT/ours-$set$tag.json"))["b8"]
b = json.load(open("$OUT/ours-$set$tag-again.json"))["b8"]
print("    second run, the same segments:", sum(x == y for x, y in zip(a, b)), "/", len(a))
PYEOF
  done
done
