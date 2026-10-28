# (development record) Long-form on clean speech (LibriSpeech test-clean chapters, scripts/make_long_ls.py) against
# faster-whisper's transcribe: references (fp32, fp16, with and without word timestamps), ours (fp16, fp32).
set -uo pipefail
E=$REPO/.cache/eval
for spec in ls:en; do
  lang=${spec#*:}; set=${spec%%:*}
  for wt in "" "--word-timestamps"; do
    tag=${wt:+-words}
    [ -f $E/lref-$set$tag.json ] || $REF_PY scripts/ref_long.py $E/long-$set.npz $E/lref-$set$tag.json --lang $lang \
      --compute-types float32 float16 $wt --model $REPO/.cache/fw-large-v3 2>&1 | grep -E "RTFx|Error|Trace"
    echo "=== $set $tag fp16"
    $PY scripts/longform_parity.py $E/long-$set.npz $E/lref-$set$tag.json --lang $lang $wt --batch 8 32 \
      --out $OUT/ours-$set$tag.json 2>&1 | grep -E "RTFx|vs |Error|Trace"
    echo "=== $set $tag fp32"
    $PY scripts/longform_parity.py $E/long-$set.npz $E/lref-$set$tag.json --lang $lang $wt --batch 8 \
      --backend torch --dtype float32 2>&1 | grep -E "RTFx|vs |Error|Trace"
  done
done
