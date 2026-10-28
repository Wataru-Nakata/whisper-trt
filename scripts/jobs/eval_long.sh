# (development record) Long-form transcription against faster-whisper's transcribe: references (fp32 and fp16,
# with and without word timestamps), ours (fp16 at 8 and 32 slots; the fp32 PyTorch backend).
set -uo pipefail
E=$REPO/.cache/eval
for lang in ja en; do
  for wt in "" "--word-timestamps"; do
    tag=${wt:+-words}
    [ -f $E/lref-$lang$tag.json ] || $REF_PY scripts/ref_long.py $E/long-$lang.npz $E/lref-$lang$tag.json --lang $lang \
      --compute-types float32 float16 $wt --model $REPO/.cache/fw-large-v3 2>&1 | grep -E "RTFx|Error|Trace"
    echo "=== $lang $tag fp16"
    $PY scripts/longform_parity.py $E/long-$lang.npz $E/lref-$lang$tag.json --lang $lang $wt --batch 8 32 \
      --out $OUT/ours-$lang$tag.json 2>&1 | grep -E "RTFx|vs |Error|Trace"
    echo "=== $lang $tag fp32"
    $PY scripts/longform_parity.py $E/long-$lang.npz $E/lref-$lang$tag.json --lang $lang $wt --batch 8 \
      --backend torch --dtype float32 2>&1 | grep -E "RTFx|vs |Error|Trace"
  done
done
