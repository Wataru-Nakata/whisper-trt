# (development record) Long-form fp32 (PyTorch backend) outputs for every set, with and without word timestamps.
set -uo pipefail
E=$REPO/.cache/eval
for spec in ja:ja en:en ls:en; do
  set=${spec%%:*}; lang=${spec#*:}
  for wt in "" "--word-timestamps"; do
    tag=${wt:+-words}
    for _ in $(seq 120); do [ -f $E/lref-$set$tag.json ] && break; sleep 30; done  # (a reference still being made)
    echo "=== $set $tag fp32"
    $PY scripts/longform_parity.py $E/long-$set.npz $E/lref-$set$tag.json --lang $lang $wt --batch 8 --backend torch \
      --dtype float32 --out $OUT/ours32-$set$tag.json 2>&1 | grep -E "RTFx|vs |Error|Trace"
  done
done
