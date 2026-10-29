# (development record) The streaming search: one recording at a time; LibriSpeech chapters over sampling seeds;
# window-level fp32 exactness (the fused kernels changed).
set -uo pipefail
E=$REPO/.cache/eval
for spec in ja:ja en:en ls:en; do
  set=${spec%%:*}; lang=${spec#*:}
  for wt in "" "--word-timestamps"; do
    tag=${wt:+-words}
    echo "=== $set $tag fp16, one recording at a time"
    $PY scripts/longform_parity.py $E/long-$set.npz $E/lref-$set$tag.json --lang $lang $wt --batch 8 --single \
      --out $OUT/ours1-$set$tag.json 2>&1 | grep -E "RTFx|vs |Error|Trace"
  done
done
for seed in 1 2 3 4 5; do
  echo "=== ls seed $seed"
  $PY scripts/longform_parity.py $E/long-ls.npz $E/lref-ls.json --lang en --batch 8 --seed $seed \
    --out $OUT/ours-ls-seed$seed.json 2>&1 | grep -E "RTFx|Error|Trace"
done
for lang in ja ta; do
  $PY scripts/parity.py $E/yodas-$lang.npz $E/ref32-yodas-$lang.json --lang $lang --backend torch --dtype float32 \
    --configs 5x8 --repeat 0 --stream 2>&1 | grep -E "vs float32|Traceback|Error" | sed "s/^/$lang /"
done
