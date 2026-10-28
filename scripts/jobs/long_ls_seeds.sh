# (development record) Long-form on LibriSpeech chapters with other seeds of the fallback's sampling: how much the
# error rate against the reference text moves with the sampled windows alone.
set -uo pipefail
E=$REPO/.cache/eval
for seed in 1 2 3 4 5; do
  echo "=== seed $seed"
  $PY scripts/longform_parity.py $E/long-ls.npz $E/lref-ls.json --lang en --batch 8 --seed $seed \
    --out $OUT/ours-ls-seed$seed.json 2>&1 | grep -E "RTFx|Error|Trace"
done
