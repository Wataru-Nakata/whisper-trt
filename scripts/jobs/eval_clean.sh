# Large clean-speech evaluation (references available): faster-whisper fp16 vs ours, utterance by utterance.
set -uo pipefail
E=$REPO/.cache/eval
M=/groups/gai51740/acc12576tt/github.com/owsm_trt/.cache
SETS=${SETS:-"ls-test-clean:en:testdata/librispeech_test_clean/manifest.json ls-dev-other:en:multilingual/librispeech_dev_other.json jvs-test:ja:multilingual/jvs_test.json fleurs-de:de:multilingual/fleurs_de_de_test.json fleurs-zh:zh:multilingual/fleurs_cmn_hans_cn_test.json fleurs-es:es:multilingual/fleurs_es_419_test.json fleurs-en:en:multilingual/fleurs_en_us_validation.json fleurs-fr:fr:multilingual/fleurs_fr_fr_validation.json fleurs-ja:ja:multilingual/fleurs_ja_jp_validation.json fleurs-ru:ru:multilingual/fleurs_ru_ru_validation.json"}
for spec in $SETS; do
  name=${spec%%:*}; rest=${spec#*:}; lang=${rest%%:*}; man=$M/${rest#*:}
  echo "=== $name ($lang)"
  [ -f $E/$name.npz ] || $PY scripts/prepare_waves.py $man $E/$name.npz --int16
  [ -f $E/ref-$name.json ] || $REF_PY scripts/ref_ct2.py $E/$name.npz $E/ref-$name.json --lang $lang \
    --compute-types float16 --configs 5x8 --save-features 0 --repeat 0 | tail -1
  $PY scripts/parity.py $E/$name.npz $E/ref-$name.json --lang $lang --configs 5x8 --repeat 0 \
    --out $OUT/ours-$name.json --report $OUT/report-$name.json 2>&1 | grep -v "hidden args\|^loaded\|^features"
done
