# Multilingual <=30 s clips: faster-whisper fp16 beam 5 batch 8 (and fp32 for exactness) vs ours.
set -uo pipefail
M=/groups/gai51740/acc12576tt/github.com/owsm_trt/.cache
for spec in ${LANGS:-ja:multilingual/jvs_test.json en:testdata/librispeech_test_clean/manifest.json de:multilingual/fleurs_de_de_test.json zh:multilingual/fleurs_cmn_hans_cn_test.json}; do
  lang=${spec%%:*}; man=$M/${spec#*:}
  echo "=== $lang"
  $PY scripts/make_clips.py $man $OUT/clips-$lang.npz --lang $lang --clips ${CLIPS:-64}
  $REF_PY scripts/ref_ct2.py $OUT/clips-$lang.npz $OUT/ref-$lang.json --lang $lang --compute-types ${REF_TYPES:-float16 float32} \
    --configs ${CONFIGS:-5x8} --save-features 0
  for b in ${BACKENDS:-tensorrt/float16}; do
    $PY scripts/parity.py $OUT/clips-$lang.npz $OUT/ref-$lang.json --backend ${b%/*} --dtype ${b#*/} --lang $lang \
      --configs ${CONFIGS:-5x8} --out $OUT/ours-$lang-${b//\//-}.json
  done
done
for lang in ${PROFILE_LANGS:-en ja}; do
  echo "=== profile $lang"
  $PY scripts/profile_decode.py $OUT/clips-$lang.npz --lang $lang --beam 5 --batch 8
  $PY scripts/profile_decode.py $OUT/clips-$lang.npz --lang $lang --beam 1 --batch 1
done
