# (development record) Streamed decode time breakdown on YODAS ja / en and LibriSpeech test-clean.
set -uo pipefail
E=$REPO/.cache/eval
for spec in yodas-ja:ja yodas-en:en ls-test-clean:en; do
  echo "=== ${spec%%:*}"; $PY scripts/profile_stream.py $E/${spec%%:*}.npz --lang ${spec#*:} --batch 8 16 32 2>&1 | grep -E "^batch|Traceback|Error"
done
