# (development record) The final code (Triton decoder by default): every test, the streamed decode's time
# breakdown and the step's split on YODAS Japanese.
set -uo pipefail
E=$REPO/.cache/eval
echo "=== pytest"; $PY -m pytest -q tests 2>&1 | tail -2
echo "=== profile"; $PY scripts/profile_stream.py $E/yodas-ja.npz --lang ja --batch 8 32 --decoder triton 2>&1 | grep -E "^batch|Traceback|Error"
echo "=== body"; $PY scripts/profile_body.py $E/yodas-ja.npz --lang ja --decoder triton 2>&1 | grep -E "^batch|Traceback|Error"
D=/groups/gai51740/acc12576tt/github.com/owsm_trt/.cache/testdata/librispeech_dummy
echo "=== cli transcribe"; $PY -m whisper_trt.cli transcribe --language en --json $D/1272-128104-0000.flac $D/1272-128104-0001.flac 2>&1 | grep -v "hidden args"
echo "=== cli detect-language"; $PY -m whisper_trt.cli detect-language $D/1272-128104-0000.flac 2>&1 | grep -v "hidden args"
