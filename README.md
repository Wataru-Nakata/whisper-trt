# whisper-trt

Fast inference for OpenAI's Whisper (`openai/whisper-large-v3` and other Hugging Face Whisper checkpoints) with
TensorRT, decoding exactly as faster-whisper / CTranslate2 does. It does not depend on faster-whisper,
CTranslate2 or transformers.

```python
from whisper_trt import Whisper

model = Whisper.from_pretrained("openai/whisper-large-v3")   # first call builds TensorRT engines (cached)

# up-to-30 s windows, each decoded on its own as faster-whisper's generate_with_fallback decodes a window:
# beam 5 at temperature 0, then best-of-5 sampling at 0.2 ... 1.0 for windows that fail its thresholds
out = model.decode_windows(["a.wav", "b.wav"], language="ja", batch_size=8)
out[0].text, out[0].avg_logprob, out[0].no_speech_prob, out[0].compression_ratio, out[0].temperature

# building blocks: features -> encoder -> CTranslate2-style generate / detect_language
cross = model.encode(model.features(["a.wav", "b.wav"]))
model.generate(cross, model.prompt("en"), beam_size=5)      # tokens, scores, no-speech probabilities
model.detect_language(cross)                                 # [[("en", 0.99), ...], ...]
```

```bash
whisper-trt build                                  # optional: pre-build the engines
whisper-trt transcribe --language en a.wav b.wav
whisper-trt detect-language a.wav
```

## Speed

(measurements in progress)

## How it works

* **No faster-whisper, no CTranslate2.** Whisper is reimplemented in plain PyTorch (`whisper_trt/modeling.py`);
  the Hugging Face checkpoints load with `strict=True`. Log-Mel features are computed on the GPU exactly as
  faster-whisper's numpy `FeatureExtractor` computes them (each window normalized on its own, zero-padded to
  3000 frames after normalization).
* **CTranslate2's search rules** (`whisper_trt/search.py`, checked against a line-by-line transcription of
  CTranslate2 4.8.2's C++ loops in `tests/test_search.py`): beam search over the best 2K of all K x V
  continuations, an ended candidate among the first K giving its place to the next non-ending one, a window
  done at K finished hypotheses, the best by score / length; greedy and best-of-N sampling (scores of the
  untempered distribution); suppressed and begin-suppressed tokens set to the lowest float; at most
  `min(448 / 2, 448 - prompt + 1)` = 224 generated tokens (the limit faster-whisper users meet); the
  no-speech probability from the logits at `<|startoftranscript|>`.
* **Static buffers and CUDA graphs.** Every decoding step (decoder call, token choice) runs on fixed buffers
  and is one CUDA-graph replay; the host checks for the end every 4 steps. The KV cache comes in length
  buckets (64 / 128 / 228 slots for a 4-token prompt), so a step reads only as much cache as it needs.
* **Beam search never moves the KV cache.** Each row has a table of the row whose cache holds each of its
  slots; a new beam takes its parent's table (a few KB) instead of copying the parent's cache (hundreds of MB
  per step at 8 windows x 5 beams). A window's beams share its cross-attention keys/values: their queries
  are attended together, as K x S queries of one window.
* **TensorRT engines are built on first use**: the encoder (with every decoder layer's cross-attention
  projection) and the decoder call are exported to ONNX and compiled into strongly-typed fp16 engines, cached
  under `~/.cache/whisper_trt` (`WHISPER_TRT_CACHE` moves them). Decoder self- and cross-attention are Triton
  kernels compiled into the engine as TensorRT plugins (ported from owsm-trt), which read the KV cache through
  the slot table and the cross-attention keys/values where they are.
* **Precision.** LayerNorm and the MLPs' GELU are computed in fp32 inside the fp16 engines: TensorRT's fp16
  GELU multiplied the encoder's error by 5 (`scripts/encoder_variants.py`).

## Development

GPU work runs on compute nodes through `scripts/qsub_pool.sh` (`scripts/gpu_job.pbs`).

* `scripts/ref_ct2.py`: reference decodes with faster-whisper / CTranslate2 (in a faster-whisper venv).
* `scripts/parity.py`: our decodes against them (identical tokens, score and no-speech differences, WER, RTFx).
* `scripts/make_clips.py`: up-to-30 s clips from consecutive utterances, for multilingual benchmarks.
* `scripts/check_engines.py`, `scripts/encoder_variants.py`, `scripts/op_precision.py`,
  `scripts/teacher_force.py`, `scripts/search_debug.py`: component-level checks of the engines.
