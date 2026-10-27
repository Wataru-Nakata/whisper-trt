# whisper-trt

Fast inference for OpenAI's Whisper (`openai/whisper-large-v3`; other Hugging Face Whisper checkpoints load the
same way) with TensorRT, decoding as faster-whisper / CTranslate2 does: the same features, beam search,
sampling, token limit, suppressed tokens and scores. It does not depend on faster-whisper, CTranslate2 or
transformers.

```python
from whisper_trt import Whisper

model = Whisper.from_pretrained("openai/whisper-large-v3")   # first call builds TensorRT engines (~5 min, cached)

# up-to-30 s windows, each decoded on its own as faster-whisper's generate_with_fallback decodes a window:
# beam 5 at temperature 0, then best-of-5 sampling at 0.2 ... 1.0 for windows that fail its thresholds
out = model.decode_windows(["a.wav", "b.wav"], language="ja", batch_size=8)
out[0].text, out[0].avg_logprob, out[0].no_speech_prob, out[0].compression_ratio, out[0].temperature

# building blocks: features -> encoder -> CTranslate2-style generate / detect_language
cross = model.encode(model.features(["a.wav", "b.wav"]))
res = model.generate(cross, model.prompt("en"), beam_size=5)   # tokens, scores, no-speech probabilities
model.detect_language(cross)                                   # [[("en", 0.99), ...], ...]
```

```bash
whisper-trt build                                  # optional: pre-build the engines
whisper-trt transcribe --language en a.wav b.wav   # --json for every value
whisper-trt detect-language a.wav
```

## Speed

One H200, `openai/whisper-large-v3`, fp16. RTFx: seconds of audio per second of encoder + decoding (ours also
includes the log-Mel features; faster-whisper's are computed apart), best of 3 passes after a warm-up.
"Identical": token sequences equal to faster-whisper's (CTranslate2 4.8.2) with the same options.

LibriSpeech dev-clean, 73 utterances (481 s), English forced, no timestamps:

| setup | faster-whisper fp16 | faster-whisper fp32 | **whisper-trt** | identical to fp32 / fp16 | WER |
|---|---:|---:|---:|---:|---:|
| greedy, batch 1 | 51.5 | 35.0 | **129.8** | 73/73 / 72/73 | 4.17% (all three) |
| beam 5, batch 1 | 45.5 | 24.8 | **92.3** | 72/73 / 72/73 | 3.91% (all three) |
| beam 5, batch 8 | 105.9 | 47.6 | **253.2** | 73/73 / 72/73 | 3.91% (all three) |

The PyTorch backend in fp32 (`backend="torch", dtype="float32"`) gives faster-whisper's float32 tokens exactly:
73/73 for greedy and beam 5, batch 1 and 8, scores within 4e-6, no-speech probabilities within 5e-6.

Up-to-30 s clips (consecutive utterances joined; 64 per language, 21.6-26.0 s on average), the language forced,
no timestamps, beam 5: the first pass of a clip-transcription workload (faster-whisper's batched pipeline, batch 8):

| clips | faster-whisper fp16, batch 8 | **whisper-trt, batch 8** | faster-whisper fp16, batch 16 | **whisper-trt, batch 16** | identical (batch 8) | WER / CER, batch 8: ours vs faster-whisper |
|---|---:|---:|---:|---:|---:|---|
| Japanese (JVS) | 161.5 | **379.4** | 224.4 | **532.1** | 59/64 | CER 5.78% vs 5.80% |
| English (LibriSpeech test-clean) | 208.8 | **478.9** | 276.6 | **627.8** | 63/64 | WER 2.05% vs 1.64% |
| German (FLEURS) | 193.7 | **419.1** | 260.8 | **554.7** | 64/64 | WER 4.59% vs 4.59% |
| Mandarin (FLEURS) | 167.2 | **373.7** | 228.9 | **497.8** | 58/64 | CER 10.44% vs 10.61% |

2.1-2.4x faster-whisper at the same batch size, and 2.9-3.3x its batch 8 with batch 16. The English WER gap is
one clip: whisper-trt's beam search ends before the clip's last sentence (47 instead of 65 tokens); the PyTorch
fp16 backend with the same search gives faster-whisper's tokens on it (see Precision). Language identification:
64/64 top-1 equal to CTranslate2's on the Japanese and English clips. With every window forced through the
temperature fallback (`scripts/smoke_workload.py --force-fallback`), 64 Japanese windows decode six times each
(beam 5, then best-of-5 sampling at 0.2 ... 1.0) at RTFx 52.

Where the time goes (beam 5, batch 8, English clips): encoder 45 ms per batch of 8 windows (12%), decoding
the rest, at 3.5 ms per decoder step for 40 rows (TensorRT's layer profiler: GEMMs 40%, cross-attention 19%,
fused LayerNorm/residual kernels 14%, GELU 9%, self-attention 8%). At batch 1 (greedy) a step takes 1.95 ms. Steps are
bound by kernel latency (about 420 TensorRT kernels each), not by memory bandwidth.

## How it works

* **No faster-whisper, no CTranslate2.** Whisper is reimplemented in plain PyTorch (`whisper_trt/modeling.py`);
  the Hugging Face checkpoints load with `strict=True`. Log-Mel features are computed on the GPU as
  faster-whisper's numpy `FeatureExtractor` computes them (each window normalized on its own, zero-padded to
  3000 frames after normalization; within 2e-5 of faster-whisper's).
* **CTranslate2's search rules** (`whisper_trt/search.py`, checked against a line-by-line transcription of
  CTranslate2 4.8.2's C++ loops in `tests/test_search.py`): beam search over the best 2K of all K x V
  continuations, an ended candidate among the first K giving its place to the next non-ending one, a window
  done at K finished hypotheses, the best by score / length; greedy and best-of-N sampling (scores of the
  untempered distribution); suppressed and begin-suppressed tokens set to the lowest float; at most
  `min(448 / 2, 448 - prompt + 1)` = 224 generated tokens (the limit faster-whisper users meet); the
  no-speech probability from the logits at `<|startoftranscript|>`; language identification from the
  language tokens' logits after `<|startoftranscript|>` (64/64 top-1 equal to CTranslate2's on Japanese and
  English clips, probabilities within 2e-3).
* **Static buffers and CUDA graphs.** Every decoding step (decoder call, token choice) runs on fixed buffers and
  is one CUDA-graph replay; the host checks for the end every 4 steps.
* **Beam search never moves the KV cache.** Each row has a table of the row whose cache holds each of its
  slots; a new beam takes its parent's table (a few KB) instead of copying the parent's cache (hundreds of MB
  per step at 8 windows x 5 beams). A window's beams share its cross-attention keys/values: their queries are
  attended together, as K x S queries of one window.
* **Sampling inside the graphs.** The temperature fallback samples with the Gumbel-max trick on random numbers
  from a counter-based hash of (seed, step, row, token): no generator state, so it runs inside the captured
  graphs, and the same seed gives the same samples (`decode_windows(..., seed=)`). Its numbers are not
  CTranslate2's (curand), so sampled decodes agree with faster-whisper in distribution only.
* **TensorRT engines are built on first use**: the encoder (with every decoder layer's cross-attention
  projection) and the decoder call are exported to ONNX and compiled into strongly-typed fp16 engines, cached
  under `~/.cache/whisper_trt` (`WHISPER_TRT_CACHE` moves them). Decoder self- and cross-attention are Triton
  kernels compiled into the engine as TensorRT plugins (after owsm-trt's): self-attention reads the KV cache
  through the slot table, so the cache needs no length buckets; cross-attention is split over 8 slices of the
  1500 keys ("flash-decoding"), which gives one window 160 programs instead of 20 (batch-1 step 2.4 -> 2.0 ms).
* **Precision.** LayerNorm and the MLPs' GELU are computed in fp32 inside the fp16 engines: TensorRT's fp16 GELU
  multiplied the encoder's error by 5 (`scripts/encoder_variants.py`). TensorRT's fp16 GEMMs are still 3-9x
  less precise than PyTorch's (`scripts/op_precision.py`; TensorRT 11 has no setting for their accumulation),
  which is where the remaining differences from faster-whisper come from: on two English clips where whisper-trt
  differs from faster-whisper, the PyTorch fp16 backend with the same search gives faster-whisper's tokens.

## Limitations

* Decoding with timestamps (and so `transcribe`-style long-form decoding and word timestamps), prompts with
  previous text and `initial_prompt` are not implemented yet; windows are decoded on their own, as
  faster-whisper's batched pipeline decodes clips.
* Every input is one window of at most 30 s: longer audio is trimmed, not chunked (features are normalized
  over the whole input, as faster-whisper normalizes a clip).
* A batch of windows decodes until its slowest window is done (no refilling of finished rows yet).
* Tested with torch 2.14.1 (CUDA 13) + TensorRT 11.4.0.106 on H200 only.

## Development

GPU work runs on compute nodes: `scripts/qsub_pool.sh "CMD=..."` submits `scripts/gpu_job.pbs`, which unpacks
the prebuilt environments onto node-local disk (Lustre is slow with torch's many small files) and runs a
snapshot of the code. `scripts/build_env_tar.sh` builds them (from `uv.lock`):
`.cache/envs/whisper-env.tar.zst` (this package's dependencies) and, with `ref`, `.cache/envs/fw-ref.tar.zst`
(faster-whisper 1.2.1 / CTranslate2 4.8.2 with CUDA 12 cuBLAS/cuDNN, from `tools/fw_ref`; reference only).

* `scripts/ref_ct2.py`: reference decodes with faster-whisper / CTranslate2 (in the reference env).
* `scripts/parity.py`: our decodes against them (identical tokens, score and no-speech differences, WER/CER,
  RTFx); `--build KEY=VALUE` builds engine variants (`build.BuildOptions`).
* `scripts/make_clips.py`: up-to-30 s clips from consecutive utterances (multilingual benchmarks).
* `scripts/smoke_workload.py`: language identification and `decode_windows` (with `--force-fallback`, every
  window through every temperature) on clips.
* `scripts/profile_decode.py`, `scripts/trt_profile.py`: time per stage / per TensorRT layer.
* `scripts/check_engines.py`, `scripts/encoder_variants.py`, `scripts/op_precision.py`,
  `scripts/teacher_force.py`, `scripts/search_debug.py`: component-level checks of the engines.
* `pytest tests`: search against CTranslate2's loops, sampling, attention kernels (GPU tests skip on CPU).
