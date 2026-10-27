# whisper-trt

Fast inference for OpenAI's Whisper (`openai/whisper-large-v3`; other Hugging Face Whisper checkpoints load the
same way) with TensorRT, decoding as faster-whisper / CTranslate2 does: the same features, beam search,
sampling, token limit, suppressed tokens and scores. It does not depend on faster-whisper, CTranslate2 or
transformers.

```python
from whisper_trt import Whisper

model = Whisper.from_pretrained("openai/whisper-large-v3")   # first call builds TensorRT engines (~6 min, cached)

# up-to-30 s windows, each decoded on its own as faster-whisper's generate_with_fallback decodes a window:
# beam 5 at temperature 0, then best-of-5 sampling at 0.2 ... 1.0 for windows that fail its thresholds.
# batch_size windows decode at a time; a finished window's slot takes the next one at once.
out = model.decode_windows(["a.wav", "b.wav", ...], language="ja", batch_size=16)
out[0].text, out[0].avg_logprob, out[0].no_speech_prob, out[0].compression_ratio, out[0].temperature

# one pass, CTranslate2-style results (tokens, scores, no-speech probabilities)
res = model.decode(["a.wav", "b.wav", ...], model.prompt("en"), beam_size=5, batch_size=32)
model.detect_language(model.encode(model.features(["a.wav"])))   # [[("en", 0.99), ...]]
```

```bash
whisper-trt build                                  # optional: pre-build the engines
whisper-trt transcribe --language en a.wav b.wav   # --json for every value
whisper-trt detect-language a.wav
```

## Speed

One H200, `openai/whisper-large-v3`, fp16, the language forced, no timestamps, beam 5 (the first pass of
a clip-transcription workload). RTFx: seconds of audio per second of features + encoder + decoding, best of two
passes; faster-whisper 1.2.1 (CTranslate2 4.8.2, fp16) decodes batches of 8 windows as its batched pipeline does
(encoder + generate, its features computed apart).

| set | faster-whisper, batch 8 | **whisper-trt, 8 slots** | **16 slots** | **32 slots** |
|---|---:|---:|---:|---:|
| LibriSpeech test-clean (2620 utterances, 5.4 h) | 103 | **397** (3.9x) | **505** (4.9x) | **550** (5.3x) |
| JVS parallel100 (1000 utterances, 2.2 h) | 99 | **332** (3.4x) | **443** (4.5x) | **491** (5.0x) |
| YODAS v3 Japanese (300 web-audio windows, 2.1 h) | 154 | **501** (3.2x) | **743** (4.8x) | **874** (5.7x) |
| YODAS v3 English (300 windows, 1.9 h) | 215 | **798** (3.7x) | **1060** (4.9x) | **1190** (5.5x) |
| YODAS v3 Hindi (300 windows, 2.0 h) | 106 | **335** (3.2x) | **482** (4.5x) | **574** (5.4x) |

Peak GPU memory 15 / 18 / 25 GiB at 8 / 16 / 32 slots. The engines take about 6 minutes to build on first use.

Where the time goes (YODAS Japanese, 8 slots): 92% of the wall time in decoder steps (3.9 ms each for 40 rows,
the encoder of the next windows running alongside on another stream), loading windows into slots 1%. A step is
about 420 TensorRT kernels: it is bound by kernel latency, not by memory bandwidth (TensorRT's layer profiler at
40 rows: GEMMs 40%, cross-attention 19%, fused LayerNorm/residual kernels 14%, GELU 9%, self-attention 8%).

## Agreement with faster-whisper

Measured window by window against faster-whisper fp16 (beam 5, batch 8), with the decisions its
`generate_with_fallback` takes from a window's decode: silence (no-speech probability > 0.6 and average
log-probability < -1), fallback (compression ratio > 2.4 or average log-probability < -1, not silence) and the
224-token limit. YODAS v3 windows are web audio (music, noise, silence, several speakers) cut from the start of
the first recordings of each language (`scripts/make_eval_yodas.py`: 30 s, or one in three of a random length),
without VAD.

* **Exactness.** In fp32 (`backend="torch", dtype="float32"`), whisper-trt gives faster-whisper fp32's tokens on
  every window: 300/300 for YODAS Japanese, English, German, Hindi, Tamil and Greek each (1800 windows; scores
  within 1.4e-5; Hindi and Tamil have about 110 windows each at the token limit and 60-170 needing the fallback),
  and 73/73 on LibriSpeech dev-clean for greedy and beam 5. The search, the refills and the decisions are
  faster-whisper's.
* **fp16** (the TensorRT engines; 8 and 32 slots give the same agreement within 0.7 points):

| set | windows | identical tokens | WER / CER: whisper-trt vs faster-whisper | decisions agree |
|---|---:|---:|---|---:|
| LibriSpeech test-clean | 2620 | 99.3% | WER 2.78% vs 2.77% | 100% |
| JVS parallel100 | 1000 | 98.9% | CER 6.14% vs 6.11% | 100% |
| FLEURS de / es / ru / fr / en | 862 / 908 / 356 / 289 / 394 | 98.7-99.7% | WER within 0.05 | 100% |
| FLEURS Mandarin / Japanese | 945 / 266 | 97.7% / 98.5% | CER 8.20% vs 8.20% / 3.74% vs 3.78% | 100% |
| YODAS, 18 languages | 5400 | 73.7% (ta) - 97.3% (pt), 90.3% overall | caption CER 33.85% vs 33.89% | 97.7-100% |

  On web audio fp16 decodes differ more often, from faster-whisper's own fp32 decodes too: faster-whisper fp16
  and fp32 give the same tokens on 94.0% / 99.0% / 83.3% of the YODAS ja / en / ta windows, whisper-trt fp16 and
  faster-whisper fp32 on 92.7% / 96.7% / 78.0% (`scripts/ct2_vs_ct2.py`, `parity.py`). Long decodes (Tamil,
  Hindi, Thai, near the token limit) diverge most: one near-tie early on changes the rest. The quality is the
  same: against the videos' own captions (the caption segments whose midpoint falls in a window; an approximate
  reference, the same for both), the character error rate over the 5076 YODAS windows with captions is 33.85%
  for whisper-trt and 33.89% for faster-whisper, within 0.7 points in every language
  (`scripts/yodas_captions.py`). Language identification: top-1 equal to CTranslate2's on 64/64 Japanese and
  English clips. With every window forced through the temperature fallback (`scripts/smoke_workload.py
  --force-fallback`), 64 Japanese clips decode six times each (beam 5, then best-of-5 sampling) at RTFx 57.

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
* **Slots, refilled.** `decode` keeps `batch_size` windows decoding in slots of K rows (K beams). Every slot
  has its own position, so one captured CUDA graph per step serves windows at any point of their decode; every
  4 steps the host takes the results of the finished windows and loads the next windows into their slots. A
  window's prompt tokens are forced tokens of its first steps (no prefill call), and its no-speech probability
  is taken inside the graph, so a load is a few asynchronous writes. The next windows' features and encoder run
  ahead on a lower-priority stream (with TensorRT contexts of their own) and are copied into a slot when it
  frees up. No window waits for the slowest of a batch (+22% on LibriSpeech dev-clean at 8 slots).
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
  under `~/.cache/whisper_trt` (`WHISPER_TRT_CACHE` moves them). Triton kernels are compiled into the decoder
  engine as TensorRT plugins (after owsm-trt's): self-attention reads the KV cache through the slot table (so the
  cache needs no length buckets); cross-attention is split over 8 slices of the 1500 keys ("flash-decoding"),
  which gives one window 160 programs instead of 20 (batch-1 step 2.4 -> 2.0 ms); the vocabulary projection
  accumulates in fp32 (below).
* **Precision.** LayerNorm and the MLPs' GELU are computed in fp32 inside the fp16 engines: TensorRT's fp16 GELU
  multiplied the encoder's error by 5 (`scripts/encoder_variants.py`). TensorRT's fp16 GEMMs accumulate less
  precisely than cuBLAS's (3-9x the error, `scripts/op_precision.py`; TensorRT 11 has no setting for it), and the
  decoder is what this moves: with the PyTorch fp16 decoder behind the TensorRT encoder, YODAS ja / ta agreement
  with faster-whisper fp32 is 283 / 252 of 300, against 264 / 214 with both from TensorRT
  (`scripts/hybrid_agree.py`). The vocabulary projection, which decides beam ties, is therefore a Triton GEMM
  plugin with fp16 inputs and fp32 accumulation: 766 -> 802 of 900 YODAS ja / ta / en windows equal to
  faster-whisper fp32 (faster-whisper fp16 itself: 829), at no step cost from 8 slots on. The PyTorch decoder with
  the same Triton attention kernels (`TorchBackend(kernels=True)`) would be 15-35% slower per step at 40-160 rows
  (`scripts/bench_decoder.py`).

## Limitations

* Decoding with timestamps (and so `transcribe`-style long-form decoding and word timestamps), prompts with
  previous text and `initial_prompt` are not implemented yet; windows are decoded on their own, as
  faster-whisper's batched pipeline decodes clips.
* Every input is one window of at most 30 s: longer audio is trimmed, not chunked (features are normalized
  over the whole input, as faster-whisper normalizes a clip).
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
* `scripts/make_eval_yodas.py`, `scripts/prepare_waves.py`: the evaluation sets (`.cache/eval`);
  `scripts/mismatch_summary.py`, `scripts/ct2_vs_ct2.py`, `scripts/yodas_captions.py`: their analysis.
* `scripts/profile_stream.py`, `scripts/profile_decode.py`, `scripts/trt_profile.py`, `scripts/bench_decoder.py`:
  time per part of a streamed decode / per stage / per TensorRT layer / per decoder step.
* `scripts/check_engines.py`, `scripts/encoder_variants.py`, `scripts/op_precision.py`,
  `scripts/teacher_force.py`, `scripts/search_debug.py`: component-level checks of the engines.
* `pytest tests`: search against CTranslate2's loops (refilled slots included), sampling, the Triton kernels (GPU
  tests skip on CPU).
