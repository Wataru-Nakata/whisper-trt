# whisper-trt

Fast inference for OpenAI's Whisper (`openai/whisper-large-v3`; other Hugging Face Whisper checkpoints load the
same way) with TensorRT, decoding as faster-whisper / CTranslate2 does: the same features, beam search,
sampling, token limit, suppressed tokens and scores. It does not depend on faster-whisper, CTranslate2 or
transformers.

```python
from whisper_trt import Whisper

model = Whisper.from_pretrained("openai/whisper-large-v3")   # first call builds the TensorRT encoder (~2 min, cached)

# up-to-30 s windows, each decoded on its own as faster-whisper's generate_with_fallback decodes a window:
# beam 5 at temperature 0, then best-of-5 sampling at 0.2 ... 1.0 for windows that fail its thresholds.
# batch_size windows decode at a time; a finished window's slot takes the next one at once.
out = model.decode_windows(["a.wav", "b.wav", ...], language="ja", batch_size=16)
out[0].text, out[0].avg_logprob, out[0].no_speech_prob, out[0].compression_ratio, out[0].temperature

# with timestamps: each window's segments, as faster-whisper splits them
out = model.decode_windows(["a.wav", ...], language="ja", without_timestamps=False)
[(seg["start"], seg["end"], seg["text"]) for seg in out[0].segments]

# recordings of any length, as faster-whisper's WhisperModel.transcribe transcribes each (seek, previous-text
# prompts, temperature fallback, timestamps; word timestamps optional), many recordings decoding together
segments = model.transcribe(["long1.wav", "long2.wav", ...], "ja", word_timestamps=True, batch_size=32)
[(s.start, s.end, s.text, [(w.word, w.start, w.end) for w in s.words]) for s in segments[0]]
model.transcribe(["long1.wav", ...])[0].language   # no language: each recording's detected (its first 30 s)

# one pass, CTranslate2-style results (tokens, scores, no-speech probabilities); a prompt without
# <|notimestamps|> (model.prompt("en", without_timestamps=False)) decodes with timestamps
res = model.decode(["a.wav", "b.wav", ...], model.prompt("en"), beam_size=5, batch_size=32)
model.detect_language(model.encode(model.features(["a.wav"])))   # [[("en", 0.99), ...]]
```

```bash
whisper-trt build                                  # optional: pre-build the encoder engine
whisper-trt transcribe --language en a.wav b.wav   # --json for every value, --timestamps for segments
whisper-trt transcribe --long --word-timestamps long1.wav long2.wav   # any length (--language: else detected)
whisper-trt detect-language a.wav
```

## Speed

One H200, `openai/whisper-large-v3`, fp16, the language forced, no timestamps, beam 5. RTFx: seconds of audio
per second of features + encoder + decoding, best of two passes; faster-whisper 1.2.1 (CTranslate2 4.8.2, fp16)
decodes batches of 8 windows as its batched pipeline does (encoder + generate in one pass; its features computed
apart).

| set | faster-whisper, batch 8 | **whisper-trt, 8 slots** | **16 slots** | **32 slots** |
|---|---:|---:|---:|---:|
| LibriSpeech test-clean (2620 utterances, 5.4 h) | 103 | **460** (4.5x) | **569** (5.5x) | **633** (6.1x) |
| JVS parallel100 (1000 utterances, 2.2 h) | 99 | **392** (4.0x) | **502** (5.1x) | **578** (5.8x) |
| YODAS v3 Japanese (300 web-audio windows, 2.1 h) | 154 | **637** (4.1x) | **878** (5.7x) | **1050** (6.8x) |
| YODAS v3 English (300 windows, 1.9 h) | 215 | **946** (4.4x) | **1230** (5.7x) | **1407** (6.5x) |
| YODAS v3 Hindi (300 windows, 2.0 h) | 106 | **421** (4.0x) | **588** (5.5x) | **717** (6.8x) |

Peak GPU memory 15 / 18 / 25 GiB at 8 / 16 / 32 slots. The encoder engine takes about 2 minutes to build on
first use. With the TensorRT decoder engine instead (`decoder="tensorrt"`, same job): 413 / 530 / 584 on
LibriSpeech, 537 / 789 / 918 on YODAS Japanese. 64 slots (`from_pretrained(max_batch=64, max_rows=320)`,
38-41 GiB) gave another 3-4% (before the fused beam bookkeeping: 649 on LibriSpeech, 1058 / 1422 on YODAS
Japanese / English).

Where the time goes (YODAS Japanese, `scripts/profile_stream.py`, `scripts/profile_body.py`): 98% of the wall
time in decoder steps at 8 slots (with the encoder of the next windows running alongside on another stream; a
step for 40 rows takes 2.8 ms alone at 80 cached positions, of which 0.16 ms beam search's bookkeeping), 93% at
32 slots (5.1 ms alone, 0.37 ms bookkeeping). The encoder costs 6.3 ms a window whatever
the batch (`scripts/bench_encoder.py`), so on short utterances, padded to 30 s windows like faster-whisper's, it
is about half of the GPU time at 32 slots (LibriSpeech: 2620 windows, 16.5 s of the 31.5 s). The encoder and
the decoder steps barely overlap (17% of the shorter, `scripts/overlap.py`): the decoder's ~400 small kernels
wait for SMs the encoder's GEMMs hold, so the two take turns. Measured and not taken: the encoder confined to
fewer SMs (a CUDA green context, its profile built for them: more overlap, more time in all, 1413-1535 against
1323 ms for 120 steps and 100 encodes), the encoder in a CUDA graph (7.35 against 7.52 ms an encode), an
encoder of cuBLAS GEMMs, cuDNN attention and Triton LayerNorm/GELU (~5.7 against 6.3 ms a window by its parts,
`scripts/bench_encoder_parts.py`), TensorRT's builder optimization level 5 (6.3 against 6.6 ms a window),
PyTorch's TunableOp on the decoder's GEMMs (within 2%: each small-M GEMM costs ~8.5 us whatever the algorithm);
for those GEMMs, Triton kernels with programmatic dependent launch or with split-K in one kernel (a tile's last
K slice adds them up: 1.63 against cuBLAS's 1.19 ms for a step's 192 GEMMs at 40 rows, `scripts/bench_splitk.py`)
and the next GEMM's weights read into L2 from a side stream (2.59 ms); a two-stage top-k in the beam
bookkeeping.

### FP8 encoder (opt-in)

`Whisper.from_pretrained(..., fp8=True)` (`whisper-trt transcribe --fp8`; Hopper or newer) runs the encoder MLPs'
GEMMs in FP8 E4M3 (`whisper_trt/fp8.py`: per-tensor input scales from the scale file the package ships,
`whisper_trt/fp8_scales/whisper-large-v3.json`, calibrated by `scripts/fp8_calibrate.py` on 742 YODAS windows in
13 languages from recordings outside the evaluation sets; per-channel weight scales). The encoder takes 5.6
instead of 6.6 ms a window. End to end, at 8 / 16 / 32 slots, same job:

| set | fp16 | **FP8 MLPs** |
|---|---|---|
| LibriSpeech test-clean | 461 / 569 / 632 | **506 / 626 / 691** (+9-10%) |
| JVS parallel100 | 403 / 505 / 578 | **415 / 540 / 623** (+3-8%) |
| YODAS Japanese | 671 / 897 / 1059 | **661 / 922 / 1108** (-1 to +5%) |
| YODAS English | 947 / 1229 / 1406 | **986 / 1311 / 1508** (+4-7%) |
| YODAS Hindi | 401 / 583 / 710 | **412 / 590 / 728** (+1-3%) |

The decodes change much more than the error rates: identical to faster-whisper fp16 on 93.0% of the clean sets'
windows (fp16 encoder: 98.9%) and 66% of the YODAS windows (91%; Tamil 109 / 300), with the error rates about
the same (LibriSpeech WER 2.77% vs 2.77%, JVS CER 6.15% vs 6.11%, FLEURS within 0.07 but Russian 5.08% vs
4.86%; YODAS caption CER 34.01% vs faster-whisper's 33.89%, per language within 1.9 points) and the decisions
(silence, fallback, limit) agreeing on 94.7-100% of the windows per set. More of the encoder in FP8 is faster but
changes even more (`fp8_layers=`, measured on YODAS ja / en, 300 each, identical to faster-whisper fp16): every
linear layer, 5.3 ms a window, 189 / 203; all but the cross-attention projections 191 / 212; the MLPs (default)
5.6 ms, 222 / 239; fc1 alone 6.2 ms, 236 / 258; the fp16 encoder 282 / 288. FP8's 3-bit mantissa is the limit,
not the scales.

## Agreement with faster-whisper

Measured window by window against faster-whisper fp16 (beam 5, batch 8), with the decisions its
`generate_with_fallback` takes from a window's decode: silence (no-speech probability > 0.6 and average
log-probability < -1), fallback (compression ratio > 2.4 or average log-probability < -1, not silence) and the
224-token limit. YODAS v3 windows are web audio (music, noise, silence, several speakers) cut from the start of
the first recordings of each language (`scripts/make_eval_yodas.py`: 30 s, or one in three of a random length),
without VAD.

* **Exactness.** In fp32 (`backend="torch", dtype="float32"`), whisper-trt gives faster-whisper fp32's tokens on
  every window: 300/300 for YODAS Japanese, English, German, Hindi, Tamil and Greek each (1800 windows; scores
  within 1.5e-5; Hindi and Tamil have about 110 windows each at the token limit and 60-170 needing the fallback),
  and 73/73 on LibriSpeech dev-clean for greedy and beam 5. The search, the refills and the decisions are
  faster-whisper's.
* **fp16** (8 slots; 32 slots: within 0.3 points on every clean set, 2.3 on a YODAS language, 91.5% overall):

| set | windows | identical tokens | WER / CER: whisper-trt vs faster-whisper | decisions agree |
|---|---:|---:|---|---:|
| LibriSpeech test-clean | 2620 | 99.2% | WER 2.78% vs 2.77% | 100% |
| JVS parallel100 | 1000 | 98.7% | CER 6.14% vs 6.11% | 100% |
| FLEURS de / es / ru / fr / en | 862 / 908 / 356 / 289 / 394 | 98.6-99.7% | WER within 0.05 | 100% |
| FLEURS Mandarin / Japanese | 945 / 266 | 97.6% / 98.9% | CER 8.26% vs 8.20% / 3.71% vs 3.78% | 100% |
| YODAS, 18 languages | 5400 | 77.3% (ta) - 97.7% (es), 91.2% overall | caption CER 33.87% vs 33.89% | 97.7-100% |

  On web audio fp16 decodes differ more often, from faster-whisper's own fp32 decodes too: faster-whisper fp16
  and fp32 give the same tokens on 94.0% / 99.0% / 83.3% of the YODAS ja / en / ta windows, whisper-trt fp16 and
  faster-whisper fp32 on 94.3% / 96.7% / 82.7% (`scripts/ct2_vs_ct2.py`, `parity.py`). Long decodes (Tamil,
  Hindi, Thai, near the token limit) diverge most: one near-tie early on changes the rest. The quality is the
  same: against the videos' own captions (the caption segments whose midpoint falls in a window; an approximate
  reference, the same for both), the character error rate over the 5076 YODAS windows with captions is 33.87%
  for whisper-trt and 33.89% for faster-whisper, within 0.6 points in every language
  (`scripts/yodas_captions.py`). Language identification: top-1 equal to CTranslate2's on 64/64 Japanese and
  English clips. With every window forced through the temperature fallback (`scripts/smoke_workload.py
  --force-fallback`), 64 Japanese clips decode six times each (beam 5, then best-of-5 sampling) at RTFx 57.
* **With timestamps** (the prompt without `<|notimestamps|>`, against faster-whisper decoding with timestamps
  too, `scripts/ref_ct2.py --timestamps`): in fp32 the tokens, timestamps included, are faster-whisper fp32's
  on 300/300 YODAS windows each of ja, en, de, hi, ta and el (1800 windows; scores within 7e-6). In fp16 they are identical to
  faster-whisper fp16's on 97.7% of the clean sets' windows (95.5% FLEURS fr - 98.2% LibriSpeech; WER/CER
  within 0.06, e.g. LibriSpeech 3.33% vs 3.27%) and 82.9% of the 5400 YODAS windows (66.0% ta - 90.7% ja),
  the decisions agreeing on 98-100%. Timestamps add near-ties (a timestamp one 20 ms step later, a segment
  closed one token earlier), for faster-whisper itself too: its fp16 and fp32 agree on 275 / 282 / 270 / 248 /
  202 / 260 of 300 YODAS ja / en / de / hi / ta / el windows, whisper-trt fp16 and faster-whisper fp32 on 274 /
  285 / 267 / 259 / 224 / 266. faster-whisper slows down with timestamps (its rules run on the host every
  step: 62 RTFx on LibriSpeech, 84 on YODAS ja), whisper-trt hardly (in the step graph: 405 / 600 and 498 /
  884 at 8 / 32 slots, one pass).

## Long-form transcription

`transcribe` against faster-whisper 1.2.1's `WhisperModel.transcribe` with its defaults (beam 5, the temperature
fallback, conditioned on the previous text, timestamps, no VAD), with and without word timestamps, on 20
recordings per set: YODAS v3 Japanese and English (the first minutes of recordings, 60-300 s, 1.17 h / 1.45 h,
`scripts/make_eval_long.py`) and LibriSpeech test-clean chapters (a chapter's utterances joined with 0.3 s
pauses, 78-239 s, 1.02 h, `scripts/make_long_ls.py`). One H200, RTFx over all the recordings, features
included (8 slots: best of two runs). faster-whisper transcribes one recording at a time; so can whisper-trt, or
the windows of all the recordings stream through `batch_size` slots.

| set | faster-whisper fp32 | fp16 | **whisper-trt fp16, one recording at a time** | **20 recordings, 8 slots** | **32 slots** |
|---|---:|---:|---:|---:|---:|
| YODAS Japanese | 15.9 | 23.5 | **64.2** (2.7x) | **309** (13.1x) | **318** (13.5x) |
| with word timestamps | 16.0 | 23.0 | **56.7** (2.5x) | **222** (9.7x) | **260** (11.3x) |
| YODAS English | 14.7 | 19.4 | **49.5** (2.6x) | **216** (11.1x) | **211** (10.9x) |
| with word timestamps | 14.5 | 22.2 | **52.4** (2.4x) | **218** (9.8x) | **202** (9.1x) |
| LibriSpeech chapters | 10.9 | 16.1 | **38.1** (2.4x) | **171** (10.6x) | **248** (15.4x) |
| with word timestamps | 11.1 | 15.2 | **39.0** (2.6x) | **160** (10.5x) | **181** (11.9x) |

(x: against faster-whisper fp16.) A recording's windows depend on each other (the seek, the previous text):
alone, its windows decode one after another in one slot. Together, a recording's next window (or a window's
retry at the next temperature) takes a slot as soon as the result before it is back, so no window waits for
another recording's; 20 recordings fill at most 20 slots, which is why 32 slots add little on English (a
recording through many fallbacks is the last to finish). Word timestamps run alongside the decoding (below).
Before (rounds of every recording's next window, waiting for the slowest, and a second encoder pass and a
host-side time warping per aligned window): 170 / 190, 106 / 146, 117 / 122, 87 / 98, 81 / 98, 77 / 97.

Agreement with faster-whisper fp32, next to faster-whisper fp16's own (8 slots):

| set | | recordings with the same segments | segments the same before the first sampled window | CER / WER of the transcripts |
|---|---|---:|---:|---:|
| YODAS Japanese | faster-whisper fp16 | 3/20 | 352/803 | 3.53% |
| | **whisper-trt fp16** | **5/20** | **455/818** | **2.26%** |
| | whisper-trt fp32 | 13/20 | 732/820 | 1.46% |
| YODAS English | faster-whisper fp16 | 2/20 | 297/525 | 20.43% |
| | **whisper-trt fp16** | **3/20** | **425/525** | **20.94%** |
| | whisper-trt fp32 | 5/20 | 499/529 | 17.29% |
| LibriSpeech chapters | faster-whisper fp16 | 2/20 | 146/183 | 9.97% |
| | **whisper-trt fp16** | **2/20** | **146/182** | **11.45%** |
| | whisper-trt fp32 | 3/20 | 184/207 | 9.17% |

The same segments: the same tokens, start and end. A recording agrees at most up to its first window decoded by
sampling (a window failing the fallback's thresholds at temperature 0, in either decode): sampled tokens are
random (our random numbers are not CTranslate2's), they become the prompt of the next windows, and the rest of
the recording differs. "Before the first sampled window" counts the segments the same up to a recording's first
difference, of those before either decode's first sampled window. The fallback is frequent on these sets
(faster-whisper fp32 samples 42 of 917 segments of YODAS Japanese, 276 of 1065 English and 73 of 562
LibriSpeech, in 16 of the 20 chapters), so whole recordings rarely agree and the transcripts' error rates against
faster-whisper fp32 are mostly those of two samplings. The quality is the same: on the LibriSpeech chapters the
WER against the reference text is 5.13-10.64% for whisper-trt fp16 over six seeds of the sampling (`seed=0..5`,
mean 7.2%), 6.33% for whisper-trt fp32, 7.98% for faster-whisper fp32 and 5.82% for fp16.

* **fp32**: 18 / 19 / 19 of the 20 recordings of each set agree with faster-whisper fp32 up to their first
  sampled window (with word timestamps 18 / 17 / 19); the differences before it are near-ties (a timestamp or a
  word's start 20 ms apart, a comma for a period, one word for another, a segment closed one word earlier).
  These windows have prompts of previous text (prefilled, up to 223 tokens), which the window-level comparison
  above doesn't.
* **The same output whatever the timing:** a window's samples depend on the seed, the recording's index, the
  window's and the temperature's (not on the slot or the moment it decodes in), so a second run gives the same
  segments (20 / 20 on every set, with and without word timestamps). One recording at a time (one slot) gives
  the same segments as twenty together up to their first sampled window on every set, fp16 included (the
  samples then differ: each call's recording is its first).

With word timestamps (the seek then follows the last word's end, so a word's timing moves the next window too);
words compared on the segments the same up to a recording's first difference:

| set | | recordings with the same segments | segments the same before the first sampled window | CER / WER | words with the same start and end |
|---|---|---:|---:|---:|---:|
| YODAS Japanese | faster-whisper fp16 | 4/20 | 289/763 | 3.00% | 98.7% of 6034 |
| | **whisper-trt fp16** | **1/20** | **384/838** | **1.75%** | **98.3% of 6464** |
| | whisper-trt fp32 | 15/20 | 920/928 | 0.50% | 99.9% of 15172 |
| YODAS English | faster-whisper fp16 | 1/20 | 201/595 | 15.90% | 98.6% of 2623 |
| | **whisper-trt fp16** | **1/20** | **208/602** | **14.92%** | **97.8% of 2684** |
| | whisper-trt fp32 | 4/20 | 547/603 | 13.89% | 99.9% of 5913 |
| LibriSpeech chapters | faster-whisper fp16 | 2/20 | 111/187 | 10.48% | 98.3% of 3338 |
| | **whisper-trt fp16** | **1/20** | **91/188** | **11.94%** | **97.7% of 2649** |
| | whisper-trt fp32 | 4/20 | 202/213 | 14.30% | 99.8% of 4328 |

The other words are within one 20 ms step but a few (whisper-trt fp16: 99.8% within 0.02 s, at most 0.12 s
off; fp32: all within 0.02 s but two of 0.06 s; faster-whisper fp16: 99.9-100%, at most 0.22 s). WER against the
LibriSpeech text: 7.40% for whisper-trt fp16, 9.65% for fp32 (one sampling each), 6.17% / 6.34% for
faster-whisper fp32 / fp16.

Without a language (`transcribe(audio)`, faster-whisper's `language=None`), each recording's is detected from
its first 30 s of the whole-recording features: on the 60 recordings above (fp32), the language faster-whisper's
`transcribe` detects on all 60, probabilities within 9e-5 (`scripts/lid_long.py`).

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
  has its own position, so one captured CUDA graph per step serves windows at any point of their decode. After
  every step the slots' done flags are copied to the host asynchronously and read one step later, so the GPU
  always has a step queued while the host takes the results of the finished windows (copied asynchronously
  too) and loads the next windows into their slots (+2-4% over blocking checks every 4 steps). A window's
  prompt tokens are forced tokens of its first steps (no prefill call), and its no-speech probability is taken
  inside the graph, so a load is a few asynchronous writes. The next windows' features and encoder run ahead on
  a lower-priority stream (with TensorRT contexts of their own) and are copied into a slot when it frees up. No
  window waits for the slowest of a batch (+22% on LibriSpeech dev-clean at 8 slots).
* **Timestamps.** A prompt without `<|notimestamps|>` decodes with CTranslate2's timestamp rules
  (`ApplyTimestampRules`) after the suppressed tokens, per row from its own history, inside the step graph:
  never `<|notimestamps|>`; a timestamp first, at most 1 s (`max_initial_timestamp_index`, 50); timestamps in
  pairs and never decreasing; text only if some text token is likelier than all timestamps together. Each
  window's tokens split into segments as faster-whisper splits them (`whisper_trt/segments.py`).
* **Long-form** (`whisper_trt/longform.py`). A recording's features are computed once over the whole recording
  (one normalization, as faster-whisper's), and each window is up to 30 s of them from the recording's seek,
  with faster-whisper's prompt (`<|startofprev|>` and at most 223 tokens of the text since the last reset).
  Without a language, each recording's is detected from its first 30 s (faster-whisper's `detect_language`: the
  top language of the first `language_detection_segments` windows, by `language_detection_threshold`). The
  windows of all the recordings stream through one search (`Search.serve`: a source gives windows as results
  come back): a window's result decides at once between its retry at the next temperature (queued first) and its
  recording's next window, so slots refill without waiting for any other recording. Windows at temperature 0
  (beam search) and the fallback's samples (best_of rows of a slot, at the slot's own temperature and seed)
  decode in the same steps: per slot, the fused kernels take beam search's or GreedySearch's rules
  (`tests/test_serve.py`: mixed slots against CTranslate2's loops, windows released one by one at 2 and 4
  slots). A window's keys/values, encoded once into a pool (`kv_pool` buffers of 245 MB, by default
  `min(recordings, batch_size + 8)`), serve its retries and its word alignment. Then faster-whisper's rules: a
  silent window skipped, the tokens split into segments, the seek moved to the last timestamp (with word
  timestamps, the last word's end), the prompt reset after a decode above temperature 0.5,
  `hallucination_silence_threshold` (implemented, not compared: the comparisons above use faster-whisper's
  defaults, as for `initial_prompt`). Prompts of any length run in the search's prefilled mode: a window's
  prompt but its last token goes through one decoder call (outside the captured graph) into its slot's cache,
  and the step graph goes on from the slot's own position, with CTranslate2's limit for that prompt, `min(224,
  448 - prompt + 1)` new tokens (`tests/test_search.py`: prompts of different lengths in refilled slots, against
  CTranslate2's loops).
* **Word timestamps** (`whisper_trt/alignment.py`): CTranslate2's `align`, then faster-whisper's words. On the
  window's kept keys/values, one call of the PyTorch decoder on the same weights over `sot_sequence,
  <|notimestamps|>`, the text tokens and `<|endoftext|>` (in a CUDA graph per length bucket, the tokens padded
  at the end: the decoder is causal): the alignment heads' cross-attention softmaxed over the window's frames,
  normalized over the tokens, median-filtered (width 7) and averaged over the heads; a dynamic time warping with
  CTranslate2's tie-breaking in one Triton kernel (anti-diagonals in a block's lanes, each row's first frame
  kept: where faster-whisper's words start; `tests/test_alignment.py`: against line-by-line transcriptions of
  `dtw.cc` and `median_filter_cpu.cc`); then faster-whisper's word splitting, punctuation merging and duration
  clamps. All of it runs on the side stream while the search goes on, the host reading each window's first
  frames and token probabilities once they are there (`alignment.Aligner`): 7.7-8.2 ms of GPU time a window
  (YODAS ja / en, 69-91 text tokens on average), the same first frames as the host's on 39 of 40 windows.
  Before: a second encoder pass (6.4 ms), the decoder call op by op (11.7 ms) and the time warping on the host
  (27-28 ms, a numpy step per anti-diagonal), 47-49 ms a window with the search waiting
  (`scripts/profile_align.py`).
* **Beam search's bookkeeping in two Triton kernels** (`whisper_trt/beam_kernels.py`): per row, the masks,
  log-softmax, cumulative score and no-speech probability; `torch.topk` over each window's K x V scores; per
  window, CTranslate2's rules (finished hypotheses, the replacement of ended candidates, the new beams' histories
  and slot tables, prompt feeding). Instead of ~80 small PyTorch kernels: 0.33 -> 0.16 ms a step at 8 slots,
  0.63 -> 0.37 ms at 32, the same decodes (the PyTorch rules stay the reference: `WHISPER_TRT_FUSED_BEAM=0`, and
  the tests check both against CTranslate2's loops).
* **Beam search never moves the KV cache.** Each row has a table of the row whose cache holds each of its
  slots; a new beam takes its parent's table (a few KB) instead of copying the parent's cache (hundreds of MB
  per step at 8 windows x 5 beams). A window's beams share its cross-attention keys/values: their queries are
  attended together, as K x S queries of one window.
* **Sampling inside the graphs.** The temperature fallback samples with the Gumbel-max trick on random numbers
  from a counter-based hash of (the window's seed, step, row of the window, token): no generator state, so it
  runs inside the captured graphs, and a window's samples do not depend on the slot or the moment it decodes in;
  the same seed gives the same samples (`decode_windows(..., seed=)`: each window's from the seed and its index;
  `transcribe`: from the seed, the recording's index, the window's and the temperature's). Its numbers are not
  CTranslate2's (curand), so sampled decodes agree with faster-whisper in distribution only.
* **The encoder is a TensorRT engine**, built on first use (about 2 minutes): the encoder with every decoder
  layer's cross-attention projection, exported to ONNX and compiled into a strongly-typed fp16 engine, cached
  under `~/.cache/whisper_trt` (`WHISPER_TRT_CACHE` moves it).
* **The decoder step is cuBLAS GEMMs and Triton kernels** (`runtime.TritonDecoderBackend`), captured in the
  step's CUDA graph: per layer, a LayerNorm kernel (fp32 statistics) that also adds the bias of the next residual
  GEMM to the residual stream, which the GEMM then adds its product to (`addmm_`: no residual kernel);
  self-attention with one query per (row, head) that reads the KV cache through the slot table (so the cache
  needs no length buckets) and writes the new token's keys/values into it; cross-attention split over 8 slices
  of the 1500 keys ("flash-decoding": 160 programs per window instead of 20); the vocabulary projection with fp32
  accumulation. Per step on an H200 this is 2.4 ms at 8 windows x 5 beams and 4.8 ms at 32 x 5, against 3.2 and
  5.9 ms for the TensorRT decoder engine (with the same attention kernels as TensorRT plugins, after owsm-trt's;
  `decoder="tensorrt"`), and closer to fp32 (below). Cross-attention reads every window's 245 MB of keys/values
  each step at ~3.9 TB/s and is the largest part at 32 windows; 8-bit keys/values would halve it but change too
  many decodes (FP8 e4m3 or int8 with per-channel scales: 193-199 instead of 234 of 300 YODAS Tamil windows equal
  to faster-whisper fp32, `scripts/fp8_cross_agree.py`), so they stay fp16.
* **Precision.** LayerNorm and the MLPs' GELU are computed in fp32 inside the fp16 encoder engine: TensorRT's
  fp16 GELU multiplied the encoder's error by 5 (`scripts/encoder_variants.py`). TensorRT's fp16 GEMMs accumulate
  less precisely than cuBLAS's (3-9x the error, `scripts/op_precision.py`; TensorRT 11 has no setting for it),
  and the decoder is what this moves: with the PyTorch fp16 decoder behind the TensorRT encoder, YODAS ja / ta
  agreement with faster-whisper fp32 is 283 / 252 of 300, against 264 / 214 with both from TensorRT
  (`scripts/hybrid_agree.py`). Hence the decoder of cuBLAS GEMMs (fp32 accumulation; this turns off
  `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`) and fp32-accumulating Triton kernels: its
  logits are within 0.02 of the fp32 decoder's (the TensorRT decoder engine: 0.04-0.05), and its decodes equal
  faster-whisper fp32's on 283 / 290 / 292 / 274 / 248 of 300 YODAS ja / en / de / hi / ta windows (the TensorRT
  decoder engine, already with an fp32-accumulating vocabulary projection: 278 / 290 / 289 / 260 / 234;
  faster-whisper fp16 itself on ja / en / ta: 282 / 297 / 250).

## Limitations

* `decode` / `decode_windows` take windows of at most 30 s (longer audio is trimmed; features are normalized
  over the whole input, as faster-whisper normalizes a clip); `transcribe` takes recordings of any length.
* `transcribe` has no VAD filter, `clip_timestamps`, `prefix`, `hotwords` or per-segment language detection
  (`multilingual`), nor `best_of` > `beam_size` > 1 (the samples decode in the beam search's rows). Its pool
  of kept keys/values adds up to `kv_pool` x 245 MB to the decoding's memory (by default `min(recordings,
  batch_size + 8)` buffers: about 10 GB at 32 slots; with a smaller `kv_pool`, the windows left out are encoded
  again for their retries and alignment). Long prompts (previous text) and word timestamps need the default
  Triton decoder or the PyTorch backend (the TensorRT decoder engine prefills at most 32 tokens).
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
  `scripts/mismatch_summary.py`, `scripts/ct2_vs_ct2.py`, `scripts/yodas_captions.py`: their analysis
  (`yodas_captions.py` needs pyarrow, in neither environment: `uv run --no-project --with pyarrow python ...`).
* `scripts/profile_stream.py`, `scripts/profile_decode.py`, `scripts/trt_profile.py`, `scripts/bench_decoder.py`:
  time per part of a streamed decode / per stage / per TensorRT layer / per decoder step;
  `scripts/profile_body.py` (decoder call against beam bookkeeping), `scripts/profile_triton_decoder.py` (per
  kernel), `scripts/sync_check.py` (host lines that synchronize with the GPU).
* `scripts/bench_self_attn.py`, `scripts/bench_gemm_chain.py`, `scripts/bench_encoder.py`,
  `scripts/bench_encoder_gemms.py`, `scripts/bench_encoder_torch.py`, `scripts/trt_profile_encoder.py`: kernel
  and engine measurements behind the choices above (Triton GEMMs and programmatic dependent launch lost to
  cuBLAS on the decoder's small-M GEMMs; the PyTorch encoder is slower than TensorRT's, cuBLAS alone ~10% faster
  on its GEMMs); `scripts/fp8_cross_agree.py`: 8-bit cross-attention keys/values (rejected);
  `scripts/build_check.py`: engine builds from an empty cache; `scripts/fused_check.py`: the beam bookkeeping
  kernels against the PyTorch rules on real windows.
* `scripts/fp8_calibrate.py`: the FP8 scale file (from `scripts/make_eval_yodas.py --tar -1`: recordings
  outside the evaluation sets).
* `scripts/make_eval_long.py`, `scripts/make_long_ls.py`: the long-form sets; `scripts/ref_long.py`:
  faster-whisper's `transcribe` on them (in the reference env); `scripts/longform_parity.py`: ours against it
  (`--single`: one recording at a time; `--seed`); `scripts/longform_compare.py`: the same scores from saved
  outputs, and the WER against a set's reference text; `scripts/longform_smoke.py`: a few recordings' segments
  and words; `scripts/profile_align.py`: where word alignment's time goes; `scripts/lid_long.py`: language
  detection against faster-whisper's.
* `scripts/check_engines.py`, `scripts/encoder_variants.py`, `scripts/op_precision.py`,
  `scripts/teacher_force.py`, `scripts/search_debug.py`: component-level checks of the engines.
* `pytest tests`: search against CTranslate2's loops (refilled slots, prompts of any length, beam-searched and
  sampled windows in the same steps, windows given as results come back), timestamp rules, sampling, word
  alignment's time warping (host and kernel) and median filter, the Triton kernels (GPU tests skip on CPU).
