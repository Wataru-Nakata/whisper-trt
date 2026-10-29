"""Long-form transcription as faster-whisper 1.2.1's ``WhisperModel.transcribe`` decodes a recording
(``generate_segments``), for many recordings at a time.

Each recording's features are computed over the whole recording (one normalization), and decoded window by
window from a seek position: a 30 s window of features from the seek, its prompt the recording's previous text
(``condition_on_previous_text``; reset after a decode at a temperature above ``prompt_reset_on_temperature``),
decoded with the temperature fallback; a window judged silent is skipped; its tokens split into segments by
their timestamps, the next window starting at the last timestamp (or after the window); with word timestamps,
each segment's words, the next window starting at the last word's end. Without a language, each recording's is
detected from its first 30 s (faster-whisper's ``detect_language``).

A window depends on the one before it in its recording, so the windows stream through one search
(Search.serve): a recording's next window, or a window's retry at the next temperature (sampled, in the same
steps as the windows decoded by beam search), is given to the search as soon as the result before it is back.
A window's keys/values are kept (a pool of buffers) for its retries and its word alignment, which runs on a
stream of its own (alignment.Aligner) while the search goes on.
"""

from collections import deque
from dataclasses import dataclass

import torch

from .alignment import Aligner, add_word_timestamps, align, find_alignment
from .audio import load_audio
from .config import HOP_LENGTH, N_FRAMES, SAMPLE_RATE, TIME_PRECISION
from .features import log_mel
from .search import Job, mix_seed
from .segments import split_by_timestamps

FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # feature frames
TIME_PER_FRAME = HOP_LENGTH / SAMPLE_RATE
PUNCTUATION = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[Word] | None
    temperature: float


class Transcript(list):
    """A recording's segments (a list of :class:`Segment`), with its language: the one given, or the one detected
    (``language_probability``, ``all_language_probs`` as faster-whisper's ``TranscriptionInfo``)."""

    def __init__(self, segments=(), language: str | None = None, language_probability: float = 1.0,
                 all_language_probs: list[tuple[str, float]] | None = None):
        super().__init__(segments)
        self.language = language
        self.language_probability = language_probability
        self.all_language_probs = all_language_probs


class _Recording:
    def __init__(self, index: int, features: torch.Tensor, initial_tokens: list[int]):
        self.index = index
        self.features = features  # [n_mels, frames + 1], normalized over the whole recording
        self.content_frames = features.shape[-1] - 1  # (faster-whisper's: the last frame is not decoded)
        self.seek = 0
        self.all_tokens = list(initial_tokens)
        self.reset_since = 0
        self.last_speech = 0.0
        self.windows = 0  # windows decoded so far (their random numbers' seeds)
        self.segments: list[Segment] = []
        self.language: str | None = None
        self.language_probability = 1.0
        self.all_language_probs = None
        self.sot_sequence: list[int] = []


class _Window:
    def __init__(self, rec: _Recording, features: torch.Tensor, size: int, prompt: list[int]):
        self.rec, self.features, self.size, self.prompt = rec, features, size, prompt
        self.seek = rec.seek
        self.ordinal = rec.windows
        rec.windows += 1
        self.ti = 0  # the temperature being decoded
        self.attempts: list[dict] = []
        self.buffer: int | None = None  # its keys/values in the pool, if kept


def recording_features(model, audio) -> torch.Tensor:
    """faster-whisper's ``feature_extractor(audio)``: [n_mels, len // 160 + 1] over the whole recording
    (normalized once; the last frame only counts for the language detection)."""
    wave = audio if isinstance(audio, torch.Tensor) and audio.dim() == 1 else load_audio(audio)
    frames = wave.shape[-1] // HOP_LENGTH + 1
    return log_mel([wave], model.cfg.n_mels, model.device, frames)[0]


def get_end(segments: list[dict]) -> float | None:
    """faster-whisper's ``get_end``: the last word's end, else the last segment's."""
    return next((w["end"] for s in reversed(segments) for w in reversed(s["words"])),
                segments[-1]["end"] if segments else None)


def detect_languages(model, recs: list[_Recording], segments: int = 1, threshold: float | None = 0.5) -> None:
    """faster-whisper's ``detect_language`` on each recording's features: its first 30 s windows (up to
    ``segments``) one by one, the first whose top language is likelier than ``threshold``, else the most frequent
    top language (its best probability)."""
    c = model.cfg
    for rec in recs:
        votes: dict[str, list[float]] = {}
        feats = rec.features[:, : segments * N_FRAMES]
        for i in range(0, max(feats.shape[-1], 1), N_FRAMES):
            window = torch.zeros(c.n_mels, N_FRAMES, device=model.device)
            part = feats[:, i : i + N_FRAMES]
            window[:, : part.shape[-1]] = part
            probs = model.detect_language(model.encode(window[None]))[0]
            language, p = probs[0]
            rec.all_language_probs = probs
            if threshold is not None and p > threshold:
                break
            votes.setdefault(language, []).append(p)
        else:
            language = max(votes, key=lambda k: len(votes[k]))
            p = max(votes[language])
        rec.language, rec.language_probability = language, p


def transcribe(model, audio: list, language: str | None = None, task: str = "transcribe", beam_size: int = 5,
               best_of: int = 5, temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
               compression_ratio_threshold: float | None = 2.4, log_prob_threshold: float | None = -1.0,
               no_speech_threshold: float | None = 0.6, condition_on_previous_text: bool = True,
               prompt_reset_on_temperature: float = 0.5, initial_prompt: str | list[int] | None = None,
               suppress_blank: bool = True, suppress_tokens: tuple[int, ...] | None = (-1,),
               without_timestamps: bool = False, max_initial_timestamp: float = 1.0, word_timestamps: bool = False,
               prepend_punctuations: str = "\"'“¿([{-", append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
               hallucination_silence_threshold: float | None = None, language_detection_threshold: float | None = 0.5,
               language_detection_segments: int = 1, batch_size: int = 8, kv_pool: int | None = None,
               seed: int = 0) -> list[Transcript]:
    if isinstance(temperatures, (int, float)):
        temperatures = (float(temperatures),)
    if 1 < beam_size < best_of and any(t > 0 for t in temperatures):
        raise NotImplementedError("best_of > beam_size > 1: the sampled windows decode in the beam search's rows")
    tok, c = model.tokenizer, model.cfg
    if isinstance(initial_prompt, str):
        initial = tok.encode(" " + initial_prompt.strip())
    else:
        initial = list(initial_prompt or [])
    recs = [_Recording(i, recording_features(model, a), initial) for i, a in enumerate(audio)]
    if language is None and c.multilingual:
        detect_languages(model, recs, language_detection_segments, language_detection_threshold)
    else:
        for rec in recs:
            rec.language = language if c.multilingual else "en"
    for rec in recs:
        rec.sot_sequence = tok.sot_sequence(rec.language, task)
    if recs:
        _Streamer(model, recs, task, beam_size, best_of, tuple(temperatures), compression_ratio_threshold,
                  log_prob_threshold, no_speech_threshold, condition_on_previous_text, prompt_reset_on_temperature,
                  suppress_blank, suppress_tokens, without_timestamps, int(round(max_initial_timestamp / TIME_PRECISION)),
                  word_timestamps, prepend_punctuations, append_punctuations, hallucination_silence_threshold,
                  batch_size, kv_pool, seed).run()
    return [Transcript(r.segments, r.language, r.language_probability, r.all_language_probs) for r in recs]


class _Streamer:
    """Search.serve's source for long-form transcription: each recording's windows, one after another."""

    def __init__(self, model, recs, task, beam_size, best_of, temperatures, cr_threshold, lp_threshold,
                 ns_threshold, condition_on_previous_text, prompt_reset_on_temperature, suppress_blank,
                 suppress_tokens, without_timestamps, max_initial, word_timestamps, prepend_punctuations,
                 append_punctuations, hallucination_silence_threshold, batch_size, kv_pool, seed):
        from .pipeline import compression_ratio

        self.compression_ratio = compression_ratio
        self.model, self.recs, self.task = model, recs, task
        self.temperatures, self.best_of = temperatures, best_of
        self.thresholds = (cr_threshold, lp_threshold, ns_threshold)
        self.condition_on_previous_text = condition_on_previous_text
        self.prompt_reset_on_temperature = prompt_reset_on_temperature
        self.without_timestamps, self.max_initial = without_timestamps, max_initial
        self.word_timestamps = word_timestamps
        self.punctuations = (prepend_punctuations, append_punctuations)
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self.seed = seed
        c = model.cfg
        dev, dtype = model.device, model.backend.dtype
        slots = min(batch_size, len(recs))
        if slots > model.max_batch:
            raise ValueError(f"at most {model.max_batch} windows at a time")
        if beam_size > 1:
            mode, rows = "beam", beam_size
        else:
            mode, rows = "greedy", max(1, best_of) if any(t > 0 for t in temperatures) else 1
        self.search = model._search(mode, slots, rows, None, not without_timestamps, mixed=True)
        self.cross = model.cross(slots)
        self.suppress = list(model.tokenizer.suppressed_tokens(suppress_tokens))
        self.begin = list(c.begin_suppress_tokens) if suppress_blank else []
        shape = (c.dec_layers, 2, 1, c.dec_heads, c.enc_frames, c.head_dim)
        # the pool of kept keys/values: a window holds a buffer from its first load until it is settled
        size = min(len(recs), slots + 8) if kv_pool is None else kv_pool
        self.pool = [torch.empty(shape, dtype=dtype, device=dev) for _ in range(size)]
        self.pool_free = list(range(size))
        self.pool_ready: list = [None] * size  # event after the buffer's encode
        self.pool_reads: list[list] = [[] for _ in range(size)]  # events after its reads (before a rewrite)
        self.scratch = torch.empty(shape, dtype=dtype, device=dev)  # (windows not kept)
        self.ready: deque[Job] = deque()
        self.aligning: deque = deque()  # (window, decode, split, alignment handle or result)
        self.aligner = None
        if word_timestamps:
            decoder = model._torch_decoder()
            if dev.type == "cuda" and _triton_ok():
                self.aligner = model._aligner(decoder)
            else:
                self.decoder = decoder
        self.side_scratch = torch.empty(shape, dtype=dtype, device=dev) if word_timestamps else None

    def run(self) -> None:
        for rec in self.recs:
            self._next_window(rec)
        self.search.serve(self, self.suppress, self.begin, max_initial_timestamp_index=self.max_initial,
                          prefill=self.model._prefill(self.search, self.cross))

    # --- Search.serve's source ---------------------------------------------------------------------------
    def take(self, n: int, block: bool = False) -> list[Job]:
        self._poll(wait=False)
        while block and not self.ready and self.aligning:
            self._poll(wait=True, one=True)
        out = []
        while self.ready and len(out) < n:
            out.append(self.ready.popleft())
        return out

    def load(self, slots: list[int], jobs: list[Job]) -> None:
        stream = torch.cuda.current_stream()
        for s, job in zip(slots, jobs):
            win = job.key
            if win.buffer is None and self.pool_free:  # kept from now on, for its retries and alignment
                b = win.buffer = self.pool_free.pop(0)
                for ev in self.pool_reads[b]:
                    stream.wait_event(ev)
                self.pool_reads[b] = []
                self.model.backend.encode(win.features[None], self.pool[b])
                self.pool_ready[b] = _event(stream)
                src = self.pool[b]
            elif win.buffer is not None:
                src = self.pool[win.buffer]
            else:
                self.model.backend.encode(win.features[None], self.scratch)
                src = self.scratch
            self.cross[:, :, s].copy_(src[:, :, 0])
            if win.buffer is not None:
                self.pool_reads[win.buffer].append(_event(stream))

    def result(self, job: Job, r) -> None:
        from .pipeline import _needs_fallback

        win = job.key
        text = self.model.tokenizer.decode(r.tokens).strip()
        d = {"text": text, "tokens": r.tokens, "avg_logprob": r.hypothesis.avg_logprob,
             "no_speech_prob": r.no_speech_prob, "compression_ratio": self.compression_ratio(text),
             "temperature": self.temperatures[win.ti], "reached_max_length": r.reached_max_length}
        win.attempts.append(d)
        cr, lp, ns = self.thresholds
        if not _needs_fallback(d, cr, lp, ns):
            self._settle(win, d)
        elif win.ti + 1 < len(self.temperatures):  # the next temperature, before any other window
            win.ti += 1
            self.ready.appendleft(self._job(win))
        else:  # every temperature failed: the best below the compression threshold, else any
            ok = [a for a in win.attempts if cr is None or a["compression_ratio"] <= cr]
            best = max(ok or win.attempts, key=lambda a: a["avg_logprob"])
            self._settle(win, {**best, "temperature": d["temperature"]})

    # --- a recording's windows ---------------------------------------------------------------------------
    def _job(self, win: _Window) -> Job:
        t = self.temperatures[win.ti]
        return Job(win.prompt, t if t > 0 else 0.0, mix_seed(self.seed, win.rec.index, win.ordinal, win.ti),
                   rows=self.best_of, key=win)

    def _next_window(self, rec: _Recording) -> None:
        if rec.seek >= rec.content_frames:
            return
        c = self.model.cfg
        size = min(N_FRAMES, rec.content_frames - rec.seek)
        feats = torch.zeros(c.n_mels, N_FRAMES, device=self.model.device)
        feats[:, :size] = rec.features[:, rec.seek : rec.seek + size]
        prompt = self.model.prompt(rec.language, self.task, self.without_timestamps,
                                   previous_tokens=rec.all_tokens[rec.reset_since :])
        self.ready.append(self._job(_Window(rec, feats, size, prompt)))

    def _release(self, win: _Window) -> None:
        if win.buffer is not None:
            self.pool_free.append(win.buffer)
            win.buffer = None

    def _settle(self, win: _Window, d: dict) -> None:
        """A window's final decode: skipped as silence, or its segments (after its word alignment)."""
        rec = win.rec
        cr, lp, ns = self.thresholds
        if ns is not None:  # no voice activity: on to the next window
            skip = d["no_speech_prob"] > ns
            if lp is not None and d["avg_logprob"] > lp:
                skip = False
            if skip:
                rec.seek += win.size
                self._release(win)
                self._next_window(rec)
                return
        tb = self.model.cfg.timestamp_begin
        split = split_by_timestamps(d["tokens"], tb, win.seek * TIME_PER_FRAME, win.size, win.size * TIME_PER_FRAME,
                                    win.seek)
        if not self.word_timestamps:
            self._finish(win, d, split, None)
            return
        eot = self.model.tokenizer.eot
        text = [t for seg in split[0] for t in seg["tokens"] if t < eot]
        if not text:
            self.aligning.append((win, d, split, text, ([], [])))
            return
        if self.aligner is None:  # (no CUDA graphs and Triton: synchronously)
            cross = self._cross_of(win, torch.cuda.current_stream())
            self.aligning.append((win, d, split, text, align(self.decoder, self.model.cfg, rec.sot_sequence, text,
                                                             cross[:, :, 0], win.size)))
            return
        side, main = self.model.side_stream, torch.cuda.current_stream()
        if win.buffer is not None:
            src, ready = self.pool[win.buffer], self.pool_ready[win.buffer]
        else:  # (its keys/values not kept: encoded again)
            with torch.cuda.stream(side):
                side.wait_stream(main)  # (the window's features)
                self.model.backend.encode(win.features[None], self.side_scratch, lane=1)
            src, ready = self.side_scratch, _event(side)
        handle = self.aligner.submit(src, ready, rec.sot_sequence, text, win.size)
        if win.buffer is not None:
            self.pool_reads[win.buffer].append(handle["used"])
        self.aligning.append((win, d, split, text, handle))

    def _cross_of(self, win: _Window, stream) -> torch.Tensor:
        if win.buffer is not None:
            return self.pool[win.buffer]
        self.model.backend.encode(win.features[None], self.scratch)
        return self.scratch

    def _poll(self, wait: bool, one: bool = False) -> None:
        """Finish the windows whose alignment is done (in order: they run on one stream)."""
        while self.aligning:
            win, d, split, text, handle = self.aligning[0]
            if isinstance(handle, dict):
                if not wait and not handle["done"].query():
                    return
                first, probs = Aligner.result(handle)
            else:
                first, probs = handle
            self.aligning.popleft()
            rec = win.rec
            alignment = find_alignment(self.model.tokenizer, rec.language, text, first, probs) if text else []
            self._finish(win, d, split, alignment)
            if one:
                return

    def _finish(self, win: _Window, d: dict, split: tuple, alignment: list[dict] | None) -> None:
        """faster-whisper's ``generate_segments`` after a window's decode: its segments, the next seek."""
        rec = win.rec
        current, seek, single_timestamp_ending = split
        time_offset = win.seek * TIME_PER_FRAME
        window_end_time = float((win.seek + N_FRAMES) * TIME_PER_FRAME)
        duration = win.size * TIME_PER_FRAME
        previous_seek = win.seek
        tok = self.model.tokenizer
        if self.word_timestamps:
            rec.last_speech = add_word_timestamps(current, tok, rec.language, alignment, *self.punctuations,
                                                  rec.last_speech)
            if not single_timestamp_ending:
                last_word_end = get_end(current)
                if last_word_end is not None and last_word_end > time_offset:
                    seek = round(last_word_end * FRAMES_PER_SECOND)
            if self.hallucination_silence_threshold is not None:
                seek, current, skipped = _skip_hallucinations(current, seek, previous_seek, time_offset, duration,
                                                              window_end_time, rec,
                                                              self.hallucination_silence_threshold)
                if skipped:
                    rec.seek = seek
                    self._release(win)
                    self._next_window(rec)
                    return
            last_word_end = get_end(current)
            if last_word_end is not None:
                rec.last_speech = last_word_end
        rec.seek = seek
        for seg in current:
            text = tok.decode(seg["tokens"])
            if seg["start"] == seg["end"] or not text.strip():
                continue
            rec.all_tokens.extend(seg["tokens"])
            words = [Word(w["start"], w["end"], w["word"], w["probability"]) for w in seg["words"]] \
                if self.word_timestamps else None
            rec.segments.append(Segment(len(rec.segments) + 1, previous_seek, seg["start"], seg["end"], text,
                                        seg["tokens"], d["avg_logprob"], d["compression_ratio"], d["no_speech_prob"],
                                        words, d["temperature"]))
        if not self.condition_on_previous_text or d["temperature"] > self.prompt_reset_on_temperature:
            rec.reset_since = len(rec.all_tokens)
        self._release(win)
        self._next_window(rec)


def _event(stream) -> torch.cuda.Event:
    ev = torch.cuda.Event()
    ev.record(stream)
    return ev


def _triton_ok() -> bool:
    from .alignment import triton

    return triton is not None


def _skip_hallucinations(current: list[dict], seek: int, previous_seek: int, time_offset: float, duration: float,
                         window_end_time: float, rec: _Recording, threshold: float):
    """faster-whisper's ``hallucination_silence_threshold``: silence skipped before a possible hallucination.
    Returns (seek, segments, whether the window is skipped)."""

    def word_anomaly_score(word: dict) -> float:
        probability = word.get("probability", 0.0)
        length = word["end"] - word["start"]
        score = 0.0
        if probability < 0.15:
            score += 1.0
        if length < 0.133:
            score += (0.133 - length) * 15
        if length > 2.0:
            score += length - 2.0
        return score

    def is_segment_anomaly(segment: dict | None) -> bool:
        if segment is None or not segment["words"]:
            return False
        words = [w for w in segment["words"] if w["word"] not in PUNCTUATION][:8]
        score = sum(word_anomaly_score(w) for w in words)
        return score >= 3 or score + 0.01 >= len(words)

    def next_words_segment(segments: list[dict]) -> dict | None:
        return next((s for s in segments if s["words"]), None)

    first = next_words_segment(current)
    if first is not None and is_segment_anomaly(first):
        gap = first["start"] - time_offset
        if gap > threshold:
            return previous_seek + round(gap * FRAMES_PER_SECOND), current, True
    content_duration = rec.content_frames * TIME_PER_FRAME
    hal_last_end = rec.last_speech
    for si in range(len(current)):
        segment = current[si]
        if not segment["words"]:
            continue
        if is_segment_anomaly(segment):
            nxt = next_words_segment(current[si + 1 :])
            hal_next_start = nxt["words"][0]["start"] if nxt is not None else time_offset + duration
            silence_before = (segment["start"] - hal_last_end > threshold or segment["start"] < threshold
                              or segment["start"] - time_offset < 2.0)
            silence_after = (hal_next_start - segment["end"] > threshold or is_segment_anomaly(nxt)
                             or window_end_time - segment["end"] < 2.0)
            if silence_before and silence_after:
                seek = round(max(time_offset + 1, segment["start"]) * FRAMES_PER_SECOND)
                if content_duration - segment["end"] < threshold:
                    seek = rec.content_frames
                current = current[:si]
                break
        hal_last_end = segment["end"]
    return seek, current, False
