"""Long-form transcription as faster-whisper 1.2.1's ``WhisperModel.transcribe`` decodes a recording
(``generate_segments``), for many recordings at a time.

Each recording's features are computed over the whole recording (one normalization), and decoded window by
window from a seek position: a 30 s window of features from the seek, its prompt the recording's previous text
(``condition_on_previous_text``; reset after a decode at a temperature above ``prompt_reset_on_temperature``),
decoded with the temperature fallback; a window judged silent is skipped; its tokens split into segments by
their timestamps, the next window starting at the last timestamp (or after the window); with word timestamps,
each segment's words, the next window starting at the last word's end. A window depends on the one before it
in its recording, so each round decodes the next window of every recording not done together
(Whisper._fallback: refilled slots, prompts of any length prefilled).
"""

from dataclasses import dataclass

import torch

from .alignment import add_word_timestamps, align, find_alignment
from .audio import load_audio
from .config import HOP_LENGTH, N_FRAMES, SAMPLE_RATE, TIME_PRECISION
from .features import log_mel
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


class _Recording:
    def __init__(self, features: torch.Tensor, initial_tokens: list[int]):
        self.features = features  # [n_mels, content frames], normalized over the whole recording
        self.content_frames = features.shape[-1]
        self.seek = 0
        self.all_tokens = list(initial_tokens)
        self.reset_since = 0
        self.last_speech = 0.0
        self.segments: list[Segment] = []


def recording_features(model, audio) -> torch.Tensor:
    """faster-whisper's ``feature_extractor(audio)`` without its last frame: [n_mels, len // 160] over the whole
    recording (normalized once)."""
    wave = audio if isinstance(audio, torch.Tensor) and audio.dim() == 1 else load_audio(audio)
    frames = wave.shape[-1] // HOP_LENGTH
    return log_mel([wave], model.cfg.n_mels, model.device, max(frames, 1))[0, :, :frames]


def get_end(segments: list[dict]) -> float | None:
    """faster-whisper's ``get_end``: the last word's end, else the last segment's."""
    return next((w["end"] for s in reversed(segments) for w in reversed(s["words"])),
                segments[-1]["end"] if segments else None)


def transcribe(model, audio: list, language: str, task: str = "transcribe", beam_size: int = 5, best_of: int = 5,
               temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
               compression_ratio_threshold: float | None = 2.4, log_prob_threshold: float | None = -1.0,
               no_speech_threshold: float | None = 0.6, condition_on_previous_text: bool = True,
               prompt_reset_on_temperature: float = 0.5, initial_prompt: str | list[int] | None = None,
               suppress_blank: bool = True, suppress_tokens: tuple[int, ...] | None = (-1,),
               without_timestamps: bool = False, max_initial_timestamp: float = 1.0, word_timestamps: bool = False,
               prepend_punctuations: str = "\"'“¿([{-", append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
               hallucination_silence_threshold: float | None = None, batch_size: int = 8,
               seed: int = 0) -> list[list[Segment]]:
    tok, c = model.tokenizer, model.cfg
    if isinstance(initial_prompt, str):
        initial = tok.encode(" " + initial_prompt.strip())
    else:
        initial = list(initial_prompt or [])
    recs = [_Recording(recording_features(model, a), initial) for a in audio]
    max_initial = int(round(max_initial_timestamp / TIME_PRECISION))
    sot_sequence = tok.sot_sequence(language, task)
    decoder = model._torch_decoder() if word_timestamps else None
    rnd = 0
    while True:
        work = [r for r in recs if r.seek < r.content_frames]
        if not work:
            break
        windows = []
        for r in work:
            size = min(N_FRAMES, r.content_frames - r.seek)
            feats = torch.zeros(c.n_mels, N_FRAMES, device=model.device)
            feats[:, :size] = r.features[:, r.seek : r.seek + size]
            prompt = model.prompt(language, task, without_timestamps, previous_tokens=r.all_tokens[r.reset_since :])
            windows.append((r, feats, prompt, size))
        results = model._fallback([w[1] for w in windows], [w[2] for w in windows], beam_size, best_of,
                                  temperatures, compression_ratio_threshold, log_prob_threshold, no_speech_threshold,
                                  suppress_tokens, suppress_blank, batch_size, seed + rnd * len(temperatures),
                                  max_initial)
        rnd += 1
        kept = []
        for (r, feats, _, size), d in zip(windows, results):
            if no_speech_threshold is not None:  # no voice activity: on to the next window
                skip = d.no_speech_prob > no_speech_threshold
                if log_prob_threshold is not None and d.avg_logprob > log_prob_threshold:
                    skip = False
                if skip:
                    r.seek += size
                    continue
            time_offset = r.seek * TIME_PER_FRAME
            split = split_by_timestamps(d.tokens, c.timestamp_begin, time_offset, size, size * TIME_PER_FRAME, r.seek)
            kept.append((r, feats, size, d, time_offset, split))
        if word_timestamps:  # the alignment of each window's segments' text tokens (its encoder output once more)
            jobs = [(feats, [t for seg in split[0] for t in seg["tokens"] if t < tok.eot], size)
                    for _, feats, size, _, _, split in kept]
            alignments = _align(model, decoder, sot_sequence, language, jobs, batch_size)
        for k, (r, _, size, d, time_offset, (current, seek, single_timestamp_ending)) in enumerate(kept):
            window_end_time = float((r.seek + N_FRAMES) * TIME_PER_FRAME)
            duration = size * TIME_PER_FRAME
            previous_seek = r.seek
            if word_timestamps:
                r.last_speech = add_word_timestamps(current, tok, language, alignments[k], prepend_punctuations,
                                                    append_punctuations, r.last_speech)
                if not single_timestamp_ending:
                    last_word_end = get_end(current)
                    if last_word_end is not None and last_word_end > time_offset:
                        seek = round(last_word_end * FRAMES_PER_SECOND)
                if hallucination_silence_threshold is not None:
                    seek, current, skipped = _skip_hallucinations(current, seek, previous_seek, time_offset,
                                                                  duration, window_end_time, r,
                                                                  hallucination_silence_threshold)
                    if skipped:
                        r.seek = seek
                        continue
                last_word_end = get_end(current)
                if last_word_end is not None:
                    r.last_speech = last_word_end
            r.seek = seek
            for seg in current:
                text = tok.decode(seg["tokens"])
                if seg["start"] == seg["end"] or not text.strip():
                    continue
                r.all_tokens.extend(seg["tokens"])
                words = [Word(w["start"], w["end"], w["word"], w["probability"]) for w in seg["words"]] \
                    if word_timestamps else None
                r.segments.append(Segment(len(r.segments) + 1, previous_seek, seg["start"], seg["end"], text,
                                          seg["tokens"], d.avg_logprob, d.compression_ratio, d.no_speech_prob,
                                          words, d.temperature))
            if not condition_on_previous_text or d.temperature > prompt_reset_on_temperature:
                r.reset_since = len(r.all_tokens)
    return [r.segments for r in recs]


def _align(model, decoder, sot_sequence: list[int], language: str, jobs: list, batch_size: int) -> list[list[dict]]:
    """faster-whisper's ``find_alignment`` for windows ``(features, text tokens, feature frames)``: their encoder
    outputs (in batches), then each window's alignment."""
    out = []
    for i in range(0, len(jobs), batch_size):
        part = jobs[i : i + batch_size]
        need = [j for j, (_, text, _) in enumerate(part) if text]
        cross = model.encode(torch.stack([part[j][0] for j in need])) if need else None
        for j, (_, text, frames) in enumerate(part):
            if not text:
                out.append([])
                continue
            path, probs = align(decoder, model.cfg, sot_sequence, text, cross[:, :, need.index(j)], frames)
            out.append(find_alignment(model.tokenizer, language, text, path, probs))
    return out


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
