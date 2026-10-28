"""Word timestamps as faster-whisper 1.2.1 computes them (``WhisperModel.add_word_timestamps``,
``find_alignment``, ``merge_punctuations``) on CTranslate2 4.8.2's ``Whisper.align`` (models/whisper.cc, dtw.cc):

* the decoder runs ``sot_sequence, <|notimestamps|>, text tokens, <|endoftext|>`` against the window's encoder
  output; the probability of each text token is its softmax over the text vocabulary (tokens below EOS);
* the alignment heads' cross-attention scores are softmaxed over the window's encoder frames (half its feature
  frames), normalized over the tokens (each frame: mean 0, variance 1), median-filtered over the frames (width 7,
  reflected at the ends) and averaged over the heads; the rows of the text tokens and EOS go through a
  dynamic time warping of their negative (CTranslate2's tie-breaking);
* each word starts at the first frame its first token is aligned to and ends where the next word starts.
"""

import numpy as np
import torch
import torch.nn.functional as F

TOKENS_PER_SECOND = 50  # encoder frames per second (feature frames / the encoder's stride of 2)


def median_filter(x: torch.Tensor, width: int) -> torch.Tensor:
    """CTranslate2's MedianFilter over the last axis, reflected at the ends (no edge repeated)."""
    rank = width // 2
    shape = x.shape
    if shape[-1] <= rank:
        return x
    padded = F.pad(x.reshape(-1, 1, shape[-1]), (rank, rank), mode="reflect").reshape(*shape[:-1], -1)
    return padded.unfold(-1, width, 1).median(-1).values


def negative_dtw(x: np.ndarray) -> list[tuple[int, int]]:
    """CTranslate2's negative_dtw: the path minimizing the sum of -x from (0, 0) to (n-1, m-1), as (row, column)
    pairs; a diagonal step when strictly cheapest, else a step down (row + 1) when strictly cheapest, else right.
    Computed along anti-diagonals (each one depends on the two before)."""
    n, m = x.shape
    x = x.astype(np.float32)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    trace = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cost[0, 0] = 0
    for diag in range(2, n + m + 1):
        i = np.arange(max(1, diag - m), min(n, diag - 1) + 1)
        j = diag - i
        c0, c1, c2 = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
        t = np.where((c0 < c1) & (c0 < c2), 0, np.where((c1 < c0) & (c1 < c2), 1, 2)).astype(np.int8)
        c = np.where(t == 0, c0, np.where(t == 1, c1, c2))
        cost[i, j] = -x[i - 1, j - 1] + c
        trace[i, j] = t
    trace[0, :] = 2
    trace[:, 0] = 1
    i, j, path = n, m, []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        t = trace[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


@torch.inference_mode()
def align(decoder, cfg, sot_sequence: list[int], text_tokens: list[int], cross: torch.Tensor, num_frames: int,
          median_filter_width: int = 7) -> tuple[list[tuple[int, int]], list[float]]:
    """CTranslate2's ``Whisper.align`` for one window: (alignment path, probability of each text token).
    ``decoder``: a modeling.Decoder; ``cross``: the window's keys/values [L, 2, H, T, d_k]; ``num_frames``: its
    feature frames."""
    tokens = list(sot_sequence) + [cfg.no_timestamps] + list(text_tokens) + [cfg.eot]
    dev = cross.device
    logits, scores = decoder(torch.tensor([tokens], device=dev), cross[:, :, None], heads=cfg.alignment_heads)
    probs = logits[0, :, : cfg.eot].float().softmax(-1)
    start = len(sot_sequence)
    index = torch.arange(len(text_tokens), device=dev)
    text_probs = probs[start + index, torch.tensor(text_tokens, dtype=torch.long, device=dev)].tolist()
    frames = num_frames // 2  # (the encoder's second convolution has a stride of 2)
    if frames == 0:
        return [], text_probs
    w = scores[0, :, :, :frames].float().softmax(-1)  # [heads, S, frames]
    mean = w.mean(-2, keepdim=True)
    w = (w - mean) / ((w - mean) ** 2).mean(-2, keepdim=True).sqrt()  # (CTranslate2's LayerNorm, epsilon 0)
    w = median_filter(w, median_filter_width).mean(0)
    matrix = w[start : start + len(text_tokens) + 1].cpu().numpy()
    return negative_dtw(matrix), text_probs


def find_alignment(tokenizer, language: str, text_tokens: list[int], path: list[tuple[int, int]],
                   text_probs: list[float]) -> list[dict]:
    """faster-whisper's ``find_alignment`` for one window: words ``{"word", "tokens", "start", "end",
    "probability"}`` (seconds from the window's start)."""
    if not path:
        return []
    text_indices = np.array([p[0] for p in path])
    time_indices = np.array([p[1] for p in path])
    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot], language)
    if len(word_tokens) <= 1:
        return []
    boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    if len(boundaries) <= 1:
        return []
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    starts, ends = jump_times[boundaries[:-1]], jump_times[boundaries[1:]]
    probabilities = [np.mean(text_probs[i:j]) for i, j in zip(boundaries[:-1], boundaries[1:])]
    return [dict(word=w, tokens=t, start=s, end=e, probability=p)
            for w, t, s, e, p in zip(words, word_tokens, starts, ends, probabilities)]


def merge_punctuations(alignment: list[dict], prepended: str, appended: str) -> None:
    """faster-whisper's ``merge_punctuations``: opening marks join the next word, closing ones the previous."""
    i, j = len(alignment) - 2, len(alignment) - 1
    while i >= 0:
        previous, following = alignment[i], alignment[j]
        if previous["word"].startswith(" ") and previous["word"].strip() in prepended:
            following["word"] = previous["word"] + following["word"]
            following["tokens"] = previous["tokens"] + following["tokens"]
            previous["word"], previous["tokens"] = "", []
        else:
            j = i
        i -= 1
    i, j = 0, 1
    while j < len(alignment):
        previous, following = alignment[i], alignment[j]
        if not previous["word"].endswith(" ") and following["word"] in appended:
            previous["word"] = previous["word"] + following["word"]
            previous["tokens"] = previous["tokens"] + following["tokens"]
            following["word"], following["tokens"] = "", []
        else:
            i = j
        j += 1


def add_word_timestamps(segments: list[dict], tokenizer, language: str, alignment: list[dict],
                        prepend_punctuations: str, append_punctuations: str, last_speech_timestamp: float,
                        frames_per_second: int = 100) -> float:
    """faster-whisper's ``add_word_timestamps`` for one window's ``segments`` (its ``alignment`` from
    :func:`find_alignment` of their text tokens): each segment's ``words``, with its start and end moved to its
    words' where they agree; returns the last speech timestamp."""
    word_durations = np.array([w["end"] - w["start"] for w in alignment])
    word_durations = word_durations[word_durations.nonzero()]
    median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    median_duration = min(0.7, float(median_duration))
    max_duration = median_duration * 2
    if len(word_durations) > 0:  # words at sentence boundaries: at most twice the median duration
        sentence_end_marks = ".。!！?？"
        for i in range(1, len(alignment)):
            if alignment[i]["end"] - alignment[i]["start"] > max_duration:
                if alignment[i]["word"] in sentence_end_marks:
                    alignment[i]["end"] = alignment[i]["start"] + max_duration
                elif alignment[i - 1]["word"] in sentence_end_marks:
                    alignment[i]["start"] = alignment[i]["end"] - max_duration
    merge_punctuations(alignment, prepend_punctuations, append_punctuations)
    text_tokens = [[t for t in seg["tokens"] if t < tokenizer.eot] for seg in segments]
    word_index = 0
    time_offset = segments[0]["seek"] / frames_per_second
    for seg, seg_tokens in zip(segments, text_tokens):
        saved_tokens = 0
        words = []
        while word_index < len(alignment) and saved_tokens < len(seg_tokens):
            timing = alignment[word_index]
            if timing["word"]:
                words.append(dict(word=timing["word"], start=round(time_offset + timing["start"], 2),
                                  end=round(time_offset + timing["end"], 2), probability=timing["probability"]))
            saved_tokens += len(timing["tokens"])
            word_index += 1
        if words:
            # the first and second word after a pause: at most twice the median word duration
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                words[0]["end"] - words[0]["start"] > max_duration
                or (len(words) > 1 and words[1]["end"] - words[0]["start"] > max_duration * 2)
            ):
                if len(words) > 1 and words[1]["end"] - words[1]["start"] > max_duration:
                    boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                    words[0]["end"] = words[1]["start"] = boundary
                words[0]["start"] = max(0, words[0]["end"] - max_duration)
            # the segment's start if the first word is too long, else the first word's
            if seg["start"] < words[0]["end"] and seg["start"] - 0.5 > words[0]["start"]:
                words[0]["start"] = max(0, min(words[0]["end"] - median_duration, seg["start"]))
            else:
                seg["start"] = words[0]["start"]
            # the segment's end if the last word is too long, else the last word's
            if seg["end"] > words[-1]["start"] and seg["end"] + 0.5 < words[-1]["end"]:
                words[-1]["end"] = max(words[-1]["start"] + median_duration, seg["end"])
            else:
                seg["end"] = words[-1]["end"]
            last_speech_timestamp = seg["end"]
        seg["words"] = words
    return last_speech_timestamp
