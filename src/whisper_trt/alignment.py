"""Word timestamps as faster-whisper 1.2.1 computes them (``WhisperModel.add_word_timestamps``,
``find_alignment``, ``merge_punctuations``) on CTranslate2 4.8.2's ``Whisper.align`` (models/whisper.cc, dtw.cc):

* the decoder runs ``sot_sequence, <|notimestamps|>, text tokens, <|endoftext|>`` against the window's encoder
  output; the probability of each text token is its softmax over the text vocabulary (tokens below EOS);
* the alignment heads' cross-attention scores are softmaxed over the window's encoder frames (half its feature
  frames), normalized over the tokens (each frame: mean 0, variance 1), median-filtered over the frames (width 7,
  reflected at the ends) and averaged over the heads; the rows of the text tokens and EOS go through a
  dynamic time warping of their negative (CTranslate2's tie-breaking);
* each word starts at the first frame its first token is aligned to and ends where the next word starts.

On CUDA, :class:`Aligner` runs it asynchronously on a stream of its own (the decoder call in a CUDA graph per
length bucket, the time warping in a Triton kernel that keeps each row's first frame): the host only reads each
row's first frame and the tokens' probabilities once they are there.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # (no Triton: the time warping on the host)
    triton = None

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


def first_frames(path: list[tuple[int, int]], rows: int) -> np.ndarray:
    """Each row's first column on a time-warping path (where faster-whisper's ``find_alignment`` jumps)."""
    out = np.zeros(rows, dtype=np.int64)
    for i, j in reversed(path):
        out[i] = j
    return out


if triton is not None:

    @triton.jit
    def _dtw_kernel(x, diag, trace, first, n, m, BI: tl.constexpr):  # fmt: skip
        """negative_dtw on one [n, m] matrix, skewed: ``x`` [n + m + 1, BI] holds cell (lo + l, d - lo - l) of
        anti-diagonal d at [d, l] (lo = max(1, d - m)); ``diag`` [3, BI] the costs of the last three diagonals
        by row; ``trace`` [n + m + 1, BI] the moves, skewed as ``x``; ``first`` [n] each row's first column."""
        lanes = tl.arange(0, BI)
        inf = float("inf")
        for d in range(2, n + m + 1):
            lo = tl.maximum(1, d - m)
            i = lanes + lo
            ok = i <= tl.minimum(n, d - 1)
            j = d - i
            # (i-1, j-1) two diagonals back, (i-1, j) and (i, j-1) one back; row 0 and column 0: inf but (0, 0)
            c0 = tl.load(diag + ((d - 2) % 3) * BI + (i - 1), mask=ok & (i > 1) & (j > 1), other=inf)
            c0 = tl.where((i == 1) & (j == 1), 0.0, c0)
            c1 = tl.load(diag + ((d - 1) % 3) * BI + (i - 1), mask=ok & (i > 1), other=inf)
            c2 = tl.load(diag + ((d - 1) % 3) * BI + i, mask=ok & (j > 1), other=inf)
            t = tl.where((c0 < c1) & (c0 < c2), 0, tl.where((c1 < c0) & (c1 < c2), 1, 2))
            c = tl.where(t == 0, c0, tl.where(t == 1, c1, c2))
            xv = tl.load(x + d * BI + lanes, mask=ok, other=0.0)
            tl.store(diag + (d % 3) * BI + i, -xv + c, mask=ok)
            tl.store(trace + d * BI + lanes, t.to(tl.int8), mask=ok)
            tl.debug_barrier()
        i = n
        j = m
        while (i > 0) | (j > 0):
            inner = (i > 0) & (j > 0)
            tl.store(first + (i - 1), j - 1, mask=inner)
            d = i + j
            t = tl.load(trace + d * BI + (i - tl.maximum(1, d - m)), mask=inner, other=0).to(tl.int32)
            t = tl.where(i == 0, 2, tl.where(j == 0, 1, t))
            i = tl.where(t != 2, i - 1, i)
            j = tl.where(t != 1, j - 1, j)


def dtw_first_frames(matrix: torch.Tensor) -> torch.Tensor:
    """:func:`negative_dtw` of ``matrix`` [n, m] (fp32, CUDA) in one Triton kernel: each row's first column on
    the path ([n] int32, on the device; the same as ``first_frames(negative_dtw(matrix))``)."""
    n, m = matrix.shape
    dev = matrix.device
    BI = triton.next_power_of_2(max(n, 16))
    d = torch.arange(n + m + 1, device=dev)[:, None]
    lo = (d - m).clamp(min=1)
    i = torch.arange(BI, device=dev)[None, :] + lo
    j = d - i
    ok = (i <= torch.minimum(torch.full_like(d, n), d - 1)) & (d >= 2)
    flat = ((i - 1) * m + (j - 1)).clamp(0, n * m - 1)
    skew = torch.where(ok, matrix.float().reshape(-1)[flat], 0.0).contiguous()
    diag = torch.empty(3, BI, dtype=torch.float32, device=dev)
    trace = torch.zeros(n + m + 1, BI, dtype=torch.int8, device=dev)
    first = torch.zeros(n, dtype=torch.int32, device=dev)
    _dtw_kernel[(1,)](skew, diag, trace, first, n, m, BI=BI, num_warps=4)
    return first


def _post(cfg, logits: torch.Tensor, scores: torch.Tensor, start: int, text_tokens: list[int], length: int,
          num_frames: int, median_filter_width: int) -> tuple[torch.Tensor | None, torch.Tensor]:
    """CTranslate2's ``Whisper.align`` after the decoder call (logits [S', V], alignment heads' scores
    [heads, S', T], the first ``length`` rows real): the matrix the time warping runs on ([text + 1, frames], None
    without frames) and each text token's probability (its softmax over the text vocabulary)."""
    dev = logits.device
    n = len(text_tokens)
    probs = logits[start : start + n, : cfg.eot].float().softmax(-1)
    text_probs = probs[torch.arange(n, device=dev), torch.tensor(text_tokens, dtype=torch.long, device=dev)]
    frames = num_frames // 2  # (the encoder's second convolution has a stride of 2)
    if frames == 0:
        return None, text_probs
    w = scores[:, :length, :frames].float().softmax(-1)  # [heads, S, frames]
    mean = w.mean(-2, keepdim=True)
    w = (w - mean) / ((w - mean) ** 2).mean(-2, keepdim=True).sqrt()  # (CTranslate2's LayerNorm, epsilon 0)
    w = median_filter(w, median_filter_width).mean(0)
    return w[start : start + n + 1], text_probs


@torch.inference_mode()
def align(decoder, cfg, sot_sequence: list[int], text_tokens: list[int], cross: torch.Tensor, num_frames: int,
          median_filter_width: int = 7) -> tuple[np.ndarray, list[float]]:
    """CTranslate2's ``Whisper.align`` for one window, synchronously: (each text row's first frame on the
    alignment path, with EOS's; the probability of each text token). ``decoder``: a modeling.Decoder; ``cross``:
    the window's keys/values [L, 2, H, T, d_k]; ``num_frames``: its feature frames."""
    tokens = list(sot_sequence) + [cfg.no_timestamps] + list(text_tokens) + [cfg.eot]
    dev = cross.device
    logits, scores = decoder(torch.tensor([tokens], device=dev), cross[:, :, None], heads=cfg.alignment_heads)
    matrix, text_probs = _post(cfg, logits[0], scores[0], len(sot_sequence), text_tokens, len(tokens), num_frames,
                               median_filter_width)
    if matrix is None:
        return np.zeros(0, dtype=np.int64), text_probs.tolist()
    m = matrix.cpu().numpy()
    return first_frames(negative_dtw(m), m.shape[0]), text_probs.tolist()


class Aligner:
    """:func:`align` for windows queued on ``stream`` (CUDA, with Triton): :meth:`submit` returns at once; the
    host reads a window's result once its event is done. The decoder call runs in a CUDA graph per length bucket
    (tokens padded at the end: the decoder is causal, the rows before are the same) against a copy of the
    window's keys/values (the window's buffer can be reused once ``used`` is done)."""

    BUCKETS = (32, 64, 96, 128, 160, 192, 256)

    def __init__(self, decoder, cfg, device: torch.device, stream: torch.cuda.Stream,
                 median_filter_width: int = 7):
        self.decoder, self.cfg, self.device, self.stream = decoder, cfg, device, stream
        self.width = median_filter_width
        dtype = next(decoder.parameters()).dtype
        c = cfg
        self.cross = torch.zeros(c.dec_layers, 2, 1, c.dec_heads, c.enc_frames, c.head_dim, dtype=dtype, device=device)
        self.tokens = torch.zeros(1, self.BUCKETS[-1], dtype=torch.long, device=device)
        self.graphs: dict[int, tuple] = {}
        self.mempool = torch.cuda.graph_pool_handle()

    def prepare(self) -> None:
        """Capture every bucket's graph (a capture synchronizes the device: before any decode)."""
        with torch.cuda.stream(self.stream):
            for b in self.BUCKETS:
                self._forward(b, replay=False)

    def _forward(self, bucket: int, replay: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if bucket not in self.graphs:
            tokens = self.tokens[:, :bucket]

            def run():
                return self.decoder(tokens, self.cross, heads=self.cfg.alignment_heads)

            with torch.inference_mode():
                for _ in range(2):  # (warm-up: cuBLAS handles and workspaces, outside the graph)
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.mempool, stream=self.stream):
                    logits, scores = run()
            self.graphs[bucket] = (graph, logits, scores)
        graph, logits, scores = self.graphs[bucket]
        if replay:
            graph.replay()
        return logits, scores

    def submit(self, cross: torch.Tensor, ready: "torch.cuda.Event | None", sot_sequence: list[int],
               text_tokens: list[int], num_frames: int) -> dict:
        """Queue one window's alignment: ``cross`` its keys/values ([L, 2, 1, H, T, d_k] or [L, 2, H, T, d_k]),
        written before ``ready``. Returns a handle: ``used`` (``cross`` read), ``done`` (results copied)."""
        tokens = list(sot_sequence) + [self.cfg.no_timestamps] + list(text_tokens) + [self.cfg.eot]
        bucket = next(b for b in self.BUCKETS if b >= len(tokens))
        with torch.cuda.stream(self.stream), torch.inference_mode():
            if ready is not None:
                self.stream.wait_event(ready)
            self.cross.copy_(cross.view(self.cross.shape))
            used = torch.cuda.Event()
            used.record(self.stream)
            ids = torch.tensor(tokens, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)
            self.tokens[0, : len(tokens)].copy_(ids)
            logits, scores = self._forward(bucket)
            matrix, text_probs = _post(self.cfg, logits[0], scores[0], len(sot_sequence), text_tokens, len(tokens),
                                       num_frames, self.width)
            probs_h = torch.empty(len(text_tokens), dtype=torch.float32, pin_memory=True)
            probs_h.copy_(text_probs, non_blocking=True)
            first_h = torch.zeros(0 if matrix is None else matrix.shape[0], dtype=torch.int32, pin_memory=True)
            if matrix is not None:
                first_h.copy_(dtw_first_frames(matrix.contiguous()), non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        return {"used": used, "done": done, "first": first_h, "probs": probs_h}

    @staticmethod
    def result(handle: dict) -> tuple[np.ndarray, list[float]]:
        """A submitted window's (first frames, text token probabilities), waiting for them if need be."""
        handle["done"].synchronize()
        return handle["first"].numpy().astype(np.int64), handle["probs"].tolist()


def find_alignment(tokenizer, language: str, text_tokens: list[int], jump_frames, text_probs: list[float]) -> list[dict]:
    """faster-whisper's ``find_alignment`` for one window from each row's first frame on the alignment path
    (``jump_frames``: the text tokens' and EOS's, :func:`first_frames` or :func:`dtw_first_frames`; empty: no
    frames): words ``{"word", "tokens", "start", "end", "probability"}`` (seconds from the window's start)."""
    if len(jump_frames) == 0:
        return []
    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot], language)
    if len(word_tokens) <= 1:
        return []
    boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    if len(boundaries) <= 1:
        return []
    # (the path visits every row: its jumps are the rows' first frames)
    jump_times = np.asarray(jump_frames) / TOKENS_PER_SECOND
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
