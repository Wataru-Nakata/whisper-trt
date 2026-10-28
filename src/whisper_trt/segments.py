"""A window's tokens decoded with timestamps, split into segments as faster-whisper 1.2.1 splits them
(``WhisperModel._split_segments_by_timestamps``), with the seek of its long-form decoding."""

from .config import TIME_PRECISION

INPUT_STRIDE = 2  # feature frames per encoder frame (a timestamp step is 2 feature frames)


def split_by_timestamps(tokens: list[int], timestamp_begin: int, time_offset: float, segment_size: int,
                        segment_duration: float, seek: int) -> tuple[list[dict], int, bool]:
    """``(segments, seek, single_timestamp_ending)``: segments ``{"seek", "start", "end", "tokens"}`` (seconds):
    each pair of consecutive timestamps ends one segment and starts the next; a segment left open at the end is
    dropped (the next window starts at its timestamp), unless the window ends with a single timestamp (the next
    starts after this one, ``segment_size`` feature frames on); without consecutive timestamps, the whole window
    is one segment, up to its last timestamp (or ``segment_duration``)."""
    single_timestamp_ending = len(tokens) >= 2 and tokens[-2] < timestamp_begin <= tokens[-1]
    consecutive = [i for i in range(1, len(tokens)) if tokens[i] >= timestamp_begin and tokens[i - 1] >= timestamp_begin]
    out = []
    if consecutive:
        slices = consecutive + ([len(tokens)] if single_timestamp_ending else [])
        last_slice = 0
        for current in slices:
            sliced = tokens[last_slice:current]
            start = time_offset + (sliced[0] - timestamp_begin) * TIME_PRECISION
            end = time_offset + (sliced[-1] - timestamp_begin) * TIME_PRECISION
            out.append({"seek": seek, "start": start, "end": end, "tokens": sliced})
            last_slice = current
        if single_timestamp_ending:
            seek += segment_size
        else:
            seek += (tokens[last_slice - 1] - timestamp_begin) * INPUT_STRIDE
        return out, seek, single_timestamp_ending
    duration = segment_duration
    timestamps = [t for t in tokens if t >= timestamp_begin]
    if timestamps and timestamps[-1] != timestamp_begin:
        duration = (timestamps[-1] - timestamp_begin) * TIME_PRECISION
    out.append({"seek": seek, "start": time_offset, "end": time_offset + duration, "tokens": tokens})
    return out, seek + segment_size, single_timestamp_ending


def split_segments(tokens: list[int], timestamp_begin: int, duration: float, time_offset: float = 0.0) -> list[dict]:
    """The segments ``{"start", "end", "tokens"}`` of one window decoded on its own (``duration``: its audio in
    seconds), as faster-whisper's batched pipeline splits each window."""
    segs, _, _ = split_by_timestamps(tokens, timestamp_begin, time_offset, 0, duration, 0)
    return [{k: s[k] for k in ("start", "end", "tokens")} for s in segs]
