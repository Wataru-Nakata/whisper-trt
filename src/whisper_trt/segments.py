"""A window's tokens decoded with timestamps, split into segments as faster-whisper 1.2.1 splits them
(``WhisperModel._split_segments_by_timestamps``, which its batched pipeline applies to each window)."""

from .config import TIME_PRECISION


def split_segments(tokens: list[int], timestamp_begin: int, duration: float, time_offset: float = 0.0) -> list[dict]:
    """Segments ``{"start", "end", "tokens"}`` (seconds from ``time_offset``) of one window's generated tokens:
    each pair of consecutive timestamps ends one segment and starts the next; a segment left open at the end is
    dropped, unless the window ends with a single timestamp; without consecutive timestamps, the whole window is
    one segment, up to its last timestamp (or ``duration``, the window's audio length in seconds)."""
    single_timestamp_ending = len(tokens) >= 2 and tokens[-2] < timestamp_begin <= tokens[-1]
    consecutive = [i for i in range(1, len(tokens)) if tokens[i] >= timestamp_begin and tokens[i - 1] >= timestamp_begin]
    if consecutive:
        slices = consecutive + ([len(tokens)] if single_timestamp_ending else [])
        out, last_slice = [], 0
        for current in slices:
            sliced = tokens[last_slice:current]
            start = time_offset + (sliced[0] - timestamp_begin) * TIME_PRECISION
            end = time_offset + (sliced[-1] - timestamp_begin) * TIME_PRECISION
            out.append({"start": start, "end": end, "tokens": sliced})
            last_slice = current
        return out
    timestamps = [t for t in tokens if t >= timestamp_begin]
    if timestamps and timestamps[-1] != timestamp_begin:
        duration = (timestamps[-1] - timestamp_begin) * TIME_PRECISION
    return [{"start": time_offset, "end": time_offset + duration, "tokens": tokens}]
