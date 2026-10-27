"""Log-Mel features on the GPU, as faster-whisper 1.2.1 computes them (``FeatureExtractor``, numpy).

For a waveform of ``L`` samples faster-whisper appends 160 zero samples, takes a centered (reflect-padded)
400-point STFT with a periodic Hann window every 160 samples, drops the last STFT frame, and normalizes
``log10(max(mel, 1e-10))`` to at least its maximum minus 8, then ``(x + 4) / 4``. The maximum is taken over
these ``L // 160 + 1`` frames; the decoder sees the first ``L // 160`` of them, padded with zeros (in the
normalized domain) to 3000 frames.

Batches: every row starts at sample 0, so the left reflection is that of its own samples; on the right the
frames read at most 40 samples past ``L + 160``, which faster-whisper reflects from its 160 padding zeros, so
zeros there give the same frames.
"""

from functools import lru_cache

import numpy as np
import torch

from .config import HOP_LENGTH, N_FFT, N_FRAMES, SAMPLE_RATE


def mel_filters_np(n_mels: int, sr: int = SAMPLE_RATE, n_fft: int = N_FFT) -> np.ndarray:
    """Slaney-style mel filterbank [n_mels, n_fft // 2 + 1] (faster-whisper's ``get_mel_filters``, float32)."""
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mels = np.linspace(0.0, 45.245640471924965, n_mels + 2)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    fdiff = np.diff(freqs)
    ramps = freqs.reshape(-1, 1) - fftfreqs.reshape(1, -1)
    lower = -ramps[:-2] / np.expand_dims(fdiff[:-1], axis=1)
    upper = ramps[2:] / np.expand_dims(fdiff[1:], axis=1)
    weights = np.maximum(np.zeros_like(lower), np.minimum(lower, upper))
    enorm = 2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels])
    weights *= np.expand_dims(enorm, axis=1)
    return weights.astype(np.float32)


@lru_cache(maxsize=8)
def _constants(n_mels: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.from_numpy(np.hanning(N_FFT + 1)[:-1].astype(np.float32)).to(device)
    return window, torch.from_numpy(mel_filters_np(n_mels)).to(device)


def content_frames(samples: int) -> int:
    """Feature frames the decoder sees for ``samples`` samples (before padding to 3000)."""
    return samples // HOP_LENGTH


def log_mel(
    waves: list[torch.Tensor], n_mels: int, device: str | torch.device = "cuda", frames: int = N_FRAMES
) -> torch.Tensor:
    """Features [B, n_mels, frames] (fp32) of 1-D waveforms (16 kHz), each normalized on its own and
    zero-padded/trimmed to ``frames`` as faster-whisper's ``pad_or_trim``."""
    device = torch.device(device)
    window, filters = _constants(n_mels, str(device))
    lengths = [int(w.shape[-1]) for w in waves]
    n = max(lengths) + HOP_LENGTH
    pad = N_FFT // 2
    batch = torch.zeros(len(waves), n + 2 * pad, device=device)
    for i, w in enumerate(waves):
        w = torch.as_tensor(w, dtype=torch.float32).to(device, non_blocking=True)
        length = w.shape[-1] + HOP_LENGTH  # with faster-whisper's 160 padding zeros
        row = torch.zeros(length, device=device)
        row[: w.shape[-1]] = w
        # centered STFT: reflect padding (the right side only ever reflects padding zeros, see the module doc)
        batch[i, pad : pad + length] = row
        batch[i, :pad] = row[1 : pad + 1].flip(0)
    n_stft = 1 + (n + 2 * pad - N_FFT) // HOP_LENGTH
    spec = torch.stft(
        batch, N_FFT, HOP_LENGTH, window=window, center=False, return_complex=True
    )  # [B, 201, n_stft]
    mags = spec[..., : n_stft - 1].abs() ** 2
    mel = filters @ mags  # [B, n_mels, n_stft - 1]
    log_spec = torch.clamp(mel, min=1e-10).log10()
    out = torch.zeros(len(waves), n_mels, frames, device=device)
    for i, length in enumerate(lengths):
        used = length // HOP_LENGTH + 1  # faster-whisper's frames: the maximum is over all of them
        x = log_spec[i, :, :used]
        x = torch.maximum(x, x.max() - 8.0)
        x = (x + 4.0) / 4.0
        keep = min(used - 1, frames)
        out[i, :, :keep] = x[:, :keep]
    return out
