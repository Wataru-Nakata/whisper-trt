"""alignment: the dynamic time warping (on the host, and in a Triton kernel) and median filter against line-by-line
transcriptions of CTranslate2 4.8.2's (dtw.cc, ops/median_filter_cpu.cc), ties included."""

import numpy as np
import pytest
import torch

from whisper_trt.alignment import dtw_first_frames, first_frames, median_filter, negative_dtw, triton


def ct2_negative_dtw(x: np.ndarray) -> list[tuple[int, int]]:
    inf = float("inf")
    n, m = x.shape
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    trace = [[-1] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            c0, c1, c2 = cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i][j] = float(np.float32(-x[i - 1, j - 1]) + np.float32(c))
            trace[i][j] = t
    for k in range(m + 1):
        trace[0][k] = 2
    for k in range(n + 1):
        trace[k][0] = 1
    i, j, out = n, m, []
    while i > 0 or j > 0:
        out.append((i - 1, j - 1))
        t = trace[i][j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def ct2_median(row: list[float], width: int) -> list[float]:
    depth, rank = len(row), width // 2
    if depth <= rank:
        return list(row)
    out = []
    for j in range(depth):
        window = []
        for k in range(-rank, rank + 1):
            read = abs(j + k)
            if read >= depth:
                read = depth - (read - depth) - 2
            window.append(row[read])
        out.append(sorted(window)[rank])
    return out


@pytest.mark.parametrize("seed", range(20))
def test_negative_dtw_matches_ctranslate2(seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(1, 12), rng.integers(1, 40)
    # small integers: many ties between the three moves
    x = (rng.integers(-2, 3, size=(n, m)) if seed % 2 else rng.normal(size=(n, m))).astype(np.float32)
    assert negative_dtw(x) == ct2_negative_dtw(x)


@pytest.mark.parametrize("depth", [1, 3, 4, 7, 30])
def test_median_filter_matches_ctranslate2(depth):
    g = torch.Generator().manual_seed(depth)
    x = torch.randn(2, 3, depth, generator=g)
    got = median_filter(x, 7)
    for a in range(2):
        for b in range(3):
            assert got[a, b].tolist() == pytest.approx(ct2_median(x[a, b].tolist(), 7))


@pytest.mark.skipif(not torch.cuda.is_available() or triton is None, reason="needs CUDA and Triton")
@pytest.mark.parametrize("seed", range(12))
def test_dtw_kernel_first_frames_match_ctranslate2(seed):
    """The time warping in one Triton kernel: each row's first column on CTranslate2's path, ties included."""
    rng = np.random.default_rng(100 + seed)
    n, m = int(rng.integers(1, 230)), int(rng.integers(1, 1500))
    if seed < 4:
        n, m = int(rng.integers(1, 6)), int(rng.integers(1, 12))
    x = (rng.integers(-2, 3, size=(n, m)) if seed % 2 else rng.normal(size=(n, m))).astype(np.float32)
    got = dtw_first_frames(torch.from_numpy(x).cuda()).cpu().numpy()
    want = first_frames(ct2_negative_dtw(x) if n * m < 5000 else negative_dtw(x), n)
    assert got.tolist() == want.tolist()
