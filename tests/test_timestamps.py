"""Decoding with timestamps: search.Search against CTranslate2 4.8.2's BeamSearch / GreedySearch loops with its
ApplyTimestampRules logits processor (models/whisper.cc), transcribed line by line, on a toy vocabulary with
timestamp tokens; and faster-whisper's segments of a window's tokens.

The fake decoder (as in test_search.py) gives logits that are a pseudo-random function of a row's whole token
history, read back from the KV cache through the slot table.
"""

import math
import zlib

import pytest
import torch

from whisper_trt import beam_kernels
from whisper_trt.config import WhisperConfig
from whisper_trt.search import Search
from whisper_trt.segments import split_segments

V, EOT, SOT, NOSPEECH, NOTS, TSB = 140, 1, 2, 3, 40, 41  # text tokens 4..39, timestamps 41..139
LOWEST = torch.finfo(torch.float32).min


def make_cfg() -> WhisperConfig:
    return WhisperConfig(n_mels=4, d_model=4, enc_layers=1, enc_heads=1, enc_ffn=4, dec_layers=1, dec_heads=1,
                         dec_ffn=4, vocab_size=V, eot=EOT, sot=SOT, no_speech=NOSPEECH, no_timestamps=NOTS,
                         timestamp_begin=TSB)


def scores_of(history: list[int], seed: int) -> torch.Tensor:
    """Random logits; timestamps a little after the latest one likely (every rule runs, and there are always
    2K tokens the rules allow: no ties among disabled tokens decide a beam, which neither CTranslate2 nor
    torch.topk orders)."""
    g = torch.Generator().manual_seed(zlib.crc32(repr(("ts", seed, history)).encode()))
    x = torch.randn(V, generator=g, dtype=torch.float64) * 2.5
    x[EOT] += 0.15 * len(history)  # sequences end eventually
    latest = max([t - TSB for t in history if t >= TSB], default=-1)
    near = torch.arange(V - TSB)
    x[TSB:] += torch.where((near > latest) & (near <= latest + 6), 2.0, -1.0).double()
    return x


def fake_step(seed: int):
    def step_fn(tokens, offset, cache, logits, new_kv, src):
        def run():
            for n in range(tokens.shape[0]):
                off = int(offset[n])
                hist = [int(round(float(cache[0, 0, int(src[n, j]), 0, j, 0]))) for j in range(off)]
                for s in range(tokens.shape[1]):
                    tok = int(tokens[n, s])
                    logits[n, s] = scores_of(hist + [tok], seed).float()
                    new_kv[0, :, n, 0, s, :] = float(tok)
                    hist.append(tok)

        return run

    return step_fn


def apply_timestamp_rules(x: torch.Tensor, generated: list[int], max_initial: int) -> torch.Tensor:
    """ApplyTimestampRules::apply for one row (sample_begin 0: Whisper's decoder starts at its last prompt token)."""
    x = x.clone()
    x[NOTS] = LOWEST  # Suppress <|notimestamps|>.
    step = len(generated)
    check = False
    if step == 0:
        x[:TSB] = LOWEST  # Suppress non timestamps at the beginning.
        x[TSB + max_initial + 1 :] = LOWEST  # Apply max_initial_timestamp option.
    else:
        last = generated[-1]
        if last >= TSB:
            penultimate = generated[-2] if step - 1 > 0 else last
            if penultimate >= TSB:  # has to be non-timestamp
                x[TSB:] = LOWEST
            else:  # cannot be normal text tokens
                x[:EOT] = LOWEST
                x[TSB:last] = LOWEST
                check = True
        else:
            check = True
            for token in reversed(generated):  # timestamps shouldn't decrease
                if token >= TSB:
                    x[TSB : token + 1] = LOWEST
                    break
    if check:
        lp = x.log_softmax(-1)
        if float(lp[TSB:].logsumexp(-1)) > float(lp[:TSB].max()):
            x[:TSB] = LOWEST
    return x


def masked_logp(prompt, generated, seed, suppress, begin, max_initial) -> torch.Tensor:
    x = scores_of(prompt + generated, seed).float()
    x[list(suppress)] = LOWEST  # SuppressTokens, then SuppressTokensBegin, then the timestamp rules
    if not generated:
        x[list(begin)] = LOWEST
    return apply_timestamp_rules(x, generated, max_initial).log_softmax(-1).double()


def ct2_beam(prompt, K, max_new, seed, suppress, begin, max_initial):
    alive = [([], 0.0)] + [([], -math.inf)] * (K - 1)
    hyps = []
    for step in range(max_new):
        cands = []
        for k, (toks, sc) in enumerate(alive):
            lp = masked_logp(prompt, toks, seed, suppress, begin, max_initial)
            cands += [(sc + float(lp[v]), k * V + v) for v in range(V)]
        cands.sort(key=lambda c: -c[0])
        cands = cands[: 2 * K]
        new_alive, sec = [], K
        last = step == max_new - 1
        for k in range(K):
            s, idx = cands[k]
            par, v = divmod(idx, V)
            nxt = k
            if v == EOT or last:
                hyps.append((alive[par][0] + ([] if v == EOT else [v]), s))
                for j in range(sec, 2 * K):
                    if cands[j][1] % V != EOT:
                        nxt, sec = j, j + 1
                        break
            s2, idx2 = cands[nxt]
            p2, v2 = divmod(idx2, V)
            new_alive.append((alive[p2][0] + [v2], s2))
        if last or len(hyps) >= K:
            break
        alive = new_alive
    return max(hyps, key=lambda h: h[1] / len(h[0]) if h[0] else -math.inf)


def ct2_greedy(prompt, max_new, seed, suppress, begin, max_initial):
    toks, score = [], 0.0
    for _ in range(max_new):
        lp = masked_logp(prompt, toks, seed, suppress, begin, max_initial)
        v = int(lp.argmax())
        score += float(lp[v])
        if v == EOT:
            break
        toks.append(v)
    return toks, score


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("mode,K", [("beam", 3), ("beam", 5), ("greedy", 1)])
@pytest.mark.parametrize("max_initial", [9, 50])
def test_timestamps_match_ct2(seed, mode, K, max_initial):
    cfg = make_cfg()
    P, M = 3, 7  # <|startoftranscript|>, language, task: no <|notimestamps|>
    suppress, begin = [5, 6, 7], [8, EOT]
    prompts = [[SOT, 10 + w, 20 + w % 3] for w in range(M)]
    search = Search(cfg, fake_step(seed), 3, K, P, mode, dtype=torch.float32, device="cpu", max_length=60,
                    graphs=False, timestamps=True)
    results = search.stream(prompts, lambda slots, windows: None, suppress, begin, check_every=1, ahead=1,
                            max_initial_timestamp_index=max_initial)
    seen_pairs = False
    for w, prompt in enumerate(prompts):
        if mode == "beam":
            toks, score = ct2_beam(prompt, K, search.max_new, seed, suppress, begin, max_initial)
            assert results[w].tokens == toks, w
            assert results[w].hypothesis.normalized == pytest.approx(score / len(toks), abs=1e-4)
        else:
            toks, score = ct2_greedy(prompt, search.max_new, seed, suppress, begin, max_initial)
            assert results[w].tokens == toks, w
            assert results[w].hypothesis.score == pytest.approx(score, abs=1e-4)
        assert toks[0] >= TSB and toks[0] - TSB <= max_initial  # a timestamp first
        seen_pairs |= any(a >= TSB and b >= TSB for a, b in zip(toks, toks[1:]))
    assert seen_pairs  # (the rules for pairs ran)


cuda = pytest.mark.skipif(not torch.cuda.is_available() or beam_kernels.triton is None, reason="needs CUDA + Triton")


@cuda
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("K", [3, 5])
def test_fused_timestamps_match_ct2(seed, K):
    """The fused beam bookkeeping (beam_kernels) with the timestamp rules: CTranslate2's results, as the PyTorch
    rules give them."""
    cfg = make_cfg()
    P, M, max_initial = 3, 7, 9
    suppress, begin = [5, 6, 7], [8, EOT]
    prompts = [[SOT, 10 + w, 20 + w % 3] for w in range(M)]
    runs = {}
    for fused in (True, False):
        search = Search(cfg, fake_step(seed), 3, K, P, "beam", dtype=torch.float32, device="cuda", max_length=60,
                        graphs=False, fused=fused, timestamps=True)
        runs[fused] = search.stream(prompts, lambda slots, windows: None, suppress, begin, check_every=1, ahead=1,
                                    max_initial_timestamp_index=max_initial)
    for w, prompt in enumerate(prompts):
        toks, score = ct2_beam(prompt, K, search.max_new, seed, suppress, begin, max_initial)
        for fused, results in runs.items():
            assert results[w].tokens == toks, (fused, w)
            assert results[w].hypothesis.normalized == pytest.approx(score / len(toks), abs=1e-4)
        assert [h.tokens for h in runs[True][w].hypotheses] == [h.tokens for h in runs[False][w].hypotheses]


def test_split_segments_like_faster_whisper():
    tsb = 50364
    ts = lambda s: tsb + round(s / 0.02)  # noqa: E731
    # two closed segments, then an unfinished one: faster-whisper keeps the closed ones
    toks = [ts(0), 100, 101, ts(2.0), ts(2.0), 102, ts(3.5), ts(4.0), 103]
    segs = split_segments(toks, tsb, duration=10.0)
    assert [s["tokens"] for s in segs] == [[ts(0), 100, 101, ts(2.0)], [ts(2.0), 102, ts(3.5)]]
    assert [(s["start"], s["end"]) for s in segs] == pytest.approx([(0.0, 2.0), (2.0, 3.5)])
    # a single timestamp at the end closes the last segment
    toks = [ts(0), 100, ts(1.0), ts(1.0), 101, ts(2.5)]
    segs = split_segments(toks, tsb, duration=10.0)
    assert [(s["start"], s["end"]) for s in segs] == pytest.approx([(0.0, 1.0), (1.0, 2.5)])
    # no consecutive timestamps: one segment to the last timestamp (or the window's duration)
    one = split_segments([ts(0), 100, 101, ts(1.5)], tsb, 10.0)
    assert [(s["start"], s["end"]) for s in one] == pytest.approx([(0.0, 1.5)])
    assert [(s["start"], s["end"]) for s in split_segments([ts(0), 100, 101], tsb, 7.25)] == [(0.0, 7.25)]
