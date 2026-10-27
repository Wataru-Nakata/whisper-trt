"""search.Search against a direct transcription of CTranslate2 4.8.2's BeamSearch / GreedySearch loops.

The fake decoder's logits are a pseudo-random function of a row's whole token history, which it reads back
from the KV cache through the slot table (each slot holds its token id): a wrong table, cache write or offset
changes the output.
"""

import math
import zlib

import pytest
import torch

from whisper_trt.config import WhisperConfig
from whisper_trt.search import Search

V, EOT, SOT, NOSPEECH = 40, 1, 2, 3
LOWEST = torch.finfo(torch.float32).min


def make_cfg() -> WhisperConfig:
    return WhisperConfig(n_mels=4, d_model=4, enc_layers=1, enc_heads=1, enc_ffn=4, dec_layers=1, dec_heads=1,
                         dec_ffn=4, vocab_size=V, eot=EOT, sot=SOT, no_speech=NOSPEECH)


def scores_of(history: list[int], seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(zlib.crc32(repr((seed, history)).encode()))
    x = torch.randn(V, generator=g, dtype=torch.float64) * 2.5
    x[EOT] += 0.15 * len(history)  # sequences end eventually
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


def fake_prefill(seed: int):
    def prefill(slots, tokens, cache, src):
        k, p = tokens.shape
        out = torch.zeros(k, p, V)
        for i in range(k):
            toks = tokens[i].tolist()
            for j in range(p):
                out[i, j] = scores_of(toks[: j + 1], seed).float()
            cache[0, :, i, 0, :p, :] = tokens[i].float()[None, :, None]
        return out

    return prefill


def masked_logp(history: list[int], seed: int, suppress, begin, first: bool) -> torch.Tensor:
    x = scores_of(history, seed).float()
    x[list(suppress)] = LOWEST
    if first:
        x[list(begin)] = LOWEST
    return x.log_softmax(-1).double()


def ct2_beam(prompt, K, max_new, seed, suppress, begin):
    """BeamSearch::search for one window (decoding.cc), in plain Python."""
    alive = [([], 0.0)] + [([], -math.inf)] * (K - 1)
    hyps = []
    for step in range(max_new):
        cands = []
        for k, (toks, sc) in enumerate(alive):
            lp = masked_logp(prompt + toks, seed, suppress, begin, step == 0)
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


def ct2_greedy(prompt, max_new, seed, suppress, begin):
    toks, score = [], 0.0
    for step in range(max_new):
        lp = masked_logp(prompt + toks, seed, suppress, begin, step == 0)
        v = int(lp.argmax())
        score += float(lp[v])
        if v == EOT:
            break
        toks.append(v)
    return toks, score


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("mode,K", [("beam", 3), ("beam", 5), ("greedy", 1)])
def test_search_matches_ct2(seed, mode, K):
    cfg = make_cfg()
    W, P = 3, 4
    suppress, begin = [5, 6, 7], [8, EOT]
    prompts = torch.tensor([[SOT, 10 + w, 20, 30] for w in range(W)])
    search = Search(cfg, fake_step(seed), W, K, P, mode, dtype=torch.float32, device="cpu", max_length=40,
                    graphs=False)
    results = search.run(prompts, fake_prefill(seed), suppress, begin, check_every=1)
    for w in range(W):
        prompt = prompts[w].tolist()
        if mode == "beam":
            toks, score = ct2_beam(prompt, K, search.max_new, seed, suppress, begin)
            assert results[w].tokens == toks
            assert results[w].hypothesis.normalized == pytest.approx(score / len(toks), abs=1e-4)
        else:
            toks, score = ct2_greedy(prompt, search.max_new, seed, suppress, begin)
            assert results[w].tokens == toks
            assert results[w].hypothesis.score == pytest.approx(score, abs=1e-4)
        p0 = scores_of(prompt[:1], seed).float().softmax(-1)[NOSPEECH].item()
        assert results[w].no_speech_prob == pytest.approx(p0, abs=1e-6)


def test_search_reaches_limit():
    """Every hypothesis registered at the last step (no EOS): tokens up to the limit."""
    cfg = make_cfg()
    prompts = torch.tensor([[SOT, 10, 20, 30]])
    search = Search(cfg, fake_step(0), 1, 2, 4, "beam", dtype=torch.float32, device="cpu", max_length=14,
                    graphs=False)
    res = search.run(prompts, fake_prefill(0), [EOT], [], check_every=1)[0]  # EOS suppressed
    assert res.reached_max_length and len(res.tokens) == search.max_new
    toks, _ = ct2_beam(prompts[0].tolist(), 2, search.max_new, 0, [EOT], [])
    assert res.tokens == toks


@pytest.mark.parametrize("mode,K", [("beam", 3), ("greedy", 1)])
@pytest.mark.parametrize("check_every", [1, 4])
def test_refilled_slots_match_independent_decodes(mode, K, check_every):
    """More windows than slots: each slot takes the next window when its own is done; every window's result is
    the one it gets decoded alone, whatever ran in the other slots meanwhile."""
    cfg = make_cfg()
    seed, P, M = 1, 4, 9
    suppress, begin = [5, 6, 7], [8, EOT]
    prompts = [[SOT, 10 + w, 20 + w % 3, 30] for w in range(M)]
    search = Search(cfg, fake_step(seed), 3, K, P, mode, dtype=torch.float32, device="cpu", max_length=40,
                    graphs=False)
    loaded = []
    results = search.stream(prompts, lambda slots, windows: loaded.append((slots, windows)), fake_prefill(seed),
                            suppress, begin, check_every=check_every)
    assert sorted(w for _, ws in loaded for w in ws) == list(range(M))
    assert len(loaded) > 1  # (refilled at least once)
    for w, prompt in enumerate(prompts):
        if mode == "beam":
            toks, score = ct2_beam(prompt, K, search.max_new, seed, suppress, begin)
            assert results[w].tokens == toks, w
            assert results[w].hypothesis.normalized == pytest.approx(score / len(toks), abs=1e-4)
        else:
            toks, score = ct2_greedy(prompt, search.max_new, seed, suppress, begin)
            assert results[w].tokens == toks, w
        p0 = scores_of(prompt[:1], seed).float().softmax(-1)[NOSPEECH].item()
        assert results[w].no_speech_prob == pytest.approx(p0, abs=1e-6)
