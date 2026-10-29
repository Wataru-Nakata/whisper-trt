"""search.Search.serve: windows given by a source as results come back, and windows decoded by beam search (or
greedily) and sampled windows in the same steps (``sampling_slots=True``), against CTranslate2's loops
(test_search) and a plain transcription of its GreedySearch sampling with search.uniform's random numbers."""

import math

import pytest
import torch
from test_search import EOT, SOT, V, ct2_beam, ct2_greedy, fake_prefill, fake_step, make_cfg, masked_logp

from whisper_trt import beam_kernels
from whisper_trt.search import Job, Search, mix_seed, uniform

SUPPRESS, BEGIN = [5, 6, 7], [8, EOT]


def ref_sample(prompt, rows, temperature, wseed, max_new, seed):
    """GreedySearch over ``rows`` rows, each sampled (Gumbel-max on log_probs / temperature) with the numbers of
    its row and step; the best row by score / length."""
    hyps = []
    for r in range(rows):
        toks, score = [], 0.0
        for step in range(max_new):
            lp = masked_logp(prompt + toks, seed, SUPPRESS, BEGIN, step == 0).float()
            u = uniform(torch.tensor([wseed]), torch.tensor([step]), torch.tensor([r]), V)[0].clamp(min=1e-20)
            v = int((lp / temperature - (-u.log()).log()).argmax())
            score += float(lp[v])
            if v == EOT:
                break
            toks.append(v)
        hyps.append((toks, score))
    return max(hyps, key=lambda h: h[1] / len(h[0]) if h[0] else -math.inf)


def check(job: Job, result, mode: str, K: int, limit: int, seed: int) -> None:
    if job.temperature > 0:
        toks, score = ref_sample(job.prompt, job.rows or K, job.temperature, job.seed, limit, seed)
    elif mode == "beam":
        toks, score = ct2_beam(job.prompt, K, limit, seed, SUPPRESS, BEGIN)
    else:
        toks, score = ct2_greedy(job.prompt, limit, seed, SUPPRESS, BEGIN)
    assert result.tokens == toks, job.key
    if toks:
        assert result.hypothesis.normalized == pytest.approx(score / len(toks), abs=1e-4)


class Source:
    """Jobs in order; with ``chain``, job (r, i + 1) only once (r, i)'s result is back (a recording's windows)."""

    def __init__(self, jobs: list[Job], chain: bool = False):
        self.jobs, self.chain = jobs, chain
        self.ready = [j for j in jobs if not chain or j.key[1] == 0]
        self.results: dict = {}
        self.loads: list = []

    def take(self, n, block=False):
        out, self.ready = self.ready[:n], self.ready[n:]
        return out

    def load(self, slots, jobs):
        self.loads.append((slots, [j.key for j in jobs]))

    def result(self, job, result):
        self.results[job.key] = result
        if self.chain:
            r, i = job.key
            self.ready += [j for j in self.jobs if j.key == (r, i + 1)]


def make_jobs(prefilled: bool, K: int, n: int = 9) -> list[Job]:
    jobs = []
    for w in range(n):
        if prefilled:  # 4 .. 28 tokens: previous text before <|startoftranscript|>
            prompt = [4 + (w * 7 + j) % 30 for j in range(3 * w)] + [SOT, 10 + w, 20, 30]
        else:
            prompt = [SOT, 10 + w, 20 + w % 3, 30]
        temperature = (0.0, 0.5, 1.0)[w % 3]
        rows = (K if w % 2 else max(1, K - 1)) if temperature > 0 else None
        jobs.append(Job(prompt, temperature, mix_seed(7, w), rows, key=w))
    return jobs


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("prefilled", [False, True])
@pytest.mark.parametrize("mode,K,device,fused", [("beam", 3, "cpu", False), ("greedy", 3, "cpu", False),
                                                 ("beam", 3, "cuda", True), ("beam", 3, "cuda", False),
                                                 ("beam", 5, "cuda", True)])
def test_mixed_windows_match_references(seed, prefilled, mode, K, device, fused):
    """Windows at temperature 0 (beam search or greedy) and sampled windows (best_of = K or fewer rows) in the same
    slots, refilled: each as CTranslate2 decodes it alone (the samples with search.uniform's numbers)."""
    if device == "cuda" and (not torch.cuda.is_available() or (fused and beam_kernels.triton is None)):
        pytest.skip("needs CUDA (and Triton)")
    cfg = make_cfg()
    search = Search(cfg, fake_step(seed), 3, K, None if prefilled else 4, mode, dtype=torch.float32, device=device,
                    max_length=40, graphs=False, fused=fused, sampling_slots=True)
    jobs = make_jobs(prefilled, K)
    source = Source(jobs)
    search.serve(source, SUPPRESS, BEGIN, prefill=fake_prefill(search, seed) if prefilled else None)
    assert len(source.loads) > 1
    for job in jobs:
        limit = min(20, 40 - (len(job.prompt) - 1)) if prefilled else search.max_new
        check(job, source.results[job.key], mode, K, limit, seed)


@pytest.mark.parametrize("device,fused", [("cpu", False), ("cuda", True)])
def test_chained_windows_do_not_depend_on_slots(device, fused):
    """A recording's next window given only after its previous one's result (slots left empty meanwhile take it
    at a later check): the same results with 2 slots and with 4, each the one decoded alone."""
    if device == "cuda" and (not torch.cuda.is_available() or beam_kernels.triton is None):
        pytest.skip("needs CUDA and Triton")
    cfg = make_cfg()
    seed, K = 2, 3
    jobs = []
    for r in range(3):
        for i in range(4):
            w = r * 4 + i
            prompt = [4 + (w * 5 + j) % 30 for j in range(2 * i)] + [SOT, 10 + r, 20 + i, 30]
            temperature = 0.0 if (r + i) % 3 else 0.6
            jobs.append(Job(prompt, temperature, mix_seed(3, r, i), key=(r, i)))
    results = {}
    for B in (2, 4):
        search = Search(cfg, fake_step(seed), B, K, None, "beam", dtype=torch.float32, device=device,
                        max_length=40, graphs=False, fused=fused, sampling_slots=True)
        source = Source(jobs, chain=True)
        search.serve(source, SUPPRESS, BEGIN, check_every=1, ahead=1, prefill=fake_prefill(search, seed))
        assert set(source.results) == {j.key for j in jobs}
        results[B] = source.results
    for job in jobs:
        assert results[2][job.key].tokens == results[4][job.key].tokens, job.key
        check(job, results[2][job.key], "beam", K, min(20, 40 - (len(job.prompt) - 1)), seed)


def test_mix_seed_is_deterministic_and_spread():
    assert mix_seed(0, 1) == mix_seed(0, 1)
    seeds = {mix_seed(s, r, i) for s in range(4) for r in range(8) for i in range(8)}
    assert len(seeds) == 256 and all(0 <= x < 2**31 for x in seeds)
