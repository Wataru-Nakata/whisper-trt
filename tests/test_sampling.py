"""The sampling search's random numbers (search.Search._uniform) and its Gumbel-max samples."""

import torch

from whisper_trt.config import WhisperConfig
from whisper_trt.search import Search

V = 40


def make_search(rows: int = 5) -> Search:
    cfg = WhisperConfig(n_mels=4, d_model=4, enc_layers=1, enc_heads=1, enc_ffn=4, dec_layers=1, dec_heads=1,
                        dec_ffn=4, vocab_size=V, eot=1, sot=2, no_speech=3)
    return Search(cfg, None, 1, rows, 4, "sample", dtype=torch.float32, device="cpu", max_length=40, graphs=False)


def test_uniform_is_uniform_and_varies():
    s = make_search()
    draws = []
    for seed in range(20):
        s.seed.fill_(seed)
        for t in range(10):
            s.t.fill_(t)
            draws.append(s._uniform())
    u = torch.stack(draws)  # [200, N, V]
    assert 0 < float(u.min()) and float(u.max()) < 1
    assert abs(float(u.mean()) - 0.5) < 0.01
    hist = torch.histc(u, bins=10, min=0, max=1) / u.numel()
    assert float((hist - 0.1).abs().max()) < 0.01
    assert not torch.equal(draws[0], draws[1])  # another step
    assert not torch.equal(draws[0][0], draws[0][1])  # another row
    s.seed.fill_(0)
    s.t.fill_(0)
    assert torch.equal(s._uniform(), draws[0])  # the same seed and step: the same numbers


def test_gumbel_samples_follow_the_tempered_distribution():
    s = make_search()
    g = torch.Generator().manual_seed(0)
    logp = (torch.randn(V, generator=g) * 2).log_softmax(-1)
    temperature = 0.7
    counts = torch.zeros(V)
    for seed in range(400):
        s.seed.fill_(seed)
        for t in range(5):
            s.t.fill_(t)
            u = s._uniform().clamp_(min=1e-20)
            scaled = (logp / temperature).log_softmax(-1)
            token = (scaled - (-u.log()).log()).argmax(-1)  # [N]
            counts += torch.bincount(token, minlength=V).float()
    freq = counts / counts.sum()
    want = (logp / temperature).softmax(-1)
    assert float((freq - want).abs().max()) < 0.02
