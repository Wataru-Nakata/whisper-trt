"""attn_plugin: the reference ops against plain attention, and the Triton kernels against the reference."""

import pytest
import torch

from whisper_trt import attn_plugin
from whisper_trt.modeling import attend


def _case(b=6, s=1, h=4, c=32, d=64, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    qkv = torch.randn(b, s, 3 * h * d, generator=g).to(device, dtype)
    kc = torch.randn(b, h, c, d, generator=g).to(device, dtype)
    vc = torch.randn(b, h, c, d, generator=g).to(device, dtype)
    offset = torch.randint(1, c - s, (b,), generator=g).to(device)
    src = torch.randint(0, b, (b, c), generator=g).to(device, torch.int32)
    return qkv, kc, vc, offset, src


def test_self_attn_reference_uses_slot_table():
    qkv, kc, vc, offset, src = _case(s=3)
    b, s = qkv.shape[:2]
    h, c, d = kc.shape[1:]
    out = attn_plugin.self_attn(qkv, kc, vc, offset, src)
    q, k, v = qkv.view(b, s, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)
    for r in range(b):
        o = int(offset[r])
        keys = torch.stack([kc[int(src[r, j]), :, j] for j in range(o)], 1)  # [H, o, D]
        vals = torch.stack([vc[int(src[r, j]), :, j] for j in range(o)], 1)
        k_all, v_all = torch.cat([keys, k[r]], 1), torch.cat([vals, v[r]], 1)
        mask = torch.ones(s, o + s, dtype=torch.bool)
        mask[:, o:] = torch.ones(s, s, dtype=torch.bool).tril()
        ref = attend(q[r : r + 1], k_all[None], v_all[None], mask[None, None], explicit=True)[0]
        assert torch.allclose(out[r], ref.transpose(0, 1).reshape(s, h * d), atol=1e-5)


cuda = pytest.mark.skipif(not torch.cuda.is_available() or attn_plugin.triton is None, reason="needs CUDA + Triton")


@cuda
@pytest.mark.parametrize("s", [1, 4, 17])
def test_self_attn_triton(s):
    qkv, kc, vc, offset, src = _case(b=10, s=s, h=20, c=96, dtype=torch.float16, device="cuda")
    ref = attn_plugin.self_attn(qkv, kc, vc, offset, src).float()
    for config in attn_plugin.TACTICS["self_attn"]:
        out = attn_plugin.self_attn_triton(qkv, kc, vc, offset, src, config).float()
        assert (out - ref).abs().max() < 2e-2, config


@cuda
@pytest.mark.parametrize("s", [1, 5, 20])
def test_cross_attn_triton(s):
    g = torch.Generator().manual_seed(1)
    q = torch.randn(3, s, 20 * 64, generator=g).cuda().half()
    k = torch.randn(3, 20, 1500, 64, generator=g).cuda().half()
    v = torch.randn(3, 20, 1500, 64, generator=g).cuda().half()
    ref = attn_plugin.cross_attn(q, k, v).float()
    for config in attn_plugin.TACTICS["cross_attn"]:
        out = attn_plugin.cross_attn_triton(q, k, v, config).float()
        assert (out - ref).abs().max() < 2e-2, config


def test_cross_split_reference_equals_attention():
    g = torch.Generator().manual_seed(2)
    q = torch.randn(2, 5, 4 * 64, generator=g)
    k = torch.randn(2, 4, 1500, 64, generator=g)
    v = torch.randn(2, 4, 1500, 64, generator=g)
    out = attn_plugin.cross_attn_combine(attn_plugin.cross_attn_split(q, k, v), q)
    assert torch.allclose(out, attn_plugin.cross_attn(q, k, v), atol=1e-5)


@cuda
@pytest.mark.parametrize("s", [1, 5, 20])
def test_cross_split_triton(s):
    g = torch.Generator().manual_seed(3)
    q = torch.randn(3, s, 20 * 64, generator=g).cuda().half()
    k = torch.randn(3, 20, 1500, 64, generator=g).cuda().half()
    v = torch.randn(3, 20, 1500, 64, generator=g).cuda().half()
    ref = attn_plugin.cross_attn(q, k, v).float()
    for config in attn_plugin.TACTICS["cross_split"]:
        part = attn_plugin.cross_attn_split_triton(q, k, v, config)
        for combine in attn_plugin.TACTICS["cross_combine"]:
            out = attn_plugin.cross_attn_combine_triton(part, q, combine).float()
            assert (out - ref).abs().max() < 2e-2, (config, combine)


@cuda
@pytest.mark.parametrize("m", [1, 40, 160])
def test_logits32_triton(m):
    g = torch.Generator().manual_seed(4)
    x = torch.randn(m, 1, 1280, generator=g).cuda().half()
    w = (torch.randn(51866, 1280, generator=g) * 0.05).cuda().half()
    ref = x.float() @ w.float().t()
    for config in attn_plugin.LINEAR_TACTICS:
        out = attn_plugin.logits32_triton(x, w, config)
        assert (out - ref).abs().max() < 1e-3, config
