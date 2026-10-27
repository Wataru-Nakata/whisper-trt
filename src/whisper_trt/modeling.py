"""Whisper in plain PyTorch: the Hugging Face checkpoints (model.safetensors) load with ``strict=True``.

Module names follow the checkpoint (``encoder.layers.0.self_attn.q_proj`` ...). The decoder runs one call
of ``S`` new tokens per row against a static KV cache (``Decoder.step``), the unit the TensorRT decoder
engines and the CUDA-graph decoding loops are built from.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from . import attn_plugin
from .config import WhisperConfig


class LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32 whatever the input type (as CTranslate2's kernels accumulate): exported as
    such, TensorRT would otherwise sum the squares of fp16 activations in fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not Precision.ln32:
            return super().forward(x)
        w, b = self.weight.float(), self.bias.float()
        return F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(x.dtype)


class Attention(nn.Module):
    explicit = False  # export option: attention written out with the scale on the query (see attend)

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h, self.d_k = n_heads, d // n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, d] -> [B, H, T, d_k]
        return x.reshape(x.shape[0], x.shape[1], self.h, self.d_k).transpose(1, 2)

    def qkv(self, x: torch.Tensor) -> torch.Tensor:
        """The three projections as one [B, S, 3d] (one GEMM once :meth:`fuse_qkv` has run)."""
        if hasattr(self, "qkv_proj"):
            return self.qkv_proj(x)
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)

    @torch.no_grad()
    def fuse_qkv(self) -> None:
        w = self.q_proj.weight
        lin = nn.Linear(w.shape[1], 3 * w.shape[0], device=w.device, dtype=w.dtype)
        lin.weight.copy_(torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]))
        lin.bias.copy_(torch.cat([self.q_proj.bias, torch.zeros_like(self.q_proj.bias), self.v_proj.bias]))
        self.qkv_proj = lin.requires_grad_(False)

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, T, d_k] -> [B, T, d]
        return self.out_proj(x.transpose(1, 2).reshape(x.shape[0], x.shape[2], self.h * self.d_k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.heads(self.q_proj(x)), self.heads(self.k_proj(x)), self.heads(self.v_proj(x))
        return self.merge(attend(q, k, v, explicit=self.explicit))


def attend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None, explicit: bool = False
) -> torch.Tensor:
    """softmax(q k^T / sqrt(d_k)) v. ``explicit``: written out with the scale on ``q``, for export (the ONNX
    exporter's ``scaled_dot_product_attention`` also scales ``k``, which TensorRT then computes in full)."""
    if not explicit:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    scores = (q * q.shape[-1] ** -0.5) @ k.transpose(-1, -2)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    if Precision.softmax32:
        return scores.float().softmax(-1).to(v.dtype) @ v
    return scores.softmax(-1) @ v


class Precision:
    """Export options: parts of the fp16 graph computed in fp32 (TensorRT keeps the types it is given)."""

    softmax32 = False  # attention probabilities
    ln32 = True  # LayerNorm (see LayerNorm)
    # The MLPs' GELU: TensorRT's fp16 erf multiplied the encoder's error by 5 (scripts/encoder_variants.py)
    gelu32 = True
    residual32 = False  # the encoder's residual stream
    logits32 = False  # the decoder's vocabulary projection


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float()).to(x.dtype) if Precision.gelu32 else F.gelu(x)


class EncoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, units: int):
        super().__init__()
        self.self_attn = Attention(d, n_heads)
        self.self_attn_layer_norm = LayerNorm(d)
        self.fc1 = nn.Linear(d, units)
        self.fc2 = nn.Linear(units, d)
        self.final_layer_norm = LayerNorm(d)

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(gelu(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.fc1.weight.dtype  # (x is fp32 with an fp32 residual stream)
        x = x + self.self_attn(self.self_attn_layer_norm(x).to(dt)).to(x.dtype)
        return x + self.mlp(self.final_layer_norm(x).to(dt)).to(x.dtype)


class Encoder(nn.Module):
    def __init__(self, cfg: WhisperConfig):
        super().__init__()
        d = cfg.d_model
        self.conv1 = nn.Conv1d(cfg.n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(cfg.enc_frames, d)
        self.layers = nn.ModuleList(EncoderLayer(d, cfg.enc_heads, cfg.enc_ffn) for _ in range(cfg.enc_layers))
        self.layer_norm = LayerNorm(d)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B, n_mels, 3000] -> [B, 1500, d]"""
        dt = self.conv1.weight.dtype
        x = gelu(self.conv1(feats.to(dt)))
        x = gelu(self.conv2(x)).transpose(1, 2)
        x = x + self.embed_positions.weight[: x.shape[1]]
        if Precision.residual32:
            x = x.float()
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x).to(dt)


class DecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, units: int):
        super().__init__()
        self.self_attn = Attention(d, n_heads)
        self.self_attn_layer_norm = LayerNorm(d)
        self.encoder_attn = Attention(d, n_heads)
        self.encoder_attn_layer_norm = LayerNorm(d)
        self.fc1 = nn.Linear(d, units)
        self.fc2 = nn.Linear(units, d)
        self.final_layer_norm = LayerNorm(d)

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(gelu(self.fc1(x)))


class Decoder(nn.Module):
    def __init__(self, cfg: WhisperConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, d)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, d)
        self.layers = nn.ModuleList(DecoderLayer(d, cfg.dec_heads, cfg.dec_ffn) for _ in range(cfg.dec_layers))
        self.layer_norm = LayerNorm(d)
        # export option: self- and/or cross-attention ("self", "cross") as attn_plugin ops (TensorRT plugins)
        self.plugins: frozenset[str] = frozenset()

    def cross_kv(self, enc: torch.Tensor) -> torch.Tensor:
        """Every layer's cross-attention keys/values, projected once: enc [B, T, d] -> [L, 2, B, H, T, d_k]"""
        kv = []
        for layer in self.layers:
            att = layer.encoder_attn
            kv.append(torch.stack([att.heads(att.k_proj(enc)), att.heads(att.v_proj(enc))]))
        return torch.stack(kv)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if "logits" in self.plugins:  # fp16 inputs, fp32 accumulation (attn_plugin.logits32)
            return attn_plugin.logits32(self.layer_norm(x), self.embed_tokens.weight)
        if Precision.logits32:  # (the vocabulary projection in fp32: the scores beam search compares)
            return self.layer_norm(x).float() @ self.embed_tokens.weight.float().t()
        return (self.layer_norm(x) @ self.embed_tokens.weight.t()).float()

    def step(
        self,
        tokens: torch.Tensor,
        offset: torch.Tensor,
        self_k: list[torch.Tensor],
        self_v: list[torch.Tensor],
        cross_k: list[torch.Tensor],
        cross_v: list[torch.Tensor],
        src: torch.Tensor | None = None,
        explicit: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``S`` new tokens per row through the decoder against a static KV cache.

        tokens:           [B, S] token ids; row b at positions offset[b] .. offset[b]+S-1
        offset:           [B] int64, number of tokens already in each row's cache
        self_k, self_v:   per layer [B, H, C, d_k] cache with capacity C (slots >= offset[b] are ignored)
        cross_k, cross_v: per layer [W, H, T, d_k] from :meth:`cross_kv`, for W windows: rows b*K .. b*K+K-1
                          (K = B / W, e.g. the beams of a window) attend to window b
        src:              [B, C] int32, the row whose cache holds each slot of a row (None: its own; see
                          attn_plugin.self_attn)
        returns:          logits [B, S, V] (fp32), new keys/values [L, 2, B, H, S, d_k]

        The caller writes the new keys/values into the cache at ``offset``. A window's rows share its
        cross-attention keys/values: their queries are attended together, as K * S queries of one window.
        """
        b, s = tokens.shape
        cap = self_k[0].shape[2]
        new_idx = torch.arange(s, device=tokens.device)
        positions = offset.unsqueeze(1) + new_idx  # [B, S]
        x = self.embed_tokens(tokens) + self.embed_positions(positions)  # [B, S, d]
        plugin_self, plugin_cross = "self" in self.plugins, "cross" in self.plugins
        if plugin_self and src is None:
            src = torch.arange(b, dtype=torch.int32, device=tokens.device)[:, None].expand(b, cap).contiguous()

        # each row attends to its cache slots < offset[b] and causally to its new tokens
        slot = torch.arange(cap, device=tokens.device)
        cache_ok = (slot < offset.unsqueeze(1))[:, None, None, :].expand(b, 1, s, cap)  # [B, 1, S, C]
        new_ok = (new_idx.unsqueeze(0) <= new_idx.unsqueeze(1)).expand(b, 1, s, s)  # [B, 1, S, S]
        mask = torch.cat([cache_ok, new_ok], dim=-1)  # [B, 1, S, C+S]

        new_kv = []
        for i, layer in enumerate(self.layers):
            att = layer.self_attn
            h = layer.self_attn_layer_norm(x)
            if plugin_self:
                qkv = att.qkv(h)  # [B, S, 3d]
                new_kv.append(qkv.view(b, s, 3, att.h, att.d_k)[:, :, 1:].permute(2, 0, 3, 1, 4))
                x = x + att.out_proj(attn_plugin.self_attn(qkv, self_k[i], self_v[i], offset, src))
            else:
                q, k, v = att.heads(att.q_proj(h)), att.heads(att.k_proj(h)), att.heads(att.v_proj(h))
                new_kv.append(torch.stack([k, v]))
                kc, vc = self_k[i], self_v[i]
                if src is not None:
                    kc, vc = attn_plugin.gather_cache(kc, src), attn_plugin.gather_cache(vc, src)
                k_all = torch.cat([kc, k], dim=2)  # [B, H, C+S, d_k]
                v_all = torch.cat([vc, v], dim=2)
                x = x + att.merge(attend(q, k_all, v_all, mask, explicit=True))

            xa = layer.encoder_attn
            h = xa.q_proj(layer.encoder_attn_layer_norm(x))  # [B, S, d]
            h = h.reshape(cross_k[i].shape[0], -1, h.shape[-1])  # [W, K*S, d]
            if "cross_split" in self.plugins:
                part = attn_plugin.cross_attn_split(h, cross_k[i], cross_v[i])
                y = xa.out_proj(attn_plugin.cross_attn_combine(part, h))
            elif plugin_cross:
                y = xa.out_proj(attn_plugin.cross_attn(h, cross_k[i], cross_v[i]))
            else:
                y = xa.merge(attend(xa.heads(h), cross_k[i], cross_v[i], explicit=explicit))  # [W, K*S, d]
            x = x + y.reshape(b, s, y.shape[-1])
            x = x + layer.mlp(layer.final_layer_norm(x))
        return self.logits(x), torch.stack(new_kv)

    def forward(
        self, tokens: torch.Tensor, cross_kv: torch.Tensor, heads: tuple[tuple[int, int], ...] = ()
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Whole sequences at once (no cache): tokens [B, S], cross_kv from :meth:`cross_kv`. Returns logits
        [B, S, V] and, for ``heads`` ((layer, head) pairs), their cross-attention weights [B, len(heads), S, T]
        (fp32; word timestamps)."""
        b, s = tokens.shape
        x = self.embed_tokens(tokens) + self.embed_positions.weight[:s]
        causal = torch.ones(s, s, dtype=torch.bool, device=tokens.device).tril()
        weights = {}  # (layer, head) -> [B, S, T]
        for i, layer in enumerate(self.layers):
            att = layer.self_attn
            h = layer.self_attn_layer_norm(x)
            q, k, v = att.heads(att.q_proj(h)), att.heads(att.k_proj(h)), att.heads(att.v_proj(h))
            x = x + att.merge(attend(q, k, v, causal))
            src = layer.encoder_attn
            q = src.heads(src.q_proj(layer.encoder_attn_layer_norm(x)))
            mine = [hd for li, hd in heads if li == i]
            if mine:
                scores = (q.float() * q.shape[-1] ** -0.5) @ cross_kv[i, 0].float().transpose(-1, -2)
                p = scores.softmax(-1)  # [B, H, S, T]
                weights.update({(i, hd): p[:, hd] for hd in mine})
                x = x + src.merge(p.to(x.dtype) @ cross_kv[i, 1])
            else:
                x = x + src.merge(attend(q, cross_kv[i, 0], cross_kv[i, 1]))
            x = x + layer.mlp(layer.final_layer_norm(x))
        stacked = torch.stack([weights[p] for p in heads], dim=1) if heads else None
        return self.logits(x), stacked


class WhisperModel(nn.Module):
    def __init__(self, cfg: WhisperConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)


def load_state_dict(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """The checkpoint's tensors without the ``model.`` prefix (and without ``proj_out``, tied to the token
    embedding in every Whisper checkpoint)."""
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {model_dir}")
    sd = {}
    for f in files:
        sd.update(load_file(str(f)))
    out = {}
    for k, v in sd.items():
        if k == "proj_out.weight":
            continue
        out[k.removeprefix("model.")] = v
    return out


def load_model(
    model_dir: str | Path, cfg: WhisperConfig, device: str | torch.device = "cuda", dtype=torch.float16
) -> WhisperModel:
    sd = load_state_dict(model_dir)
    with torch.device("meta"):
        model = WhisperModel(cfg)
    model.load_state_dict(sd, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Whisper's encoder positions (the checkpoints store them; for reference)."""
    inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-inc * torch.arange(channels // 2))
    t = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([t.sin(), t.cos()], dim=1)
