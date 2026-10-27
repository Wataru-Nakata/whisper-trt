"""Decoding with CTranslate2 4.8.2's rules (what faster-whisper runs), on static GPU buffers.

One :class:`Search` decodes ``B`` windows with ``K`` rows each (``N = B * K`` decoder rows, grouped by window):

* beam search (``mode="beam"``, K = beam size): CTranslate2's ``BeamSearch``: the cumulative log-probabilities
  of all K x V continuations of a window, the best 2K of them; each of the first K that ends (EOS, or any at the
  last step) is a finished hypothesis and gives its place to the next of the other K that does not end. A
  window is done once it has K hypotheses; the best one by score / length (``length_penalty=1``, EOS not
  counted) wins. Only the first beam takes part in the first step.
* greedy (``mode="greedy"``, K = 1) and sampling (``mode="sample"``, K = best_of independent rows; softmax of
  ``log_probs / temperature``; the scores stay those of the untempered log-probabilities): CTranslate2's
  ``GreedySearch``; the best of a window's K rows by score / length wins.

Every step disables the suppressed tokens (to the lowest float, as ``DisableTokens``), and the first one also
the "begin" tokens (blank, EOS). At most ``min(max_length // 2, max_length - (prompt length - 1))`` tokens are
generated (224 for Whisper's 448, the limit faster-whisper users meet). The no-speech probability is the
softmax of the logits at <|startoftranscript|> at <|nospeech|>.

A step is all device work (decoder call, token choice) on fixed buffers, so it can be captured in a CUDA graph;
the host checks whether every window is done every few steps. Beam search never moves the KV cache: each row
has a table ``src`` [N, C] of the row whose cache holds each of its slots (attn_plugin.self_attn); a new beam
takes its parent's table, and the token it is fed next is written to its own row's slot.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .config import WhisperConfig

LOWEST = torch.finfo(torch.float32).min


@dataclass
class Hypothesis:
    tokens: list[int]
    score: float  # cumulative log-probability (EOS included)
    length: int  # generated tokens, EOS not counted

    @property
    def normalized(self) -> float:
        return self.score / self.length if self.length else float("-inf")

    @property
    def avg_logprob(self) -> float:
        """faster-whisper's: the cumulative log-probability over ``length + 1`` tokens."""
        return self.score / (self.length + 1)


@dataclass
class WindowResult:
    hypothesis: Hypothesis
    no_speech_prob: float
    hypotheses: list[Hypothesis]  # every finished one (beam) or every row (greedy/sampling)
    max_new: int  # the most tokens the search could generate

    @property
    def tokens(self) -> list[int]:
        return self.hypothesis.tokens

    @property
    def reached_max_length(self) -> bool:
        """Whether the decode stopped at the token limit (no EOS before it)."""
        return self.hypothesis.length >= self.max_new


def max_new_tokens(prompt_length: int, max_length: int = 448) -> int:
    """Tokens CTranslate2 generates at most for a prompt of ``prompt_length`` (its decode starts at step
    ``prompt_length - 1``, the prompt's last token being the first one fed)."""
    return min(max_length // 2, max_length - (prompt_length - 1))


def cache_buckets(capacity: int, smallest: int = 64) -> list[int]:
    out, c = [], smallest
    while c < capacity:
        out.append(c)
        c *= 2
    return out + [capacity]


class Search:
    """Static buffers and the step of one decoding of ``batch`` windows x ``rows`` rows.

    ``step_fn(tokens, offset, cache, logits, new_kv, src)`` returns a closure that runs a decoder call: tokens
    [N, S], offset [N] (int64), the self-attention cache as a [L, 2, N, H, C, d_k] view, logits [N, S, V]
    (fp32), new keys/values [L, 2, N, H, S, d_k] written by the call and the slot table src [N, C] (int32)
    (cross-attention keys/values are bound by the caller of ``step_fn``)."""

    def __init__(
        self,
        cfg: WhisperConfig,
        step_fn: Callable,
        batch: int,
        rows: int,
        prompt_length: int,
        mode: str = "beam",
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        max_length: int = 448,
        graphs: bool = True,
    ):
        if mode not in ("beam", "greedy", "sample"):
            raise ValueError(f"unknown search mode {mode!r}")
        if mode == "greedy" and rows != 1:
            raise ValueError("greedy search has one row per window")
        self.cfg, self.mode, self.device = cfg, mode, torch.device(device)
        self.B, self.K, self.N = batch, rows, batch * rows
        self.P = prompt_length
        self.max_new = max_new_tokens(prompt_length, max_length)
        self.capacity = prompt_length + self.max_new
        self.buckets = [c for c in cache_buckets(self.capacity) if c >= prompt_length + 1] or [self.capacity]
        L, H, dk, V = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size
        N, B, K, T = self.N, self.B, self.K, self.max_new
        dev = self.device
        self.step_fn = step_fn
        self.graphs = graphs
        # self-attention cache: one flat buffer, laid out [L, 2, N, H, C, d_k] for the current bucket C
        self.cache_flat = torch.zeros(L * 2 * N * H * self.capacity * dk, dtype=dtype, device=dev)
        self.src_flat = torch.zeros(N * self.capacity, dtype=torch.int32, device=dev)
        self.src_scratch = torch.zeros_like(self.src_flat)
        self.row_ids = torch.arange(N, dtype=torch.int32, device=dev)[:, None]
        self.tokens = torch.zeros(N, 1, dtype=torch.long, device=dev)
        self.offset = torch.zeros(N, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)  # cache slot of the token fed this step
        self.t = torch.zeros(1, dtype=torch.long, device=dev)  # generated tokens so far
        self.logits = torch.zeros(N, 1, V, dtype=torch.float32, device=dev)
        self.new_kv = torch.zeros(L, 2, N, H, 1, dk, dtype=dtype, device=dev)
        self.suppress = torch.zeros(V, dtype=torch.bool, device=dev)
        self.hist = torch.zeros(N, T, dtype=torch.long, device=dev)  # generated tokens of each row
        self.cum = torch.zeros(N, dtype=torch.float32, device=dev)  # cumulative log-probabilities
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)
        self.row_done = torch.zeros(N, dtype=torch.bool, device=dev)  # greedy/sampling rows
        self.length = torch.zeros(N, dtype=torch.long, device=dev)  # greedy/sampling: tokens without EOS
        self.temperature = torch.ones(1, dtype=torch.float32, device=dev)
        if mode == "beam":
            S = 2 * K + 1  # finished hypotheses: at most K - 1 + K, and a slot for the ones not registered
            self.h_count = torch.zeros(B, dtype=torch.long, device=dev)
            self.h_score = torch.full((B, S), float("-inf"), dtype=torch.float32, device=dev)
            self.h_len = torch.zeros(B, S, dtype=torch.long, device=dev)
            self.h_tokens = torch.zeros(B, S, T, dtype=torch.long, device=dev)
        self._steps: dict[int, Callable] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    # --- cache layout -----------------------------------------------------------------------------------
    def cache(self, c: int) -> torch.Tensor:
        cfg = self.cfg
        shape = (cfg.dec_layers, 2, self.N, cfg.dec_heads, c, cfg.head_dim)
        n = 1
        for s in shape:
            n *= s
        return self.cache_flat[:n].view(shape)

    def src(self, c: int, flat: torch.Tensor | None = None) -> torch.Tensor:
        return (self.src_flat if flat is None else flat)[: self.N * c].view(self.N, c)

    def _grow(self, old: int, new: int) -> None:
        """Re-lay the cache and slot table out from bucket ``old`` to bucket ``new`` (the slots kept)."""
        kept = self.cache(old).clone()
        dst = self.cache(new)
        dst.zero_()
        dst[..., :old, :] = kept
        table = self.src(old).clone()
        self.src(new).copy_(self.row_ids.expand(self.N, new))
        self.src(new)[:, :old] = table

    # --- the step ---------------------------------------------------------------------------------------
    def _call(self, c: int) -> Callable:
        if c not in self._steps:
            self._steps[c] = self.step_fn(self.tokens, self.offset, self.cache(c), self.logits, self.new_kv,
                                          self.src(c))
        return self._steps[c]

    def _body(self, c: int) -> None:
        """One step on device: decoder call on the last tokens, their keys/values into the cache, the next
        tokens chosen (and, for beam search, the cache reordered)."""
        self.offset.copy_(self.pos.expand(self.N))
        self._call(c)()
        self.cache(c).index_copy_(4, self.pos, self.new_kv)
        self.src(c).index_copy_(1, self.pos, self.row_ids)
        self._choose(self.logits[:, 0], c, first=False)
        self.pos += 1
        self.t += 1

    def _run_step(self, c: int) -> None:
        if self.graphs:
            self._graphs[c].replay()
        else:
            self._body(c)

    def prepare(self) -> None:
        """Capture every bucket's step graph (their warm-up runs change the state, which ``run`` resets)."""
        if not self.graphs or self._graphs:
            return
        self.begin = torch.zeros_like(self.suppress)
        for c in self.buckets:
            self._graphs[c] = _capture(lambda c=c: self._body(c))

    # --- token choice -----------------------------------------------------------------------------------
    def _log_probs(self, logits: torch.Tensor, first: bool) -> torch.Tensor:
        logits = logits.masked_fill(self.suppress, LOWEST)
        if first:
            logits = logits.masked_fill(self.begin, LOWEST)
        return logits.log_softmax(-1)

    def _choose(self, logits: torch.Tensor, c: int, first: bool) -> None:
        logp = self._log_probs(logits, first)  # [N, V]
        if self.mode == "beam":
            self._beam(logp, c, first)
        else:
            self._greedy(logp)

    def _greedy(self, logp: torch.Tensor) -> None:
        if self.mode == "sample":
            scaled = (logp / self.temperature).log_softmax(-1)
            u = torch.rand_like(scaled).clamp_(min=1e-20)
            token = (scaled - (-u.log()).log()).argmax(-1)  # Gumbel-max: a sample of softmax(logp / T)
        else:
            token = logp.argmax(-1)
        live = ~self.row_done
        score = logp.gather(1, token[:, None])[:, 0]
        self.cum += torch.where(live, score, 0.0)
        eos = token == self.cfg.eot
        last = self.t == self.max_new - 1
        keep = live & ~eos  # a generated token (EOS is not part of the hypothesis)
        t = self.t.expand(self.N)
        self.hist.scatter_(1, t[:, None], torch.where(keep, token, self.hist.gather(1, t[:, None])[:, 0])[:, None])
        self.length += keep.long()
        self.row_done |= eos | last
        self.done.copy_(self.row_done.view(self.B, self.K).all(1))
        self.tokens.copy_(token[:, None])

    def _beam(self, logp: torch.Tensor, c: int, first: bool) -> None:
        B, K, V = self.B, self.K, self.cfg.vocab_size
        scores = (logp + self.cum[:, None]).view(B, K * V)
        top_s, top_i = scores.topk(2 * K, dim=1)  # [B, 2K], best first
        tok, par = top_i % V, top_i // V
        eos = tok == self.cfg.eot
        last = (self.t == self.max_new - 1).expand(B, K)
        live = ~self.done[:, None]
        finish = (eos[:, :K] | last) & live  # [B, K]: hypotheses registered this step
        # register them: slot count + rank, beyond the hypotheses' storage for the others
        rank = finish.long().cumsum(1) - 1
        slot = torch.where(finish, self.h_count[:, None] + rank, 2 * K)  # [B, K]
        T = self.max_new  # the history buffer's length
        parent_hist = self.hist.view(B, K, T).gather(1, par[:, :K, None].expand(B, K, T))  # [B, K, T]
        with_token = (~eos[:, :K]).long()  # hypotheses ending without EOS (last step) keep their token
        step = self.t.view(1, 1, 1).expand(B, K, 1)
        parent_hist = parent_hist.scatter(2, step, torch.where(with_token[..., None] > 0, tok[:, :K, None],
                                                               parent_hist.gather(2, step)))
        self.h_tokens.scatter_(1, slot[..., None].expand(B, K, T), parent_hist)
        self.h_score.scatter_(1, slot, top_s[:, :K])
        self.h_len.scatter_(1, slot, self.t.expand(B, K) + with_token)
        self.h_count += finish.long().sum(1)
        # each primary candidate that ended gives its place to the next secondary one that does not end
        sec_ok = ~eos[:, K:]  # [B, K]
        ar = torch.arange(K, device=tok.device)
        order = torch.where(sec_ok, ar, K + ar).argsort(1)  # secondary non-EOS candidates first, in order
        n_ok = sec_ok.long().sum(1, keepdim=True)
        ended = eos[:, :K]
        r = ended.long().cumsum(1) - 1
        repl = K + order.gather(1, r.clamp(min=0, max=K - 1))
        sel = torch.where(ended & (r < n_ok), repl, ar.expand(B, K))  # [B, K] candidate of each new beam
        new_tok, new_par, new_s = tok.gather(1, sel), par.gather(1, sel), top_s.gather(1, sel)
        self.done |= (self.h_count >= K) | last[:, 0]
        # the new beams' histories and caches are their parents'
        rows = (new_par + torch.arange(B, device=tok.device)[:, None] * K).view(-1)  # [N] parent rows
        hist = self.hist.index_select(0, rows)
        hist.scatter_(1, self.t.expand(self.N)[:, None], new_tok.view(-1, 1))
        self.hist.copy_(hist)
        self.cum.copy_(new_s.view(-1))
        self.tokens.copy_(new_tok.view(-1, 1))
        table, scratch = self.src(c), self.src(c, self.src_scratch)
        torch.index_select(table, 0, rows, out=scratch)
        table.copy_(scratch)

    # --- driver -----------------------------------------------------------------------------------------
    def run(
        self,
        prompt: torch.Tensor,
        prefill: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        suppress: list[int],
        begin_suppress: list[int],
        temperature: float = 1.0,
        check_every: int = 4,
    ) -> list[WindowResult]:
        """Decode. ``prompt`` [B, P] token ids (every window's rows get its prompt). ``prefill(tokens, cache,
        src)`` runs the decoder over the prompt of all N rows into the cache at slots 0..P-1 and returns
        their logits [N, P, V] (fp32) (``src``: the slot table, every row its own)."""
        B, K, P = self.B, self.K, self.P
        cfg = self.cfg
        self.prepare()
        self.suppress.zero_()
        if suppress:
            self.suppress[torch.tensor(list(suppress), device=self.device)] = True
        self.begin = torch.zeros_like(self.suppress)
        if begin_suppress:
            self.begin[torch.tensor(list(begin_suppress), device=self.device)] = True
        self.temperature.fill_(temperature)
        self.cum.zero_()
        if self.mode == "beam":  # only the first beam takes part in the first step
            self.cum.view(B, K)[:, 1:] = float("-inf")
            self.h_count.zero_()
            self.h_score.fill_(float("-inf"))
            self.h_len.zero_()
        self.done.zero_()
        self.row_done.zero_()
        self.length.zero_()
        self.hist.zero_()
        self.t.zero_()
        self.pos.fill_(P)
        rows = prompt.to(self.device).repeat_interleave(K, dim=0)  # [N, P]
        bucket = self.buckets[0]
        self.cache(bucket).zero_()
        self.src(bucket).copy_(self.row_ids.expand(self.N, bucket))
        logits = prefill(rows, self.cache(bucket), self.src(bucket))  # [N, P, V]
        sot = (prompt[0] == cfg.sot).nonzero()[0, 0].item()
        no_speech = logits[:, sot].softmax(-1)[:, cfg.no_speech].view(B, K)[:, 0].tolist()
        self._choose(logits[:, -1], bucket, first=True)
        self.t += 1
        steps = 1
        while steps < self.max_new:
            slots = P + steps  # cache slots needed by the next call
            if slots > bucket:
                nxt = next(c for c in self.buckets if c >= slots)
                self._grow(bucket, nxt)
                bucket = nxt
            self._run_step(bucket)
            steps += 1
            if steps % check_every == 0 and bool(self.done.all()):
                break
        return self._results(no_speech)

    def _results(self, no_speech: list[float]) -> list[WindowResult]:
        out = []
        B, K = self.B, self.K
        if self.mode == "beam":
            count, score = self.h_count.tolist(), self.h_score.tolist()
            lengths, tokens = self.h_len.tolist(), self.h_tokens.tolist()
            for b in range(B):
                hyps = [Hypothesis(tokens[b][j][: lengths[b][j]], score[b][j], lengths[b][j])
                        for j in range(min(count[b], 2 * K))]
                best = max(hyps, key=lambda h: h.normalized)
                out.append(WindowResult(best, no_speech[b], hyps, self.max_new))
            return out
        hist, cum, length = self.hist.tolist(), self.cum.tolist(), self.length.tolist()
        for b in range(B):
            hyps = [Hypothesis(hist[r][: length[r]], cum[r], length[r]) for r in range(b * K, (b + 1) * K)]
            best = max(hyps, key=lambda h: h.normalized)
            out.append(WindowResult(best, no_speech[b], hyps, self.max_new))
        return out


def _capture(body: Callable[[], None]) -> torch.cuda.CUDAGraph:
    """Warm ``body`` up on a side stream (TensorRT shape setup, allocator), then capture it."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    return graph
