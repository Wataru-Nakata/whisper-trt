"""Decoding with CTranslate2 4.8.2's rules (what faster-whisper runs), on static GPU buffers.

One :class:`Search` decodes windows in ``B`` slots of ``K`` rows each (``N = B * K`` decoder rows, grouped by
slot):

* beam search (``mode="beam"``, K = beam size): CTranslate2's ``BeamSearch``: the cumulative log-probabilities
  of all K x V continuations of a window, the best 2K of them; each of the first K that ends (EOS, or any at the
  last step) is a finished hypothesis and gives its place to the next of the other K that does not end. A
  window is done once it has K hypotheses; the best one by score / length (``length_penalty=1``, EOS not
  counted) wins. Only the first beam takes part in the first step.
* greedy (``mode="greedy"``, K = 1) and sampling (``mode="sample"``, K = best_of independent rows; softmax of
  ``log_probs / temperature``; the scores stay those of the untempered log-probabilities): CTranslate2's
  ``GreedySearch``; the best of a window's K rows by score / length wins.

Every step disables the suppressed tokens (to the lowest float, as ``DisableTokens``), and a window's first step
also the "begin" tokens (blank, EOS). At most ``min(max_length // 2, max_length - (prompt length - 1))`` tokens
are generated (224 for Whisper's 448, the limit faster-whisper users meet). A window's prompt tokens are fed by
steps too (forced tokens, no search), so loading a window into a slot is a few state writes: no prefill call,
no host synchronization. The no-speech probability is the softmax of the logits at <|startoftranscript|> at
<|nospeech|>, as CTranslate2 takes it from its prompt's forward pass.

A step is all device work (decoder call, token choice) on fixed buffers, captured in one CUDA graph; every slot
has its own position, so windows of any lengths share the steps. The host checks the slots after every few steps
and refills those whose window is done with the next window (:meth:`Search.stream`): a batch does not wait for
its slowest window. The checks read flags copied asynchronously a few steps back, so the GPU always has steps
queued while the host collects results and loads windows. Beam search never moves the KV cache: each row has a
table ``src`` [N, C] of the row whose cache holds each of its slots (attn_plugin.self_attn); a new beam takes its
parent's table, and the token it is fed next is written to its own row's slot.
"""

import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import torch

from . import beam_kernels
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
    """Static buffers and the step of decodings in ``batch`` slots x ``rows`` rows.

    ``step_fn(tokens, offset, cache, logits, new_kv, src)`` returns a closure that runs a decoder call: tokens
    [N, 1], offset [N] (int64), the self-attention cache [L, 2, N, H, C, d_k], logits [N, 1, V] (fp32), new
    keys/values [L, 2, N, H, 1, d_k] written by the call and the slot table src [N, C] (int32) (the slots'
    cross-attention keys/values are bound by the caller of ``step_fn``). A closure with ``writes_cache = True``
    writes the new keys/values into the cache at ``offset`` itself instead."""

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
        graphs: bool | None = None,
        fused: bool | None = None,
    ):
        if mode not in ("beam", "greedy", "sample"):
            raise ValueError(f"unknown search mode {mode!r}")
        if mode == "greedy" and rows != 1:
            raise ValueError("greedy search has one row per window")
        self.cfg, self.mode, self.device = cfg, mode, torch.device(device)
        self.B, self.K, self.N = batch, rows, batch * rows
        self.P = prompt_length
        self.max_new = max_new_tokens(prompt_length, max_length)
        self.capacity = C = prompt_length + self.max_new
        L, H, dk, V = cfg.dec_layers, cfg.dec_heads, cfg.head_dim, cfg.vocab_size
        N, B, K, T = self.N, self.B, self.K, self.max_new
        dev = self.device
        self.step_fn = step_fn
        # CUDA graphs: on unless WHISPER_TRT_GRAPHS=0 (for debugging)
        self.graphs = graphs if graphs is not None else os.environ.get("WHISPER_TRT_GRAPHS", "1") != "0"
        self.cache = torch.zeros(L, 2, N, H, C, dk, dtype=dtype, device=dev)
        self.src = torch.zeros(N, C, dtype=torch.int32, device=dev)
        self.src_scratch = torch.zeros_like(self.src)
        self.row_ids = torch.arange(N, dtype=torch.int32, device=dev)[:, None]
        self.row_ar = torch.arange(N, device=dev)
        self.tokens = torch.zeros(N, 1, dtype=torch.long, device=dev)
        self.offset = torch.zeros(N, dtype=torch.long, device=dev)
        self.pos = torch.zeros(B, dtype=torch.long, device=dev)  # per slot: cache slot of the token fed next
        # per slot: tokens generated so far; negative while the prompt is fed (-(P-1) at its first token)
        self.t = torch.zeros(B, dtype=torch.long, device=dev)
        self.forced = torch.zeros(B, prompt_length + 1, dtype=torch.long, device=dev)  # the slots' prompts
        self.sot_pos = torch.zeros(B, dtype=torch.long, device=dev)  # position of <|startoftranscript|>
        self.no_speech = torch.zeros(B, dtype=torch.float32, device=dev)
        self.logits = torch.zeros(N, 1, V, dtype=torch.float32, device=dev)
        self.new_kv = torch.zeros(L, 2, N, H, 1, dk, dtype=dtype, device=dev)
        self.suppress = torch.zeros(V, dtype=torch.bool, device=dev)
        self.begin = torch.zeros(V, dtype=torch.bool, device=dev)
        self.hist = torch.zeros(N, T, dtype=torch.long, device=dev)  # generated tokens of each row
        self.cum = torch.zeros(N, dtype=torch.float32, device=dev)  # cumulative log-probabilities
        self.done = torch.ones(B, dtype=torch.bool, device=dev)  # (empty slots are done)
        self.row_done = torch.ones(N, dtype=torch.bool, device=dev)  # greedy/sampling rows
        self.length = torch.zeros(N, dtype=torch.long, device=dev)  # greedy/sampling: tokens without EOS
        self.temperature = torch.ones(1, dtype=torch.float32, device=dev)
        self.seed = torch.zeros(1, dtype=torch.int64, device=dev)
        if mode == "sample":  # (row, token) counters of the random numbers, see _uniform
            self.counter = (torch.arange(N, device=dev)[:, None] * V + torch.arange(V, device=dev)).view(N, V)
        if mode == "beam":
            S = 2 * K + 1  # finished hypotheses: at most K - 1 + K, and a slot for the ones not registered
            self.h_count = torch.zeros(B, dtype=torch.long, device=dev)
            self.h_score = torch.full((B, S), float("-inf"), dtype=torch.float32, device=dev)
            self.h_len = torch.zeros(B, S, dtype=torch.long, device=dev)
            self.h_tokens = torch.zeros(B, S, T, dtype=torch.long, device=dev)
        # beam search's bookkeeping in Triton kernels (beam_kernels) on CUDA, unless WHISPER_TRT_FUSED_BEAM=0
        if fused is None:
            fused = (mode == "beam" and dev.type == "cuda" and beam_kernels.triton is not None
                     and os.environ.get("WHISPER_TRT_FUSED_BEAM", "1") != "0")
        self.fused = fused
        if fused:
            self.scores = torch.zeros(N, V, dtype=torch.float32, device=dev)
            self.ns = torch.zeros(B, dtype=torch.float32, device=dev)
        self._step: Callable | None = None
        self._graph: torch.cuda.CUDAGraph | None = None

    def rows(self, slot: int) -> slice:
        return slice(slot * self.K, (slot + 1) * self.K)

    # --- the step ---------------------------------------------------------------------------------------
    def _body(self) -> None:
        """One step on device: decoder call on every row's last token at its slot's position, the new
        keys/values into the cache there, the next tokens chosen; the slots still decoding advance."""
        if self._step is None:
            self._step = self.step_fn(self.tokens, self.offset, self.cache, self.logits, self.new_kv, self.src)
        self.offset.copy_(self.pos.repeat_interleave(self.K))
        self._step()
        if not getattr(self._step, "writes_cache", False):  # (a backend may write them into the cache itself)
            self.cache[:, :, self.row_ar, :, self.offset, :] = self.new_kv[:, :, :, :, 0, :].permute(2, 0, 1, 3, 4)
        if self.fused:  # the same as below, in three kernels
            beam_kernels.scores(self.logits[:, 0], self.suppress, self.begin, self.t, self.cum, self.scores,
                                self.ns, self.K, self.cfg.no_speech)
            top_s, top_i = self.scores.view(self.B, -1).topk(2 * self.K, dim=1)
            beam_kernels.update(self, top_s, top_i, self.ns)
            return
        self.src[self.row_ar, self.offset] = self.row_ids[:, 0]
        lg = self.logits[:: self.K, 0]  # [B, V] (a window's rows are the same until its first step)
        ns = (lg[:, self.cfg.no_speech] - lg.logsumexp(-1)).exp()
        self.no_speech.copy_(torch.where(self.pos == self.sot_pos, ns, self.no_speech))
        self._choose(self.logits[:, 0])
        live = (~self.done).long()  # (a window that ended stays at its last step, within the buffers)
        self.pos += live
        self.t += live

    def _run_step(self) -> None:
        if not self.graphs:
            self._body()
            return
        if self._graph is None:  # (its warm-up runs change the state: capture before any window is loaded)
            raise RuntimeError("Search.prepare() must run before the first window is loaded")
        self._graph.replay()

    def prepare(self) -> None:
        """Capture the step graph (its warm-up runs change the state; ``stream`` resets every slot it loads)."""
        if self.graphs and self._graph is None:
            self._graph = _capture(self._body)
            self.done.fill_(True)
            self.row_done.fill_(True)

    # --- token choice -----------------------------------------------------------------------------------
    def _choose(self, logits: torch.Tensor) -> None:
        first = (self.t == 0).repeat_interleave(self.K)[:, None]  # [N, 1]: a window's first step
        logits = logits.masked_fill(self.suppress, LOWEST).masked_fill(self.begin & first, LOWEST)
        logp = logits.log_softmax(-1)  # [N, V]
        prompt = (self.t < 0).repeat_interleave(self.K)  # [N]: rows still fed their prompt
        kept = (self.hist.clone(), self.cum.clone(), self.src.clone())
        if self.mode == "beam":
            self._beam(logp, prompt)
        else:
            self._greedy(logp, prompt)
        # rows in their prompt: the next prompt token, nothing else changes
        hist, cum, src = kept
        self.hist.copy_(torch.where(prompt[:, None], hist, self.hist))
        self.cum.copy_(torch.where(prompt, cum, self.cum))
        self.src.copy_(torch.where(prompt[:, None], src, self.src))
        nxt = self.forced.gather(1, (self.P + self.t).clamp(max=self.P)[:, None])[:, 0].repeat_interleave(self.K)
        self.tokens.copy_(torch.where(prompt, nxt, self.tokens[:, 0])[:, None])

    def _uniform(self) -> torch.Tensor:
        """Uniform random numbers [N, V] in (0, 1) from a counter-based hash (PCG's output function, on 32-bit
        words in int64 arithmetic) of the seed, the window's step and each (row, token): no generator state, so
        it runs inside a captured CUDA graph (torch's generator does not, on the model's stream), and a seed
        gives the same samples."""
        mask = 0xFFFFFFFF
        t = self.t.repeat_interleave(self.K)[:, None]
        x = (self.counter + (self.seed * 0x6C8E9CF5 + t * 0x85EBCA77)) & mask  # (products < 2**63)
        for _ in range(2):
            x = (x * 747796405 + 2891336453) & mask
            x = (((x >> ((x >> 28) + 4)) ^ x) * 277803737) & mask
            x = ((x >> 22) ^ x) & mask
        return (x.float() + 0.5) / 4294967296.0

    def _greedy(self, logp: torch.Tensor, prompt: torch.Tensor) -> None:
        if self.mode == "sample":
            scaled = (logp / self.temperature).log_softmax(-1)
            u = self._uniform().clamp_(min=1e-20)
            token = (scaled - (-u.log()).log()).argmax(-1)  # Gumbel-max: a sample of softmax(logp / T)
        else:
            token = logp.argmax(-1)
        t = self.t.repeat_interleave(self.K).clamp(min=0)  # [N]
        live = ~self.row_done & ~prompt
        score = logp.gather(1, token[:, None])[:, 0]
        self.cum += torch.where(live, score, 0.0)
        eos = token == self.cfg.eot
        last = t == self.max_new - 1
        keep = live & ~eos  # a generated token (EOS is not part of the hypothesis)
        self.hist.scatter_(1, t[:, None], torch.where(keep, token, self.hist.gather(1, t[:, None])[:, 0])[:, None])
        self.length += keep.long()
        self.row_done |= (eos | last) & ~prompt
        self.done.copy_(self.row_done.view(self.B, self.K).all(1))
        self.tokens.copy_(token[:, None])

    def _beam(self, logp: torch.Tensor, prompt: torch.Tensor) -> None:
        B, K, V = self.B, self.K, self.cfg.vocab_size
        scores = (logp + self.cum[:, None]).view(B, K * V)
        top_s, top_i = scores.topk(2 * K, dim=1)  # [B, 2K], best first
        tok, par = top_i % V, top_i // V
        eos = tok == self.cfg.eot
        last = (self.t == self.max_new - 1)[:, None].expand(B, K)
        live = ~self.done[:, None] & ~prompt.view(B, K)[:, :1]
        finish = (eos[:, :K] | last) & live  # [B, K]: hypotheses registered this step
        # register them: slot count + rank, beyond the hypotheses' storage for the others
        rank = finish.long().cumsum(1) - 1
        slot = torch.where(finish, self.h_count[:, None] + rank, 2 * K)  # [B, K]
        T = self.max_new  # the history buffer's length
        parent_hist = self.hist.view(B, K, T).gather(1, par[:, :K, None].expand(B, K, T))  # [B, K, T]
        with_token = (~eos[:, :K]).long()  # hypotheses ending without EOS (last step) keep their token
        t = self.t.clamp(min=0)
        step = t.view(B, 1, 1).expand(B, K, 1)
        parent_hist = parent_hist.scatter(2, step, torch.where(with_token[..., None] > 0, tok[:, :K, None],
                                                               parent_hist.gather(2, step)))
        self.h_tokens.scatter_(1, slot[..., None].expand(B, K, T), parent_hist)
        self.h_score.scatter_(1, slot, top_s[:, :K])
        self.h_len.scatter_(1, slot, t[:, None].expand(B, K) + with_token)
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
        # the new beams' histories and slot tables are their parents'
        rows = (new_par + torch.arange(B, device=tok.device)[:, None] * K).view(-1)  # [N] parent rows
        hist = self.hist.index_select(0, rows)
        hist.scatter_(1, t.repeat_interleave(K)[:, None], new_tok.view(-1, 1))
        self.hist.copy_(hist)
        self.cum.copy_(new_s.view(-1))
        self.tokens.copy_(new_tok.view(-1, 1))
        torch.index_select(self.src, 0, rows, out=self.src_scratch)
        self.src.copy_(self.src_scratch)

    # --- driver -----------------------------------------------------------------------------------------
    def _start(self, slots: list[int], prompts: list[list[int]]) -> None:
        """Load windows into ``slots``: their prompts are fed from the next step on (the slot's position 0, then
        one token per step, see _choose). Only asynchronous writes from pinned memory: the queued steps go on."""
        K, P, dev = self.K, self.P, self.device

        def to_dev(x) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.long, pin_memory=True).to(dev, non_blocking=True)

        s_idx = to_dev(slots)
        r_idx = (s_idx[:, None] * K + torch.arange(K, device=dev)).view(-1)
        prompt = to_dev([p + [0] for p in prompts])  # [n, P + 1] (a pad: the index past the prompt)
        self.forced[s_idx] = prompt
        self.sot_pos[s_idx] = to_dev([p.index(self.cfg.sot) for p in prompts])
        self.tokens[r_idx, 0] = prompt[:, 0].repeat_interleave(K)
        # (index_fill_, not `x[idx] = value`: that copies the value to the GPU and waits for the queued steps)
        self.pos.index_fill_(0, s_idx, 0)
        self.t.index_fill_(0, s_idx, -(P - 1))
        self.src[r_idx] = self.row_ids[r_idx]
        first = torch.arange(len(r_idx), device=dev) % K == 0  # only the first beam takes part in the first step
        self.cum[r_idx] = torch.where(first | (self.mode != "beam"), 0.0, float("-inf"))
        if self.mode == "beam":
            self.h_count.index_fill_(0, s_idx, 0)
            self.h_score.index_fill_(0, s_idx, float("-inf"))
            self.h_len.index_fill_(0, s_idx, 0)
        self.hist.index_fill_(0, r_idx, 0)
        self.length.index_fill_(0, r_idx, 0)
        self.row_done.index_fill_(0, r_idx, False)
        self.done.index_fill_(0, s_idx, False)

    def stream(
        self,
        prompts: list[list[int]],
        load: Callable[[list[int], list[int]], None],
        suppress: list[int],
        begin_suppress: list[int],
        temperature: float = 1.0,
        seed: int = 0,
        check_every: int = 1,
        ahead: int = 1,
    ) -> list[WindowResult]:
        """Decode len(prompts) windows (prompts of length P), at most B at a time: a slot whose window is done
        takes the next one. ``load(slots, windows)`` puts the windows' cross-attention keys/values into those
        slots (on the current stream, before the next step).

        The slots' done flags are copied to the host after every ``check_every`` steps and read ``ahead``
        checks later, so ``ahead * check_every`` steps stay queued while the host collects the finished windows
        (also asynchronously) and loads the next ones; a finished window's slot runs that many more steps
        (which change nothing in it) before it is refilled."""
        self.prepare()
        self.suppress.zero_()
        if suppress:
            self.suppress[torch.tensor(list(suppress), device=self.device)] = True
        self.begin.zero_()
        if begin_suppress:
            self.begin[torch.tensor(list(begin_suppress), device=self.device)] = True
        self.temperature.fill_(temperature)
        self.seed.fill_(seed & 0x7FFFFFFF)
        self.done.fill_(True)
        self.row_done.fill_(True)
        M = len(prompts)
        results: list[WindowResult | None] = [None] * M
        window = [-1] * self.B  # the window in each slot
        loaded = [-1] * self.B  # the last check before each slot's window was loaded (older flags are stale)
        flags: deque = deque()  # (event, done flags on the host, check), oldest first
        finished: deque = deque()  # (windows, their final state on the host) not yet results
        checks = 0
        nxt = 0

        def fill(slots: list[int]) -> None:
            nonlocal nxt
            pairs = []
            for s in slots:
                if nxt < M:
                    pairs.append((s, nxt))
                    nxt += 1
            if not pairs:
                return
            load([s for s, _ in pairs], [w for _, w in pairs])
            self._start([s for s, _ in pairs], [prompts[w] for _, w in pairs])
            for s, w in pairs:
                window[s] = w
                loaded[s] = checks - 1

        def collect(wait: bool) -> None:
            while finished and (wait or _ready(finished[0][1][0])):
                windows, state = finished.popleft()
                for w, r in zip(windows, self._results(state)):
                    results[w] = r

        fill(list(range(self.B)))
        while any(w >= 0 for w in window):
            for _ in range(check_every):
                self._run_step()
            done = _to_host(self.done)
            flags.append((_record(self.device), done, checks))  # (the event after the copy)
            checks += 1
            if len(flags) <= ahead:
                continue
            ev, done, check = flags.popleft()
            _wait(ev)
            done = done.tolist()
            slots = [s for s in range(self.B) if window[s] >= 0 and done[s] and check > loaded[s]]
            if slots:
                finished.append(([window[s] for s in slots], self._final_state(slots)))
                for s in slots:
                    window[s] = -1
                fill(slots)
            collect(wait=False)
        collect(wait=True)
        return results

    def run(self, prompt: torch.Tensor, suppress: list[int], begin_suppress: list[int], temperature: float = 1.0,
            seed: int = 0, check_every: int = 1, ahead: int = 1) -> list[WindowResult]:
        """The B windows whose cross-attention keys/values are already in the slots (``prompt`` [B, P])."""
        return self.stream(prompt.tolist(), lambda slots, windows: None, suppress, begin_suppress, temperature,
                           seed, check_every, ahead)

    def _final_state(self, slots: list[int]) -> tuple:
        """What the results of the windows in ``slots`` (all done) are made of, copied to the host
        asynchronously (queued before the slots are loaded again): see :meth:`_results`."""
        dev, K = self.device, self.K
        idx = torch.tensor(slots, dtype=torch.long, pin_memory=dev.type == "cuda").to(dev, non_blocking=True)
        if self.mode == "beam":
            parts = [self.no_speech[idx], self.h_count[idx], self.h_score[idx], self.h_len[idx], self.h_tokens[idx]]
        else:
            rows = (idx[:, None] * K + torch.arange(K, device=dev)).view(-1)
            parts = [self.no_speech[idx], self.hist[rows], self.cum[rows], self.length[rows]]
        parts = [_to_host(p) for p in parts]
        return _record(dev), parts

    def _results(self, state: tuple) -> list[WindowResult]:
        ev, parts = state
        _wait(ev)
        K = self.K
        out = []
        if self.mode == "beam":
            no_speech, count, score, lengths, tokens = (p.tolist() for p in parts)
            for j in range(len(no_speech)):
                hyps = [Hypothesis(tokens[j][i][: lengths[j][i]], score[j][i], lengths[j][i])
                        for i in range(min(count[j], 2 * K))]
                best = max(hyps, key=lambda h: h.normalized)
                out.append(WindowResult(best, no_speech[j], hyps, self.max_new))
            return out
        no_speech, hist, cum, length = (p.tolist() for p in parts)
        for j in range(len(no_speech)):
            hyps = [Hypothesis(hist[r][: length[r]], cum[r], length[r]) for r in range(j * K, (j + 1) * K)]
            best = max(hyps, key=lambda h: h.normalized)
            out.append(WindowResult(best, no_speech[j], hyps, self.max_new))
        return out


def _to_host(t: torch.Tensor) -> torch.Tensor:
    """A copy of ``t`` on the host: asynchronous (pinned memory) from a GPU."""
    return t.to("cpu", non_blocking=True) if t.is_cuda else t.clone()


def _record(device: torch.device) -> "torch.cuda.Event | None":
    if device.type != "cuda":
        return None
    ev = torch.cuda.Event()
    ev.record()
    return ev


def _wait(ev) -> None:
    if ev is not None:
        ev.synchronize()


def _ready(ev) -> bool:
    return ev is None or ev.query()


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
