"""Beam search's step bookkeeping (search.Search, beam mode on CUDA) in two Triton kernels and one top-k, instead
of ~80 small PyTorch kernels (0.3-0.6 ms a step at 8-32 windows).

* :func:`scores`: per row, the suppressed (and, at a window's first step, begin-suppressed) tokens set to the
  lowest float, with timestamps also those CTranslate2's ApplyTimestampRules disables (as
  ``Search._timestamp_rules``), log-softmax, plus the row's cumulative log-probability: the K x V scores of each
  window's candidates; and each window's no-speech probability from its unmasked logits.
* the best 2K of each window's K x V scores (``torch.topk``).
* :func:`update`: per window, CTranslate2's rules as ``Search._beam`` / ``_choose`` / ``_body`` apply them: the
  first K candidates that end (EOS, or any at the last step) become finished hypotheses, each ended one gives its
  place to the next secondary candidate that does not end, the new beams take their parents' histories and slot
  tables; a window still in its prompt is fed its next prompt token instead; the no-speech probability is kept at
  ``<|startoftranscript|>``, the slot table of each row marks its new cache slot, and windows not done advance.
  Windows already done are left as they are (their results are their finished hypotheses).
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # (no Triton: Search keeps its PyTorch bookkeeping)
    triton = None

LOWEST = torch.finfo(torch.float32).min


if triton is not None:

    @triton.jit
    def _mask(v, ok, suppress, begin, first, TS: tl.constexpr, NOTS, TSB, EOT, max_init, first_ts, both, pair,
              fresh, last, latest):  # fmt: skip
        """The tokens a row may not take: suppressed ones, at a window's first step the begin-suppressed ones,
        and with timestamps CTranslate2's ApplyTimestampRules (before its probability check)."""
        m = tl.load(suppress + v, mask=ok, other=0) | (tl.load(begin + v, mask=ok, other=0) & first)
        if TS:
            ts = v >= TSB
            m |= v == NOTS
            m |= first_ts & (~ts | (v > TSB + max_init))
            m |= both & ts
            m |= pair & ((v < EOT) | (ts & (v < last)))
            m |= fresh & ts & (v <= latest)
        return m

    @triton.jit
    def _scores_kernel(logits, suppress, begin, t, cum, out, ns, hist, max_initial, V, NO_SPEECH, NOTS, TSB, EOT,
                       K: tl.constexpr, T: tl.constexpr, TS: tl.constexpr, LOW: tl.constexpr, BV: tl.constexpr,
                       TB: tl.constexpr):  # fmt: skip
        r = tl.program_id(0)
        b = r // K
        tw = tl.load(t + b)
        first = tw == 0
        base = r.to(tl.int64) * V
        # with timestamps: the row's last two tokens and latest timestamp (its history, positions < t)
        first_ts, both, pair, fresh, check = first, first, first, first, first
        last = tl.zeros((), tl.int64)
        latest = tl.zeros((), tl.int64)
        max_init = tl.zeros((), tl.int64)
        if TS:
            g = tl.maximum(tw, 0)
            j = tl.arange(0, TB)
            h = tl.load(hist + r.to(tl.int64) * T + j, mask=j < T, other=0)
            last = tl.sum(tl.where(j == g - 1, h, 0), 0)
            pen = tl.where(g >= 2, tl.sum(tl.where(j == g - 2, h, 0), 0), last)
            at = tl.max(tl.where((j < g) & (h >= TSB), j, -1), 0)
            latest = tl.sum(tl.where(j == at, h, 0), 0)
            first_ts = g == 0
            last_ts, pen_ts = last >= TSB, pen >= TSB
            both = ~first_ts & last_ts & pen_ts
            pair = ~first_ts & last_ts & ~pen_ts
            fresh = ~first_ts & ~last_ts & (at >= 0)
            check = pair | (~first_ts & ~last_ts)
            max_init = tl.load(max_initial)
        m_raw = tl.full((), float("-inf"), tl.float32)
        m_msk = tl.full((), float("-inf"), tl.float32)
        m_ts = tl.full((), float("-inf"), tl.float32)
        mx_text = tl.full((), float("-inf"), tl.float32)
        s_raw = tl.zeros((), tl.float32)
        s_msk = tl.zeros((), tl.float32)
        s_ts = tl.zeros((), tl.float32)
        for v0 in range(0, V, BV):  # online maxima and sums of exp
            v = v0 + tl.arange(0, BV)
            ok = v < V
            x = tl.load(logits + base + v, mask=ok, other=float("-inf"))
            masked = _mask(v, ok, suppress, begin, first, TS, NOTS, TSB, EOT, max_init, first_ts, both, pair, fresh,
                           last, latest)
            xm = tl.where(ok, tl.where(masked, LOW, x), float("-inf"))
            n_raw = tl.maximum(m_raw, tl.max(x, 0))
            n_msk = tl.maximum(m_msk, tl.max(xm, 0))
            s_raw = s_raw * tl.exp(m_raw - n_raw) + tl.sum(tl.exp(x - n_raw), 0)
            s_msk = s_msk * tl.exp(m_msk - n_msk) + tl.sum(tl.exp(xm - n_msk), 0)
            m_raw, m_msk = n_raw, n_msk
            if TS:  # the timestamps' sum and the text tokens' max, for the probability check
                xt = tl.where(v >= TSB, xm, float("-inf"))
                n_ts = tl.maximum(m_ts, tl.max(xt, 0))
                s_ts = s_ts * tl.exp(m_ts - n_ts) + tl.sum(tl.exp(xt - n_ts), 0)
                m_ts = n_ts
                mx_text = tl.maximum(mx_text, tl.max(tl.where(v < TSB, xm, float("-inf")), 0))
        log_sum = tl.log(s_msk)
        text_out = first & ~first  # (False) text masked: a timestamp is likelier than every text token
        if TS:
            lse = m_msk + log_sum
            text_out = check & ((m_ts - lse) + tl.log(s_ts) > mx_text - lse)
            m_msk = tl.where(text_out, m_ts, m_msk)
            log_sum = tl.where(text_out, tl.log(s_ts), log_sum)
        c = tl.load(cum + r)
        for v0 in range(0, V, BV):
            v = v0 + tl.arange(0, BV)
            ok = v < V
            x = tl.load(logits + base + v, mask=ok, other=0.0)
            masked = _mask(v, ok, suppress, begin, first, TS, NOTS, TSB, EOT, max_init, first_ts, both, pair, fresh,
                           last, latest)
            masked |= text_out & (v < TSB)
            xm = tl.where(masked, LOW, x)
            tl.store(out + base + v, (xm - m_msk - log_sum) + c, mask=ok)
        if r % K == 0:
            tl.store(ns + b, tl.exp(tl.load(logits + base + NO_SPEECH) - (m_raw + tl.log(s_raw))))

    @triton.jit
    def _update_kernel(top_s, top_i, t, pos, sot_pos, ns, no_speech, done, hist, cum, src, tokens, forced,
                       h_count, h_score, h_len, h_tokens, V, EOT, MAX_NEW, P, K: tl.constexpr, T: tl.constexpr,
                       C: tl.constexpr, S: tl.constexpr, KB: tl.constexpr, TB: tl.constexpr,
                       CB: tl.constexpr):  # fmt: skip
        b = tl.program_id(0)
        j = tl.arange(0, KB)
        jm = j < K
        rows = b * K + j
        tb = tl.load(t + b)
        pb = tl.load(pos + b)
        db = tl.load(done + b)
        if pb == tl.load(sot_pos + b):
            tl.store(no_speech + b, tl.load(ns + b))
        cc = tl.arange(0, CB)
        cm = jm[:, None] & (cc < C)[None, :]
        src_rows = src + rows.to(tl.int64)[:, None] * C + cc[None, :]
        if tb < 0:  # in the prompt: its next token; the rows' slot tables mark their new cache slot
            nxt = tl.load(forced + b * (P + 1) + tl.minimum(P + tb, P))
            tl.store(tokens + rows, tl.zeros([KB], tl.int64) + nxt, mask=jm)
            marked = tl.where(cc[None, :] == pb, rows.to(tl.int32)[:, None], tl.load(src_rows, mask=cm, other=0))
            tl.store(src_rows, marked, mask=cm)
            tl.store(pos + b, pb + 1)
            tl.store(t + b, tb + 1)
        elif db == 0:
            cand = b * 2 * K + j
            p_i = tl.load(top_i + cand, mask=jm, other=0)
            p_s = tl.load(top_s + cand, mask=jm, other=float("-inf"))
            s_i = tl.load(top_i + cand + K, mask=jm, other=0)
            s_s = tl.load(top_s + cand + K, mask=jm, other=float("-inf"))
            tok, par = p_i % V, p_i // V
            s_tok, s_par = s_i % V, s_i // V
            eos = (tok == EOT) & jm
            last = tb == MAX_NEW - 1
            finish = (eos | last) & jm
            slot = tl.load(h_count + b) + tl.cumsum(finish.to(tl.int64), 0) - 1
            # each ended candidate's place: the r-th secondary candidate that does not end
            sec_ok = (s_tok != EOT) & jm
            sec_rank = tl.cumsum(sec_ok.to(tl.int32), 0) - 1
            r = tl.cumsum(eos.to(tl.int32), 0) - 1
            take = eos & (r < tl.sum(sec_ok.to(tl.int32), 0))
            match = sec_ok[None, :] & (sec_rank[None, :] == r[:, None])  # [beam, secondary]
            new_tok = tl.where(take, tl.sum(tl.where(match, s_tok[None, :], 0), 1), tok)
            new_par = tl.where(take, tl.sum(tl.where(match, s_par[None, :], 0), 1), par)
            new_s = tl.where(take, tl.sum(tl.where(match, s_s[None, :], 0.0), 1), p_s)
            # every read of the window's state before any write (beams take other beams' rows)
            tt = tl.arange(0, TB)
            tm = jm[:, None] & (tt < T)[None, :]
            par_hist = tl.load(hist + (b * K + par).to(tl.int64)[:, None] * T + tt[None, :], mask=tm, other=0)
            new_hist = tl.load(hist + (b * K + new_par).to(tl.int64)[:, None] * T + tt[None, :], mask=tm, other=0)
            src_par = src + (b * K + new_par).to(tl.int64)[:, None] * C + cc[None, :]
            new_src = tl.load(src_par, mask=cm, other=0)
            new_src = tl.where(cc[None, :] == pb, (b * K + new_par).to(tl.int32)[:, None], new_src)
            tl.debug_barrier()
            # finished hypotheses (EOS not part of them; at the last step the token is)
            with_tok = ~eos
            h_row = tl.where(with_tok[:, None] & (tt[None, :] == tb), tok[:, None], par_hist)
            hm = finish[:, None] & (tt < T)[None, :]
            tl.store(h_tokens + (b * S + slot)[:, None] * T + tt[None, :], h_row, mask=hm)
            tl.store(h_score + b * S + slot, p_s, mask=finish)
            tl.store(h_len + b * S + slot, tb + with_tok.to(tl.int64), mask=finish)
            count = tl.load(h_count + b) + tl.sum(finish.to(tl.int64), 0)
            tl.store(h_count + b, count)
            # the new beams
            tl.store(hist + rows.to(tl.int64)[:, None] * T + tt[None, :],
                     tl.where(tt[None, :] == tb, new_tok[:, None], new_hist), mask=tm)
            tl.store(cum + rows, new_s, mask=jm)
            tl.store(tokens + rows, new_tok, mask=jm)
            tl.store(src_rows, new_src, mask=cm)
            ended = (count >= K) | last
            tl.store(done + b, ended)
            if not ended:
                tl.store(pos + b, pb + 1)
                tl.store(t + b, tb + 1)


def scores(s, logits: torch.Tensor, out: torch.Tensor, ns: torch.Tensor) -> None:
    """``out`` [N, V] = log_softmax(masked logits [N, V]) + cum[:, None] for ``s`` (a search.Search in beam mode:
    its suppressed tokens, its windows' steps, and with ``s.timestamps`` the timestamp rules on its rows'
    histories); ``ns`` [B]: no-speech probabilities."""
    n, v = logits.shape
    c = s.cfg
    T = s.hist.shape[1]
    _scores_kernel[(n,)](logits, s.suppress, s.begin, s.t, s.cum, out, ns, s.hist, s.max_initial, v, c.no_speech,
                         c.no_timestamps, c.timestamp_begin or 0, c.eot, K=s.K, T=T, TS=s.timestamps, LOW=LOWEST,
                         BV=4096, TB=triton.next_power_of_2(T), num_warps=8)


def update(s, top_s: torch.Tensor, top_i: torch.Tensor, ns: torch.Tensor) -> None:
    """The step's bookkeeping on ``s`` (a search.Search in beam mode) from the best 2K candidates."""
    T, C = s.hist.shape[1], s.src.shape[1]
    _update_kernel[(s.B,)](
        top_s, top_i, s.t, s.pos, s.sot_pos, ns, s.no_speech, s.done, s.hist, s.cum, s.src, s.tokens, s.forced,
        s.h_count, s.h_score, s.h_len, s.h_tokens, s.cfg.vocab_size, s.cfg.eot, s.max_new, s.P, K=s.K, T=T, C=C,
        S=s.h_score.shape[1], KB=triton.next_power_of_2(s.K), TB=triton.next_power_of_2(T),
        CB=triton.next_power_of_2(C), num_warps=4,
    )  # fmt: skip
