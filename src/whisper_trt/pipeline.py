"""The public API: :class:`Whisper` (``from_pretrained``, ``generate``, ``decode_windows``, ``detect_language``)."""

import functools
import logging
import zlib
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch

from . import hub
from .audio import AudioLike, load_audio
from .config import N_FRAMES, WhisperConfig, load_config
from .features import log_mel
from .modeling import load_model
from .runtime import TorchBackend, TRTBackend
from .search import Search, WindowResult
from .tokenizer import Tokenizer

log = logging.getLogger(__name__)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data))


@dataclass
class Decoded:
    """One window's transcript, with the values faster-whisper's ``generate_with_fallback`` judges it by."""

    text: str
    tokens: list[int]
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    temperature: float
    reached_max_length: bool
    attempts: list[dict] = field(default_factory=list)  # earlier temperatures' decodes (fallback)


def _on_stream(fn):
    """Run a method's GPU work on the model's own stream (TensorRT synchronizes extra on the default one)."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with torch.cuda.stream(self.stream):
            return fn(self, *args, **kwargs)

    return wrapper


class Whisper:
    def __init__(self, cfg: WhisperConfig, tokenizer: Tokenizer, backend, model=None, max_batch: int = 16):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.backend = backend
        self.model = model  # the PyTorch model (torch backend, word timestamps)
        self.max_batch = max_batch
        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream()
        L, H, T, dk = cfg.dec_layers, cfg.dec_heads, cfg.enc_frames, cfg.head_dim
        self._cross_flat = torch.empty(L * 2 * max_batch * H * T * dk, dtype=backend.dtype, device=self.device)
        self._searches: dict[tuple, Search] = {}
        self._prefills: dict[tuple, tuple] = {}

    @classmethod
    def from_pretrained(
        cls,
        name: str = "openai/whisper-large-v3",
        backend: str = "tensorrt",
        dtype: str = "float16",
        max_batch: int = 16,
        max_rows: int = 80,
        cache_dir: str | None = None,
    ) -> "Whisper":
        model_dir = hub.resolve(name)
        cfg = load_config(model_dir)
        tokenizer = Tokenizer(model_dir, cfg)
        if backend == "torch":
            model = load_model(model_dir, cfg, dtype=DTYPES[dtype])
            return cls(cfg, tokenizer, TorchBackend(model), model, max_batch)
        if backend != "tensorrt":
            raise ValueError(f"backend must be 'tensorrt' or 'torch', got {backend!r}")
        from .build import BuildOptions, ensure_engines

        opts = BuildOptions(dtype=dtype, max_batch=max_batch, max_rows=max_rows)
        engine_dir = ensure_engines(model_dir, cfg, opts, cache_dir)
        return cls(cfg, tokenizer, TRTBackend(cfg, engine_dir), None, max_batch)

    # --- building blocks --------------------------------------------------------------------------------
    @_on_stream
    def features(self, audio: list[AudioLike]) -> torch.Tensor:
        """[W, n_mels, 3000] features of up-to-30 s windows (each normalized on its own, like faster-whisper's
        batched pipeline)."""
        waves = [a if isinstance(a, torch.Tensor) and a.dim() == 1 else load_audio(a) for a in audio]
        return log_mel(waves, self.cfg.n_mels, self.device, N_FRAMES)

    def cross(self, windows: int) -> torch.Tensor:
        """The cross-attention buffer [L, 2, W, H, T, d_k] for ``windows`` windows (one buffer for all W)."""
        c = self.cfg
        shape = (c.dec_layers, 2, windows, c.dec_heads, c.enc_frames, c.head_dim)
        return self._cross_flat[: int(np.prod(shape))].view(shape)

    @_on_stream
    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.shape[0] > self.max_batch:
            raise ValueError(f"at most {self.max_batch} windows per call")
        cross = self.cross(feats.shape[0])
        self.backend.encode(feats.to(self.device, torch.float32).contiguous(), cross)
        return cross

    def _search(self, mode: str, windows: int, rows: int, prompt_length: int) -> Search:
        key = (mode, windows, rows, prompt_length)
        if key not in self._searches:
            step = partial(self.backend.step_fn, cross=self.cross(windows))
            self._searches[key] = Search(self.cfg, step, windows, rows, prompt_length, mode,
                                         dtype=self.backend.dtype, device=self.device)
        return self._searches[key]

    def _prefill(self, windows: int, rows: torch.Tensor, cache: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Run the decoder over the prompt rows [N, P] into ``cache`` slots 0..P-1; logits [N, P, V]."""
        n, p = rows.shape
        key = (windows, n, p, cache.shape[4], cache.data_ptr(), src.data_ptr())
        if key not in self._prefills:
            c = self.cfg
            tokens = torch.zeros(n, p, dtype=torch.long, device=self.device)
            offset = torch.zeros(n, dtype=torch.long, device=self.device)
            logits = torch.zeros(n, p, c.vocab_size, dtype=torch.float32, device=self.device)
            new_kv = torch.zeros(c.dec_layers, 2, n, c.dec_heads, p, c.head_dim, dtype=self.backend.dtype,
                                 device=self.device)
            fn = self.backend.step_fn(tokens, offset, cache, logits, new_kv, src, cross=self.cross(windows))
            self._prefills[key] = (fn, tokens, logits, new_kv)
        fn, tokens, logits, new_kv = self._prefills[key]
        tokens.copy_(rows)
        fn()
        cache[..., :p, :] = new_kv
        return logits

    def prompt(
        self,
        language: str | None,
        task: str = "transcribe",
        without_timestamps: bool = True,
        previous_tokens: list[int] | None = None,
    ) -> list[int]:
        """faster-whisper's ``get_prompt`` (no prefix or hotwords)."""
        out = []
        if previous_tokens:
            out = [self.cfg.sot_prev] + previous_tokens[-(self.cfg.max_target_positions // 2 - 1) :]
        out += self.tokenizer.sot_sequence(language, task)
        if without_timestamps:
            out.append(self.cfg.no_timestamps)
        return out

    @_on_stream
    def generate(
        self,
        cross: torch.Tensor,
        prompt: list[int] | list[list[int]],
        beam_size: int = 5,
        temperature: float = 0.0,
        best_of: int = 5,
        suppress_tokens: tuple[int, ...] | None = (-1,),
        suppress_blank: bool = True,
    ) -> list[WindowResult]:
        """CTranslate2's ``Whisper.generate`` on encoded windows (``cross`` from :meth:`encode`), as
        faster-whisper calls it: beam search at temperature 0 (greedy for ``beam_size=1``), else ``best_of``
        samples. ``prompt``: one for all windows or one per window (all of the same length)."""
        windows = cross.shape[2]
        prompts = [prompt] * windows if not prompt or isinstance(prompt[0], int) else prompt
        if len({len(p) for p in prompts}) != 1:
            raise ValueError("every window's prompt must have the same length")
        if not self.cfg.timestamp_begin or prompts[0][-1] != self.cfg.no_timestamps:
            raise NotImplementedError("decoding with timestamps is not supported yet")
        if temperature > 0:
            mode, rows = "sample", best_of
        else:
            mode, rows = ("beam", beam_size) if beam_size > 1 else ("greedy", 1)
        search = self._search(mode, windows, rows, len(prompts[0]))
        suppress = self.tokenizer.suppressed_tokens(suppress_tokens)
        begin = self.cfg.begin_suppress_tokens if suppress_blank else ()
        rows_t = torch.tensor(prompts, dtype=torch.long)
        return search.run(rows_t, partial(self._prefill, windows), list(suppress), list(begin),
                          temperature=temperature if temperature > 0 else 1.0)

    @_on_stream
    def detect_language(self, cross: torch.Tensor) -> list[list[tuple[str, float]]]:
        """Language probabilities of encoded windows: the softmax of the language tokens' logits after
        <|startoftranscript|> (CTranslate2's ``detect_language``), best first."""
        windows = cross.shape[2]
        c = self.cfg
        key = ("lid", windows)
        if key not in self._searches:
            # a one-token prefill against a cache of the smallest bucket
            cache = torch.zeros(c.dec_layers, 2, windows, c.dec_heads, 16, c.head_dim, dtype=self.backend.dtype,
                                device=self.device)
            src = torch.arange(windows, dtype=torch.int32, device=self.device)[:, None].repeat(1, 16)
            self._searches[key] = (cache, src)
        cache, src = self._searches[key]
        rows = torch.full((windows, 1), c.sot, dtype=torch.long, device=self.device)
        logits = self._prefill(windows, rows, cache, src)[:, 0]
        probs = logits[:, list(c.lang_ids)].softmax(-1).tolist()
        return [sorted(zip(c.languages, p), key=lambda x: -x[1]) for p in probs]

    # --- windows with faster-whisper's rules -----------------------------------------------------------
    @_on_stream
    def decode_windows(
        self,
        audio: list[AudioLike],
        language: str,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: float | None = 2.4,
        log_prob_threshold: float | None = -1.0,
        no_speech_threshold: float | None = 0.6,
        suppress_tokens: tuple[int, ...] | None = (-1,),
        suppress_blank: bool = True,
        batch_size: int = 8,
    ) -> list[Decoded]:
        """Up-to-30 s windows decoded on their own (no timestamps, no previous text), each as faster-whisper's
        ``generate_with_fallback`` decodes a window, batched: every temperature decodes the windows still
        unsettled at it together."""
        feats = self.features(audio)
        prompt = self.prompt(language, task, without_timestamps=True)
        n = feats.shape[0]
        attempts: list[list[dict]] = [[] for _ in range(n)]
        final: list[Decoded | None] = [None] * n
        pending = list(range(n))
        for temperature in temperatures:
            if not pending:
                break
            nxt = []
            for start in range(0, len(pending), batch_size):
                idx = pending[start : start + batch_size]
                cross = self.encode(feats[idx])
                results = self.generate(cross, prompt, beam_size, temperature, best_of, suppress_tokens,
                                        suppress_blank)
                for i, r in zip(idx, results):
                    text = self.tokenizer.decode(r.tokens).strip()
                    d = {"text": text, "tokens": r.tokens, "avg_logprob": r.hypothesis.avg_logprob,
                         "no_speech_prob": r.no_speech_prob, "compression_ratio": compression_ratio(text),
                         "temperature": temperature, "reached_max_length": r.reached_max_length}
                    attempts[i].append(d)
                    if not _needs_fallback(d, compression_ratio_threshold, log_prob_threshold,
                                           no_speech_threshold):
                        final[i] = _decoded(d, attempts[i][:-1])
                    else:
                        nxt.append(i)
            pending = nxt
        for i in pending:  # every temperature failed: the best below the compression threshold, else any
            ok = [a for a in attempts[i] if compression_ratio_threshold is None
                  or a["compression_ratio"] <= compression_ratio_threshold]
            best = max(ok or attempts[i], key=lambda a: a["avg_logprob"])
            final[i] = _decoded({**best, "temperature": attempts[i][-1]["temperature"]}, attempts[i])
        return final


def _needs_fallback(d: dict, cr: float | None, lp: float | None, ns: float | None) -> bool:
    needs = False
    if cr is not None and d["compression_ratio"] > cr:
        needs = True  # too repetitive
    if lp is not None and d["avg_logprob"] < lp:
        needs = True  # too unlikely
    if ns is not None and lp is not None and d["no_speech_prob"] > ns and d["avg_logprob"] < lp:
        needs = False  # silence
    return needs


def _decoded(d: dict, attempts: list[dict]) -> Decoded:
    return Decoded(d["text"], d["tokens"], d["avg_logprob"], d["no_speech_prob"], d["compression_ratio"],
                   d["temperature"], d["reached_max_length"], attempts)

