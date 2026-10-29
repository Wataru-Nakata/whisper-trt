"""The public API: :class:`Whisper` (``from_pretrained``, ``generate``, ``decode_windows``, ``detect_language``)."""

import functools
import hashlib
import logging
import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch

from . import hub
from .audio import AudioLike, load_audio
from .config import N_FRAMES, SAMPLE_RATE, TIME_PRECISION, WhisperConfig, load_config
from .features import log_mel
from .modeling import load_model
from .runtime import TorchBackend, TritonDecoderBackend, TRTBackend
from .search import Search, WindowResult
from .segments import split_segments
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
    # decoded with timestamps: faster-whisper's segments of the window ({"start", "end", "text", "tokens"}, s)
    segments: list[dict] = field(default_factory=list)


def _on_stream(fn):
    """Run a method's GPU work on the model's own stream (TensorRT synchronizes extra on the default one),
    ordered after the caller's work on its stream, and the caller's later work after it (the returned tensors
    are written on the model's stream)."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        outer = torch.cuda.current_stream()
        self.stream.wait_stream(outer)  # inputs the caller produced
        with torch.cuda.stream(self.stream):
            result = fn(self, *args, **kwargs)
        outer.wait_stream(self.stream)  # results the caller will read
        return result

    return wrapper


# the layers of the FP8 encoder (fp8.py) by default: the MLPs'; the attention and cross-attention projections in
# FP8 too change many more decodes for little more speed (README)
FP8_LAYERS = r"^encoder\.layers\.\d+\.fc[12]$"


class Whisper:
    def __init__(self, cfg: WhisperConfig, tokenizer: Tokenizer, backend, model=None, max_batch: int = 16):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.backend = backend
        self.model = model  # the PyTorch model (torch backend, word timestamps)
        self.max_batch = max_batch
        # windows per encoder call (the TensorRT engine's largest profile)
        trt = backend if hasattr(backend, "profiles") else getattr(backend, "encoder", None)
        profiles = getattr(trt, "profiles", None)
        self.encode_batch = max(p["batch"][2] for p in profiles["encoder"]) if profiles else max_batch
        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream(priority=-1)  # (decoding first)
        self.side_stream = torch.cuda.Stream(priority=0)  # windows encoded ahead (_Prefetch)
        self._cross_flat = torch.empty(0, dtype=backend.dtype, device=self.device)  # (grown on demand: cross)
        self._searches: dict[tuple, Search] = {}
        self._prefills: dict[tuple, tuple] = {}
        self._aligners: dict[int, object] = {}

    @classmethod
    def from_pretrained(
        cls,
        name: str = "openai/whisper-large-v3",
        backend: str = "tensorrt",
        dtype: str = "float16",
        max_batch: int = 32,
        max_rows: int = 160,
        cache_dir: str | None = None,
        decoder: str | None = None,
        fp8: bool | str = False,
        fp8_layers: str = FP8_LAYERS,
        **build_options,
    ) -> "Whisper":
        """``backend="tensorrt"``: the TensorRT encoder, and a decoder step of cuBLAS GEMMs and Triton kernels
        (``decoder="triton"``, runtime.TritonDecoderBackend: float16; only the encoder engine is built) or the
        TensorRT decoder engine (``decoder="tensorrt"``: slower, and its GEMMs accumulate less precisely).
        ``decoder=None``: "triton" for float16, else "tensorrt".

        ``fp8`` (opt-in, Hopper or newer): the encoder's GEMMs in FP8 (fp8.py), with the input ranges of a scale
        file: True takes the one the package ships for this checkpoint, a path that file; ``fp8_layers``: a regular
        expression on the names (modeling's, e.g. ``encoder.layers.3.fc1``) of the layers made FP8 (by default
        the MLPs': the best trade-off measured). Faster encoding, different decodes (see the README)."""
        model_dir = hub.resolve(name)
        cfg = load_config(model_dir)
        tokenizer = Tokenizer(model_dir, cfg)
        if backend == "torch":
            model = load_model(model_dir, cfg, dtype=DTYPES[dtype])
            return cls(cfg, tokenizer, TorchBackend(model), model, max_batch)
        if backend != "tensorrt":
            raise ValueError(f"backend must be 'tensorrt' or 'torch', got {backend!r}")
        if decoder is None:
            decoder = "triton" if dtype == "float16" else "tensorrt"
        if decoder not in ("tensorrt", "triton"):
            raise ValueError(f"decoder must be 'tensorrt' or 'triton', got {decoder!r}")
        if decoder == "triton" and dtype != "float16":
            raise ValueError("the Triton decoder runs float16")
        from . import fp8 as fp8_mod
        from .build import BuildOptions, ensure_engines

        amax = None
        if fp8:
            if isinstance(fp8, (str, Path)):
                path = Path(fp8)
            else:
                path = fp8_mod.packaged_scales(fp8_mod.checkpoint_sha256(next(model_dir.glob("model*.safetensors"))))
            if path is None:
                raise ValueError(f"no FP8 scale file ships for {name}: pass fp8=PATH (scripts/fp8_calibrate.py)")
            amax = {k: v for k, v in fp8_mod.load_scales(path)["amax"].items() if re.search(fp8_layers, k)}
            if not amax:
                raise ValueError(f"fp8_layers={fp8_layers!r} matches none of the scale file's layers")
            build_options["fp8_scales"] = (f"s{fp8_mod.scales_digest(path)}-l"
                                           f"{hashlib.sha1(fp8_layers.encode()).hexdigest()[:8]}")
        opts = BuildOptions(dtype=dtype, max_batch=max_batch, max_rows=max_rows, **build_options)
        engine_dir = ensure_engines(model_dir, cfg, opts, cache_dir, decoder=decoder == "tensorrt", fp8_amax=amax)
        backend_ = TRTBackend(cfg, engine_dir, decoder=decoder == "tensorrt")
        if decoder == "triton":
            model = load_model(model_dir, cfg, device="cpu", dtype=torch.float16)
            del model.encoder
            backend_ = TritonDecoderBackend(model.to("cuda"), backend_)
        return cls(cfg, tokenizer, backend_, None, max_batch)

    # --- building blocks --------------------------------------------------------------------------------
    @_on_stream
    def features(self, audio: list[AudioLike]) -> torch.Tensor:
        """[W, n_mels, 3000] features of up-to-30 s windows (each normalized on its own, like faster-whisper's
        batched pipeline); an item that already is features ([n_mels, 3000], e.g. a window of a long
        recording's) is taken as it is."""
        return self._features(audio)

    def _features(self, audio: list[AudioLike]) -> torch.Tensor:
        """:meth:`features` on the current stream."""
        if all(isinstance(a, torch.Tensor) and a.dim() == 2 for a in audio):
            return torch.stack([a.to(self.device, torch.float32) for a in audio])
        if any(isinstance(a, torch.Tensor) and a.dim() == 2 for a in audio):
            return torch.cat([self._features([a]) for a in audio])
        waves = [a if isinstance(a, torch.Tensor) and a.dim() == 1 else load_audio(a) for a in audio]
        return log_mel(waves, self.cfg.n_mels, self.device, N_FRAMES)

    def cross(self, windows: int) -> torch.Tensor:
        """The cross-attention buffer [L, 2, W, H, T, d_k] for ``windows`` windows (one buffer for all W, grown
        to the most windows asked for: 245 MB each for large-v3; growing drops the searches bound to it)."""
        c = self.cfg
        shape = (c.dec_layers, 2, windows, c.dec_heads, c.enc_frames, c.head_dim)
        n = int(np.prod(shape))
        if n > self._cross_flat.numel():
            self._searches.clear()
            self._prefills.clear()
            self._cross_flat = torch.empty(0, dtype=self.backend.dtype, device=self.device)
            torch.cuda.empty_cache()
            self._cross_flat = torch.empty(n, dtype=self.backend.dtype, device=self.device)
        return self._cross_flat[:n].view(shape)

    @_on_stream
    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """Every decoder layer's cross-attention keys/values of the windows' features, as a view of the model's
        buffer (valid until a call with more windows grows it)."""
        if feats.shape[0] > self.encode_batch:
            raise ValueError(f"at most {self.encode_batch} windows per call")
        cross = self.cross(feats.shape[0])
        self.backend.encode(feats.to(self.device, torch.float32).contiguous(), cross)
        return cross

    def _search(self, mode: str, windows: int, rows: int, prompt_length: int | None, timestamps: bool = False,
                mixed: bool = False) -> Search:
        """``mixed``: windows sampled at a temperature in the same steps (Search's ``sampling_slots``)."""
        key = (mode, windows, rows, prompt_length, timestamps, mixed)
        if key not in self._searches:
            step = partial(self.backend.step_fn, cross=self.cross(windows))
            self._searches[key] = Search(self.cfg, step, windows, rows, prompt_length, mode,
                                         dtype=self.backend.dtype, device=self.device, timestamps=timestamps,
                                         sampling_slots=mixed)
        return self._searches[key]

    def _aligner(self, decoder):
        """The word alignment's alignment.Aligner for ``decoder`` (on the side stream; graphs captured once)."""
        from .alignment import Aligner

        if id(decoder) not in self._aligners:
            aligner = Aligner(decoder, self.cfg, self.device, self.side_stream)
            aligner.prepare()
            self._aligners[id(decoder)] = aligner
        return self._aligners[id(decoder)]

    def _timestamps(self, prompt: list[int]) -> bool:
        """Whether a prompt decodes with timestamps: as CTranslate2 decides, unless it ends with
        <|notimestamps|>."""
        return prompt[-1] != self.cfg.no_timestamps

    def _decoder(self, key: tuple, tokens: torch.Tensor, cache: torch.Tensor, src: torch.Tensor,
                 cross: torch.Tensor) -> torch.Tensor:
        """One decoder call of ``tokens`` [n, p] at positions 0..p-1 against ``cache`` (written there) and
        ``cross``; logits [n, p, V]. The call's buffers and context are kept under ``key``."""
        n, p = tokens.shape
        if key not in self._prefills:
            c = self.cfg
            buf = torch.zeros(n, p, dtype=torch.long, device=self.device)
            offset = torch.zeros(n, dtype=torch.long, device=self.device)
            logits = torch.zeros(n, p, c.vocab_size, dtype=torch.float32, device=self.device)
            new_kv = torch.zeros(c.dec_layers, 2, n, c.dec_heads, p, c.head_dim, dtype=self.backend.dtype,
                                 device=self.device)
            fn = self.backend.step_fn(buf, offset, cache, logits, new_kv, src, cross=cross)
            self._prefills[key] = (fn, buf, logits, new_kv)  # (fn keeps every bound tensor alive)
        fn, buf, logits, new_kv = self._prefills[key]
        buf.copy_(tokens)
        fn()
        if not getattr(fn, "writes_cache", False):
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
        seed: int = 0,
        max_initial_timestamp_index: int = 50,
    ) -> list[WindowResult]:
        """CTranslate2's ``Whisper.generate`` on encoded windows (``cross`` from :meth:`encode`), as
        faster-whisper calls it: beam search at temperature 0 (greedy for ``beam_size=1``), else ``best_of``
        samples (``seed``: their random numbers; the same seed, the same samples). ``prompt``: one for all
        windows or one per window (all of the same length); without <|notimestamps|> at its end the decodes
        have timestamps (CTranslate2's timestamp rules; the first at most ``max_initial_timestamp_index``)."""
        windows = cross.shape[2]
        prompts = [prompt] * windows if not prompt or isinstance(prompt[0], int) else prompt
        if len({len(p) for p in prompts}) != 1:
            raise ValueError("every window's prompt must have the same length")
        if temperature > 0:
            mode, rows = "sample", best_of
        else:
            mode, rows = ("beam", beam_size) if beam_size > 1 else ("greedy", 1)
        search = self._search(mode, windows, rows, len(prompts[0]), self._timestamps(prompts[0]))
        suppress = self.tokenizer.suppressed_tokens(suppress_tokens)
        begin = self.cfg.begin_suppress_tokens if suppress_blank else ()
        rows_t = torch.tensor(prompts, dtype=torch.long)
        return search.run(rows_t, list(suppress), list(begin), temperature=temperature if temperature > 0 else 1.0,
                          seed=seed, max_initial_timestamp_index=max_initial_timestamp_index)

    @_on_stream
    def decode(
        self,
        audio: list[AudioLike],
        prompt: list[int] | list[list[int]],
        beam_size: int = 5,
        temperature: float = 0.0,
        best_of: int = 5,
        batch_size: int = 8,
        suppress_tokens: tuple[int, ...] | None = (-1,),
        suppress_blank: bool = True,
        seed: int = 0,
        max_initial_timestamp_index: int = 50,
    ) -> list[WindowResult]:
        """:meth:`generate` for any number of up-to-30 s windows, ``batch_size`` decoding at a time: a slot
        whose window is done takes the next window (its features computed and encoded then), so no window
        waits for the slowest of a batch. ``prompt``: one for every window, or one per window; prompts with
        previous text, or of different lengths, are run in one decoder call before their window's decode (as
        CTranslate2 runs them), the others token by token in its first steps."""
        if not audio:
            return []
        prompts = [prompt] * len(audio) if not prompt or isinstance(prompt[0], int) else list(prompt)
        timestamps = {self._timestamps(p) for p in prompts}
        if len(timestamps) != 1:
            raise ValueError("every window decodes with timestamps, or none does")
        prefilled = len({len(p) for p in prompts}) > 1 or any(p.index(self.cfg.sot) > 0 for p in prompts)
        if temperature > 0:
            mode, rows = "sample", best_of
        else:
            mode, rows = ("beam", beam_size) if beam_size > 1 else ("greedy", 1)
        slots = min(batch_size, len(audio))
        if slots > self.max_batch:
            raise ValueError(f"at most {self.max_batch} windows at a time")
        search = self._search(mode, slots, rows, None if prefilled else len(prompts[0]), timestamps.pop())
        cross = self.cross(slots)
        at_once = slots <= self.encode_batch  # the first windows in one encoder call (else one by one, ahead)
        ahead = _Prefetch(self, audio, ring=min(slots, len(audio) - slots, 8) if at_once else min(len(audio), 8))

        def load(slot_ids: list[int], windows: list[int]) -> None:
            if at_once and slot_ids == list(range(slots)) and windows == list(range(slots)):
                self.backend.encode(self._features([audio[w] for w in windows]), cross)
                ahead.skip_to(slots)
            else:
                for s_, w in zip(slot_ids, windows):
                    ahead.take(w, cross[:, :, s_])
            ahead.top_up()

        suppress = self.tokenizer.suppressed_tokens(suppress_tokens)
        begin = self.cfg.begin_suppress_tokens if suppress_blank else ()
        return search.stream(prompts, load, list(suppress), list(begin),
                             temperature=temperature if temperature > 0 else 1.0, seed=seed,
                             max_initial_timestamp_index=max_initial_timestamp_index,
                             prefill=self._prefill(search, cross) if prefilled else None)

    def _prefill(self, search: Search, cross: torch.Tensor) -> Callable[[list[int], list[list[int]]], None]:
        """Search.stream's prefill: each window's prompt but its last token in one decoder call (its slot's
        keys/values), the keys/values into the slot's first row of the cache, the no-speech probability from the
        logits at <|startoftranscript|>."""
        c, K, dev = self.cfg, search.K, self.device
        cache = torch.zeros(c.dec_layers, 2, 1, c.dec_heads, 16, c.head_dim, dtype=self.backend.dtype, device=dev)
        src = torch.zeros(1, 16, dtype=torch.int32, device=dev)
        offset = torch.zeros(1, dtype=torch.long, device=dev)

        limit = getattr(self.backend, "max_prefill", None) if getattr(self.backend, "decoder", None) else None

        def prefill(slots: list[int], prompts: list[list[int]]) -> None:
            for s, p in zip(slots, prompts):
                n = len(p) - 1
                if limit is not None and n > limit:
                    raise NotImplementedError(f"a prompt of {n + 1} tokens: the TensorRT decoder engine prefills at "
                                              f"most {limit}; prompts with previous text need the Triton decoder "
                                              "(the default) or the PyTorch backend")
                tokens = torch.tensor([p[:-1]], dtype=torch.long, pin_memory=True).to(dev, non_blocking=True)
                logits = torch.empty(1, n, c.vocab_size, dtype=torch.float32, device=dev)
                new_kv = torch.empty(c.dec_layers, 2, 1, c.dec_heads, n, c.head_dim, dtype=self.backend.dtype,
                                     device=dev)
                self.backend.step_fn(tokens, offset, cache, logits, new_kv, src, cross=cross[:, :, s : s + 1])()
                search.cache[:, :, s * K, :, :n] = new_kv[:, :, 0]
                lg = logits[0, p.index(c.sot)]
                search.no_speech[s : s + 1] = (lg[c.no_speech] - lg.logsumexp(-1)).exp()

        return prefill

    @_on_stream
    def detect_language(self, cross: torch.Tensor) -> list[list[tuple[str, float]]]:
        """Language probabilities of encoded windows: the softmax of the language tokens' logits after
        <|startoftranscript|> (CTranslate2's ``detect_language``), best first."""
        windows = cross.shape[2]
        c = self.cfg
        key = ("lid", windows)
        if key not in self._searches:
            # a one-token call against a small cache
            cache = torch.zeros(c.dec_layers, 2, windows, c.dec_heads, 16, c.head_dim, dtype=self.backend.dtype,
                                device=self.device)
            src = torch.arange(windows, dtype=torch.int32, device=self.device)[:, None].repeat(1, 16)
            self._searches[key] = (cache, src)
        cache, src = self._searches[key]
        rows = torch.full((windows, 1), c.sot, dtype=torch.long, device=self.device)
        logits = self._decoder(key, rows, cache, src, self.cross(windows))[:, 0]
        probs = logits[:, list(c.lang_ids)].softmax(-1).tolist()
        return [sorted(zip(c.languages, p), key=lambda x: -x[1]) for p in probs]

    def _torch_decoder(self):
        """The PyTorch decoder on the model's weights (word timestamps)."""
        dec = getattr(self.backend, "torch_decoder", None)
        if dec is None and getattr(self.backend, "model", None) is not None:
            dec = self.backend.model.decoder
        if dec is None:
            raise NotImplementedError("word timestamps need the Triton decoder or the PyTorch backend")
        return dec

    # --- long-form, with faster-whisper's rules -----------------------------------------------------------
    @_on_stream
    def transcribe(self, audio: list[AudioLike], language: str | None = None, task: str = "transcribe",
                   **options):
        """Recordings of any length transcribed as faster-whisper's ``WhisperModel.transcribe`` transcribes
        each (``longform.transcribe``: window by window from a seek position, conditioned on the previous text,
        with the temperature fallback, timestamps and optionally word timestamps), the windows of all the
        recordings streaming through ``batch_size`` slots: a recording's next window starts as soon as its
        previous one is decided. ``language=None``: each recording's detected from its first 30 s (as
        faster-whisper's ``detect_language``). Returns each recording's segments (a longform.Transcript: a list
        of longform.Segment, with ``words`` if ``word_timestamps=True``, and its ``language``). Options: those
        of faster-whisper's (beam_size, best_of, temperatures, compression_ratio_threshold, log_prob_threshold,
        no_speech_threshold, condition_on_previous_text, prompt_reset_on_temperature, initial_prompt,
        suppress_blank, suppress_tokens, without_timestamps, max_initial_timestamp, word_timestamps,
        prepend_punctuations, append_punctuations, hallucination_silence_threshold,
        language_detection_threshold, language_detection_segments), batch_size, ``kv_pool`` (windows whose
        keys/values are kept for their retries and alignment, default ``min(recordings, batch_size + 8)``),
        seed (a window's samples depend on it, the recording's index, the window's and the temperature's
        only)."""
        from .longform import transcribe

        return transcribe(self, audio, language, task, **options)

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
        seed: int = 0,
        without_timestamps: bool = True,
        max_initial_timestamp: float = 1.0,
    ) -> list[Decoded]:
        """Up-to-30 s windows decoded on their own (no previous text), each as faster-whisper's
        ``generate_with_fallback`` decodes a window, batched: every temperature decodes the windows still
        unsettled at it together, ``batch_size`` at a time with refilled slots (:meth:`decode`). ``seed``: the
        sampling's random numbers (the windows' order and the seed give the same samples).
        ``without_timestamps=False``: decoded with timestamps (the first at most ``max_initial_timestamp``
        seconds), each result with faster-whisper's segments of its window."""
        waves = [a if isinstance(a, torch.Tensor) and a.dim() == 1 else load_audio(a) for a in audio]
        prompt = self.prompt(language, task, without_timestamps=without_timestamps)
        final = self._fallback(waves, [prompt] * len(waves), beam_size, best_of, temperatures,
                               compression_ratio_threshold, log_prob_threshold, no_speech_threshold, suppress_tokens,
                               suppress_blank, batch_size, seed, int(round(max_initial_timestamp / TIME_PRECISION)))
        if not without_timestamps:
            for i, d in enumerate(final):
                duration = min(len(waves[i]) / SAMPLE_RATE, N_FRAMES / 100)
                d.segments = [{**seg, "text": self.tokenizer.decode(seg["tokens"])} for seg in
                              split_segments(d.tokens, self.cfg.timestamp_begin, duration)]
        return final

    def _fallback(self, items: list, prompts: list[list[int]], beam_size: int, best_of: int,
                  temperatures: tuple[float, ...], compression_ratio_threshold: float | None,
                  log_prob_threshold: float | None, no_speech_threshold: float | None,
                  suppress_tokens: tuple[int, ...] | None, suppress_blank: bool, batch_size: int, seed: int,
                  max_initial: int) -> list[Decoded]:
        """faster-whisper's ``generate_with_fallback`` for windows (waves or features) with their prompts: every
        temperature decodes the windows still unsettled at it together (:meth:`decode`)."""
        n = len(items)
        attempts: list[list[dict]] = [[] for _ in range(n)]
        final: list[Decoded | None] = [None] * n
        pending = list(range(n))
        for round_, temperature in enumerate(temperatures):
            if not pending:
                break
            results = self.decode([items[i] for i in pending], [prompts[i] for i in pending], beam_size,
                                  temperature, best_of, batch_size, suppress_tokens, suppress_blank, seed=seed + round_,
                                  max_initial_timestamp_index=max_initial)
            nxt = []
            for i, r in zip(pending, results):
                text = self.tokenizer.decode(r.tokens).strip()
                d = {"text": text, "tokens": r.tokens, "avg_logprob": r.hypothesis.avg_logprob,
                     "no_speech_prob": r.no_speech_prob, "compression_ratio": compression_ratio(text),
                     "temperature": temperature, "reached_max_length": r.reached_max_length}
                attempts[i].append(d)
                if not _needs_fallback(d, compression_ratio_threshold, log_prob_threshold, no_speech_threshold):
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


class _Prefetch:
    """The next windows of a :meth:`Whisper.decode`, encoded ahead on a stream of their own (while the decoder
    steps run, which leave much of the GPU idle) into a ring of one-window buffers; ``take`` copies one into a
    slot when the slot is free."""

    def __init__(self, model: "Whisper", audio: list, ring: int):
        c = model.cfg
        self.model, self.audio = model, audio
        shape = (c.dec_layers, 2, 1, c.dec_heads, c.enc_frames, c.head_dim)
        self.bufs = [torch.empty(shape, dtype=model.backend.dtype, device=model.device) for _ in range(ring)]
        self.free = list(range(ring))
        self.copied: list = [None] * ring  # event after a buffer's copy into its slot (the model's stream)
        self.staged: dict[int, tuple[int, torch.cuda.Event]] = {}  # window -> (buffer, encoded event)
        self.next = 0  # the next window to encode

    def skip_to(self, window: int) -> None:
        self.next = window

    def top_up(self) -> None:
        """Encode the next windows into the free buffers, one at a time, on the side stream."""
        m = self.model
        while self.free and self.next < len(self.audio):
            b, w = self.free.pop(0), self.next
            self.next += 1
            with torch.cuda.stream(m.side_stream):
                if self.copied[b] is not None:
                    m.side_stream.wait_event(self.copied[b])
                m.backend.encode(m._features([self.audio[w]]), self.bufs[b], lane=1)
                done = torch.cuda.Event()
                done.record()
            self.staged[w] = (b, done)

    def take(self, window: int, slot: torch.Tensor) -> None:
        """Window ``window``'s keys/values into ``slot`` (a [L, 2, H, T, d_k] view), on the current stream."""
        if window not in self.staged:  # (not encoded ahead: the ring was full)
            self.top_up()
        b, done = self.staged.pop(window)
        stream = torch.cuda.current_stream()
        stream.wait_event(done)
        slot.copy_(self.bufs[b][:, :, 0])
        self.copied[b] = torch.cuda.Event()
        self.copied[b].record(stream)
        self.free.append(b)


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

