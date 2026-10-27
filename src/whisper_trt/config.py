"""Model configuration, read from a Hugging Face Whisper checkpoint directory (config.json and friends)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_FFT = 400
CHUNK_SECONDS = 30
N_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE  # 480000 samples per window
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 feature frames per window
TIME_PRECISION = 0.02  # seconds per timestamp token (and per encoder frame)


@dataclass(frozen=True)
class WhisperConfig:
    n_mels: int
    d_model: int
    enc_layers: int
    enc_heads: int
    enc_ffn: int
    dec_layers: int
    dec_heads: int
    dec_ffn: int
    vocab_size: int
    enc_frames: int = 1500  # encoder positions (30 s at 20 ms)
    max_target_positions: int = 448
    # special tokens (multilingual vocabularies; see tokenizer.py)
    eot: int = 50257
    sot: int = 50258
    translate: int = 50358
    transcribe: int = 50359
    sot_lm: int = 50360
    sot_prev: int = 50361
    no_speech: int = 50362
    no_timestamps: int = 50363
    timestamp_begin: int = 50364
    blank: int = 220
    lang_ids: tuple[int, ...] = ()
    languages: tuple[str, ...] = ()  # codes without <| |>, in the order of lang_ids
    suppress_tokens: tuple[int, ...] = ()  # generation_config's (the non-speech symbols)
    begin_suppress_tokens: tuple[int, ...] = (220, 50257)
    alignment_heads: tuple[tuple[int, int], ...] = field(default=())  # (layer, head) for word timestamps

    @property
    def multilingual(self) -> bool:
        return self.vocab_size >= 51865

    @property
    def head_dim(self) -> int:
        return self.d_model // self.dec_heads


def _token_id(vocab: dict[str, int], token: str) -> int:
    if token not in vocab:
        raise KeyError(f"token {token} is not in the vocabulary")
    return vocab[token]


def load_config(model_dir: str | Path) -> WhisperConfig:
    model_dir = Path(model_dir)
    hf = json.loads((model_dir / "config.json").read_text())
    if hf.get("model_type") != "whisper":
        raise ValueError(f"{model_dir} is not a Whisper checkpoint (model_type {hf.get('model_type')!r})")
    gen_path = model_dir / "generation_config.json"
    gen = json.loads(gen_path.read_text()) if gen_path.exists() else {}
    vocab = _special_tokens(model_dir)
    # Language tokens sit between <|startoftranscript|> and <|translate|>; their number differs by version
    # (99 in v1/v2, 100 in v3: <|yue|>).
    sot, translate = _token_id(vocab, "<|startoftranscript|>"), _token_id(vocab, "<|translate|>")
    by_id = {i: t for t, i in vocab.items()}
    lang_ids = tuple(range(sot + 1, translate))
    languages = tuple(by_id[i][2:-2] for i in lang_ids)
    no_speech = vocab.get("<|nospeech|>", vocab.get("<|nocaptions|>"))
    eot = _token_id(vocab, "<|endoftext|>")
    suppress = gen.get("suppress_tokens", hf.get("suppress_tokens")) or ()
    begin = gen.get("begin_suppress_tokens") or hf.get("begin_suppress_tokens") or (220, eot)
    heads = gen.get("alignment_heads") or ()
    return WhisperConfig(
        n_mels=hf["num_mel_bins"],
        d_model=hf["d_model"],
        enc_layers=hf["encoder_layers"],
        enc_heads=hf["encoder_attention_heads"],
        enc_ffn=hf["encoder_ffn_dim"],
        dec_layers=hf["decoder_layers"],
        dec_heads=hf["decoder_attention_heads"],
        dec_ffn=hf["decoder_ffn_dim"],
        vocab_size=hf["vocab_size"],
        enc_frames=hf.get("max_source_positions", 1500),
        max_target_positions=hf.get("max_target_positions", 448),
        eot=eot,
        sot=sot,
        translate=translate,
        transcribe=_token_id(vocab, "<|transcribe|>"),
        sot_lm=_token_id(vocab, "<|startoflm|>"),
        sot_prev=_token_id(vocab, "<|startofprev|>"),
        no_speech=no_speech,
        no_timestamps=_token_id(vocab, "<|notimestamps|>"),
        timestamp_begin=_token_id(vocab, "<|notimestamps|>") + 1,
        lang_ids=lang_ids,
        languages=languages,
        suppress_tokens=tuple(int(t) for t in suppress),
        begin_suppress_tokens=tuple(int(t) for t in begin),
        alignment_heads=tuple((int(a), int(b)) for a, b in heads),
    )


def _special_tokens(model_dir: Path) -> dict[str, int]:
    """Added (special) tokens: token -> id, from added_tokens.json or tokenizer.json."""
    added = model_dir / "added_tokens.json"
    if added.exists():
        return json.loads(added.read_text())
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    return {t["content"]: t["id"] for t in tok["added_tokens"]}
