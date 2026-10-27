"""Whisper's byte-level BPE tokenizer (the checkpoint's tokenizer.json, through the `tokenizers` library).

The text-side rules follow openai-whisper / faster-whisper 1.2.1 (both MIT): decoding drops special tokens,
the non-speech tokens are the ones Whisper suppresses by default, and words are split at spaces (or at
unicode boundaries for languages written without spaces) for word timestamps.
"""

import string
from functools import cached_property
from pathlib import Path

from .config import WhisperConfig

NO_SPACE_LANGUAGES = {"zh", "ja", "th", "lo", "my", "yue"}


class Tokenizer:
    def __init__(self, model_dir: str | Path, cfg: WhisperConfig):
        import tokenizers

        self.tokenizer = tokenizers.Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.cfg = cfg
        self.eot = cfg.eot
        self.timestamp_begin = cfg.timestamp_begin

    def language_id(self, language: str) -> int:
        token = self.tokenizer.token_to_id(f"<|{language}|>")
        if token is None or token not in self.cfg.lang_ids:
            raise ValueError(f"unknown language {language!r}; one of {', '.join(self.cfg.languages)}")
        return token

    def sot_sequence(self, language: str | None, task: str = "transcribe") -> list[int]:
        """<|startoftranscript|> [<|lang|> <|task|>] (multilingual models)."""
        if not self.cfg.multilingual:
            return [self.cfg.sot]
        if task not in ("transcribe", "translate"):
            raise ValueError(f"task must be 'transcribe' or 'translate', got {task!r}")
        task_id = self.cfg.transcribe if task == "transcribe" else self.cfg.translate
        return [self.cfg.sot, self.language_id(language or "en"), task_id]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode([t for t in tokens if t < self.eot])

    def decode_with_timestamps(self, tokens: list[int]) -> str:
        outputs: list = [[]]
        for token in tokens:
            if token >= self.timestamp_begin:
                outputs.append(f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>")
                outputs.append([])
            else:
                outputs[-1].append(token)
        return "".join(s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs)

    @cached_property
    def non_speech_tokens(self) -> tuple[int, ...]:
        """Tokens of speaker tags and non-speech annotations (♪♪♪, ( SPEAKING ... ), [DAVID] ...), keeping basic
        punctuation."""
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        # U+2640..U+267F symbols share their first two UTF-8 bytes: suppressing the first token is safe
        miscellaneous = set("♩♪♫♬♭♮♯")
        # hyphens and single quotes are allowed between words, not at the beginning of one
        result = {self.encode(" -")[0], self.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in (self.encode(symbol), self.encode(" " + symbol)):
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])
        return tuple(sorted(result))

    def suppressed_tokens(self, suppress_tokens: tuple[int, ...] | None = (-1,)) -> tuple[int, ...]:
        """faster-whisper's ``get_suppressed_tokens``: -1 stands for the non-speech tokens; the task, start and
        no-speech tokens are always suppressed."""
        out = [t for t in suppress_tokens or () if t >= 0]
        if suppress_tokens and -1 in suppress_tokens:
            out += self.non_speech_tokens
        c = self.cfg
        out += [c.transcribe, c.translate, c.sot, c.sot_prev, c.sot_lm, c.no_speech]
        return tuple(sorted(set(out)))

    def split_to_word_tokens(self, tokens: list[int], language: str) -> tuple[list[str], list[list[int]]]:
        if language in NO_SPACE_LANGUAGES:
            return self.split_tokens_on_unicode(tokens)
        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens: list[int]) -> tuple[list[str], list[list[int]]]:
        decoded_full = self.decode_with_timestamps(tokens)
        replacement = "�"
        words, word_tokens, current, offset = [], [], [], 0
        for token in tokens:
            current.append(token)
            decoded = self.decode_with_timestamps(current)
            index = decoded.find(replacement)
            index = None if index < 0 else index + offset
            if index is None or (index < len(decoded_full) and decoded_full[index] == replacement):
                words.append(decoded)
                word_tokens.append(current)
                current = []
                offset += len(decoded)
        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: list[int]) -> tuple[list[str], list[list[int]]]:
        subwords, subword_tokens = self.split_tokens_on_unicode(tokens)
        words, word_tokens = [], []
        for subword, toks in zip(subwords, subword_tokens):
            special = toks[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or not words:
                words.append(subword)
                word_tokens.append(toks)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(toks)
        return words, word_tokens
