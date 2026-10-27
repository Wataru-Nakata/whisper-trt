"""Checkpoint lookup: a local directory, or a Hugging Face repository (downloaded once)."""

from pathlib import Path

FILES = ["config.json", "generation_config.json", "tokenizer.json", "added_tokens.json", "model*.safetensors",
         "preprocessor_config.json", "vocab.json", "merges.txt", "normalizer.json", "special_tokens_map.json",
         "tokenizer_config.json"]


def resolve(name: str | Path) -> Path:
    path = Path(name).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(name), allow_patterns=FILES))
