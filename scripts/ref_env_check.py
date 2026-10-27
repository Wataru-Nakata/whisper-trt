"""Versions of the reference (faster-whisper) environment."""

import ctranslate2
import faster_whisper

print("ctranslate2", ctranslate2.__version__, "cuda devices", ctranslate2.get_cuda_device_count())
print("faster_whisper", faster_whisper.__version__)
print(ctranslate2.get_supported_compute_types("cuda"))
