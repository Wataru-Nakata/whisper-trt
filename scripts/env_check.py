"""Print the versions and GPU of the job environment."""

import sys

import tensorrt
import torch

print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "trt", tensorrt.__version__)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
for name in ("tokenizers", "safetensors", "onnx", "onnxscript", "triton", "soundfile", "huggingface_hub", "numpy"):
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "?"), getattr(mod, "__file__", ""))
    except Exception as e:  # noqa: BLE001
        print(name, "MISSING", e)
