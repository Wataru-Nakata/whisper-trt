"""List TensorRT API members relevant to precision (builder flags, matmul layer attributes)."""
import tensorrt as trt

print(trt.__version__)
print("BuilderFlag:", [m for m in dir(trt.BuilderFlag) if not m.startswith("_")])
print("IMatrixMultiplyLayer:", [m for m in dir(trt.IMatrixMultiplyLayer) if not m.startswith("_")])
print("ILayer:", [m for m in dir(trt.ILayer) if not m.startswith("_")])
print("TacticSource:", [m for m in dir(trt.TacticSource) if not m.startswith("_")])
print("PreviewFeature:", [m for m in dir(trt.PreviewFeature) if not m.startswith("_")])
