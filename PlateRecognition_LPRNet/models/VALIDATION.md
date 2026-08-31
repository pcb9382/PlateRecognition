# Export validation record

Validation was run with the user-provided environment at
`P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe`:

- Python 3.11.9
- PyTorch 2.7.1+cpu
- ONNX 1.19.1
- ONNX Runtime 1.23.0 (CPUExecutionProvider)
- OpenCV 4.11.0
- NumPy 1.26.4

No installed package was upgraded or downgraded.

Results:

- The official PyTorch checkpoint loaded strictly into the 68-class network.
- The export-compatible pool rewrite was bit-for-bit equal to the untouched
  upstream 4-D `MaxPool3d` implementation on both deterministic export inputs,
  one validator random input, and 20 official repository test images.
- `onnx.checker.check_model(..., full_check=True)` passed.
- ONNX Runtime returned `[1,68,18]`; PyTorch and ONNX Runtime had zero
  per-time-step argmax differences and identical decoded strings on all 21
  validator samples.
- The deterministic export check observed maximum absolute logit error
  `0.000152587891` and mean absolute error `0.0000270477503`.
- Repeating the complete export produced the same ONNX SHA-256:
  `150aed71b7b543926b45cea905c19f68e97862d8c05b0da09452e5bc2de22426`.

A TensorRT engine was not generated on the Windows export host because the
project's TensorRT 8.2.5.1 build/runtime is a Linux target. The module uses the
same common TensorRT runtime and exact SDK version guard as OCRv5/OCRv6; final
FP32/FP16 engine execution should be validated on that target GPU.

