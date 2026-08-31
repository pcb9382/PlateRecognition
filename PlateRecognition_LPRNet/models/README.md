# LPRNet model files

| File | Origin | SHA-256 |
| --- | --- | --- |
| `Final_LPRNet_model.pth` | Official `sirius-ai/LPRNet_Pytorch` checkpoint at commit `7c976664b3f3879efabeaff59c7a117e49d5f29e` | `d1ca22ca7a251a90702a166c8ad8ce7f05acb39f5ef8055139a33d462716d1ad` |
| `lprnet_official_68.onnx` | Reproducible export by `tools/export_lprnet_onnx.py` | `150aed71b7b543926b45cea905c19f68e97862d8c05b0da09452e5bc2de22426` |

The ONNX model uses IR version 7 and default-domain opset 13. It is static
batch 1: input `input`, float32 `[1,3,24,94]`; output `logits`, float32
`[1,68,18]`. ONNX Runtime and TensorRT use NCHW BGR input with the official
preprocessing `(pixel - 127.5) * 0.0078125`.

TensorRT engines are intentionally not stored here. They depend on the exact
TensorRT/CUDA stack and GPU architecture. The C++ runtime builds and caches a
TensorRT 8.2.5.1 FP32 or FP16 engine beside the ONNX model (or at the explicitly
configured cache path).

See `VALIDATION.md` for the recorded PyTorch/ONNX numerical checks and the
remaining target-side TensorRT validation step.
