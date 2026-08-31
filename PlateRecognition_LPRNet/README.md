# PlateRecognition LPRNet

This module converts the official `sirius-ai/LPRNet_Pytorch` checkpoint to a
static ONNX model and runs it through the project's common TensorRT 8.2.5.1
runtime. It exposes both a C++ class and a C-compatible opaque-handle API.

## Model contract

- Input: `input`, float32 NCHW `[1,3,24,94]`
- Image order: BGR, resized to `94x24`
- Normalization: `(pixel - 127.5) * 0.0078125`
- Output: `logits`, float32 `[1,68,18]`
- Decode: greedy CTC; class 67 (`-`) is blank
- Characters: all 67 non-blank classes from the official project, including
  `I` and `O`; no `PlateRecognition` whitelist is applied

## Re-export and validate

Use the existing project Python environment without upgrading or downgrading
its packages:

```powershell
& 'P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe' `
  .\PlateRecognition_LPRNet\tools\export_lprnet_onnx.py

& 'P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe' `
  .\PlateRecognition_LPRNet\tools\validate_lprnet.py --images <plate.jpg>
```

The exporter checks the official checkpoint SHA-256, verifies the 4-D
`MaxPool3d` compatibility rewrite against the untouched upstream model, runs
the ONNX full checker, and compares ONNX Runtime logits. The validator also
requires identical per-time-step argmax and decoded strings.

## Build

The parent project supplies `TensorRT_INCLUDE` and `TensorRT_LIB`; standalone
configuration defaults to the same TensorRT 8.2.5.1 location as the root
project.

```bash
cmake -S . -B build \
  -DTensorRT_INCLUDE=/home/pcb/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/include \
  -DTensorRT_LIB=/home/pcb/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/lib
cmake --build build --target PlateRecognition_LPRNet_demo
```

TensorRT engines are generated on first initialization. FP32 is the default;
set `enable_fp16=true` only after validating the target GPU's result. Engines
must be rebuilt when TensorRT/CUDA or GPU architecture changes.

## C++ API

```cpp
PlateRecognitionLPRNetConfig config;
config.model_path = "PlateRecognition_LPRNet/models/lprnet_official_68.onnx";
config.gpu_id = 0;

PlateRecognitionLPRNet recognizer;
if (recognizer.Init(config) != HZ_SUCCESS) {
    // recognizer.LastError()
}
std::string plate;
recognizer.Run(bgr_image, plate);
recognizer.Release();
```

The aliases `PlateRecognitionLPRNetInit`, `PlateRecognitionLPRNetRun`, and
`PlateRecognitionLPRNetRelease` match the existing `PlateRecognition` naming
style. Every `Run` clears its output string first. The library does not print;
errors are available through `LastError()`.

For C callers include `PlateRecognitionLPRNet_c.h`, allocate a handle with
`plate_lprnet_create`, initialize a `PlateRecognitionLPRNetCConfig` with
`struct_size = sizeof(PlateRecognitionLPRNetCConfig)`, then use
`plate_lprnet_run`. The returned text is UTF-8.

Third-party provenance and license details are in
`third_party/LPRNet_Pytorch/SOURCE.md` and `third_party/LPRNet_Pytorch/LICENSE`.

