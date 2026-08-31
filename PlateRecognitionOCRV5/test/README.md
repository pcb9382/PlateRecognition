# PP-OCRv5 测试说明

本目录包含三类测试，测试结论不能混为一谈：

1. `PlateRecognitionOCRV5_decode_test` 是不加载模型的 C++ 单元测试，验证
   CTC 去重、blank、固化类别映射和 77 字符输出规则。
2. `recognition_effect_test.py` 使用 **ONNX Runtime CPU** 对仓库内两张车牌图
   运行两份 V5 部署模型，并断言最终公开文字。这是可复现的识别效果回归，
   不是 TensorRT 性能测试。
3. `PlateRecognitionOCRV5_demo` 使用 C++/TensorRT 8.2.5.1。它必须在安装了
   对应 Linux TensorRT、CUDA 和 NVIDIA GPU 的目标机上运行，才能完成最终
   TensorRT 精度及耗时验收。

## ONNX Runtime 识别效果回归

使用项目指定的现有 Python 环境；脚本不会安装或修改任何包：

```powershell
& 'P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe' `
  .\PlateRecognitionOCRV5\test\recognition_effect_test.py
```

脚本依次测试项目内的两种实际部署规格：

- `models/mobile/inference.onnx`：mobile/light 规格；
- `models/server/inference.onnx`：server 规格，使用适配 TensorRT 8.2.5.1 的
  IR6/opset11 部署图。

当前固定回归结果：

| 模型 | 苏A0GY63 | 藏ABT766 |
| --- | --- | --- |
| mobile/light | `A0GY63` | `藏ABT766` |
| server | `苏A0GY63` | `藏ABT766` |

mobile/light 在第一张图上没有正确输出省份字符，这是模型效果差异，测试如实
固定当前结果，不把它伪装为通过准确率评测。脚本输出的 ORT CPU 单次耗时只用于
观察，不设置性能门限，也不能代表 TensorRT 性能。

脚本直接解析 `src/OCRV5Text.cpp` 中 mobile/server 两张独立数组，
用实际 C++ 固化映射解码 ONNX 输出。运行时不读取完整 PP-OCR
字典。若开发机仍保留原始字典，可以额外核对全部 sparse 类别索引：

```powershell
& 'P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe' `
  .\PlateRecognitionOCRV5\test\recognition_effect_test.py `
  --source-dictionary 'Q:\PPOCRV5\PPOCRV5\test\ppocrv5_key.json'
```

此前 V5 实际执行过 `tools/validate_onnx.py`，使用的是 ONNX Runtime 和
`resized.jpg`，得到 raw `X1S25109062-225054206`、过滤后
`X1S25109062225054206`。由于脚本位于 `tools/`、样例来自 Q 盘且内容并非车牌，
而 `test/` 下只有解码单元测试和 TensorRT demo，所以从测试目录看不出端到端
识别效果。本回归脚本补齐了这一缺口。
