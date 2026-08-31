# PlateRecognitionOCRV5

PP-OCRv5 文字识别模块，复用项目公共 TensorRT 8.2.5.1 推理封装。输入为
OpenCV BGR 图像，高度固定为 48，宽度动态并对齐到 20 的倍数，像素按 `/255`
归一化。CTC 解码后只返回 `PlateRecognition` 已有的 77 个可见字符。

模块包含两种真实部署规格：

- Mobile：`models/mobile/inference.onnx`；
- Server：`models/server/inference.onnx`。

两个模型都输出 18,385 类。代码在 `OCRV5Text.cpp` 中分别维护
`kMobileClassIndices` 和 `kServerClassIndices` 两份 77 项稀疏映射。两份上游模型
当前采用相同类别顺序，但映射没有共用同一个数组，后续任一规格升级时可以独立校验。
运行时不复制、读取或依赖完整 PP-OCR 字典。

调用方必须让模型路径和 `model_variant` 一致：

```cpp
PlateRecognitionOCRV5Config config;
config.model_variant = PlateRecognitionOCRV5ModelVariant::kMobile;
config.onnx_model_path =
    "PlateRecognitionOCRV5/models/mobile/inference.onnx";

PlateRecognitionOCRV5 recognizer;
recognizer.Init(&config);
recognizer.Run(image, plate_text);
recognizer.Release();
```

公共接口为 `Init`、`Run`、`Release`、`LastError`，并保留与基准模块同风格的
`PlateRecognitionOCRV5Init/Run/Release` 别名。库内部不向 stdout/stderr 输出信息；
demo 成功时也只输出最终车牌字符串。

- 模型来源、哈希和准备命令：[`models/README.md`](models/README.md)
- ONNX Runtime 单图校验：[`tools/validate_onnx.py`](tools/validate_onnx.py)
- 两规格真实车牌效果回归：[`test/README.md`](test/README.md)
- 三种识别后端的统一说明：[`../OCR_BACKENDS.md`](../OCR_BACKENDS.md)

当前 ONNX Runtime CPU 回归中，Server 在两张仓库车牌图上分别输出
`苏A0GY63`、`藏ABT766`；Mobile 分别输出 `A0GY63`、`藏ABT766`。这反映模型
实际效果差异，不会用后处理伪造缺失的省份字符。TensorRT 精度和性能仍需在安装
TensorRT 8.2.5.1、CUDA 和 NVIDIA GPU 的目标 Linux 机器上验收。
