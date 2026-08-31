# PP-OCRv6 recognition models

本目录已经随模块放入全部三档文字识别（recognition）资产，不包含检测模型、TFLite 文件或构建产物。

默认使用 `medium`；`tiny` 适合更关注体积和速度的场景。运行时字符映射固化在模块代码中，不读取外部字典：

```text
models/
  tiny/
    inference.onnx
  small/
    inference.onnx
  medium/
    inference.onnx
```

原始资产位置：

```text
Q:\PPOCRV5\PPOCRV6\models\<tier>\rec\inference.onnx
```

资产清单（SHA-256）：

| 档位 | ONNX 大小 | ONNX SHA-256 | ONNX 契约 |
| --- | ---: | --- | --- |
| tiny | 4,462,639 B | `9EF676D6ED3C88256A2D92C640C44F25B0C40947E111B14B8BE8F594091563E6` | IR 6，opset 11，`[N,3,48,W] -> [N,T,6906]` |
| small | 21,159,378 B | `5435FD747C9E0EFE15A96D0B378D5BD157E9492ED8FD80EDF08F30D02FA24634` | IR 6，opset 11，`[N,3,48,W] -> [N,T,18710]` |
| medium | 76,554,979 B | `9C09ABF0957F7968C7586464B7397B84AD2387A0497A351AF40E9ACC71B673BA` | IR 6，opset 11，`[N,3,48,W] -> [N,T,18710]` |

可用导入脚本从上述原始位置重新复制默认的 medium：

```powershell
powershell -ExecutionPolicy Bypass -File .\PlateRecognitionOCRV6\tools\import_models.ps1
```

重新导入轻量 tiny 或全部三档：

```powershell
powershell -ExecutionPolicy Bypass -File .\PlateRecognitionOCRV6\tools\import_models.ps1 -Tier tiny
powershell -ExecutionPolicy Bypass -File .\PlateRecognitionOCRV6\tools\import_models.ps1 -All
```

如目标文件已存在，脚本默认拒绝覆盖；确认替换时显式传入 `-Force`。

测试程序只传图片路径时默认使用本目录的 medium 资产；也可显式传入 ONNX 和可选 engine 缓存路径：

```text
PlateRecognitionOCRV6_demo <image>
PlateRecognitionOCRV6_demo <inference.onnx> <image> [cache.engine]
```

可使用现有 Python 环境先核对 ONNX 预处理、输出类别及识别结果；脚本不会安装或修改包：

```powershell
P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe .\PlateRecognitionOCRV6\tools\validate_onnx.py
P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe .\PlateRecognitionOCRV6\tools\validate_onnx.py --all-tiers
```

默认车牌图按 tiny、small、medium 的顺序仅输出以下三行公开识别文字：

```text
5AGY63
A0GY63
苏A0GY63
```

模型原始输出类别数为：tiny 6906 类，small/medium 18710 类。模块只固化这些输出张量中对应 PlateRecognition 77 个可见字符的稀疏索引，CTC 解码及识别运行均不依赖外部字典。TensorRT engine 由 ONNX 首次初始化时生成，缓存名包含 TensorRT 8.2.5.1、精度和动态宽度 profile，不应作为通用模型资产提交。
