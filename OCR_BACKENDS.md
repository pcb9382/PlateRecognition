# 文字识别后端说明

项目包含四个只做文字识别的独立模块，代码布局均为 `include/`、`src/`、
`test/`、`models/`，并统一复用 `common/TensorRTRecognizer.cpp`。四个模块的
C++ 推理统一使用项目指定的 **TensorRT 8.2.5.1**；编译期会检查版本，engine
仍需在实际部署 GPU 上首次生成。

| 模块 | 本地模型 | 模型输出 | 对外字符规则 |
| --- | --- | --- | --- |
| `PlateRecognitionOCRV5` | `models/mobile/inference.onnx`、`models/server/inference.onnx` | 两个规格均为 `[1,T,18385]` | Mobile、Server 各自使用一份固化的 77 字符稀疏映射 |
| `PlateRecognitionOCRV6` | `models/tiny/inference.onnx`、`models/small/inference.onnx`、`models/medium/inference.onnx` | tiny 为 6906 类，small/medium 为 18710 类 | tiny 使用一套固化映射，small/medium 共用 full 固化映射；每套均为 77 字符 |
| `PlateRecognition_LPRNet` | `models/lprnet_official_68.onnx` | 静态 `[1,68,18]` | 使用官网 68 类表，`-` 为 blank；保留官网 `I`、`O` |
| `PlateRecognitionFastOCR` | `models/official` 全部 10 个官网 ONNX；`models/trt8` 全部 10 个部署图 | 7、9 或 10 个独立字符槽，每槽 37 类 | 官网 `0-9A-Z_`；逐槽 argmax，只删除末尾 `_`，不是 CTC |

V5/V6 保持原始模型的分类层不变，在 CTC 解码时按原生类别索引折叠，再通过
代码内的稀疏映射只发布 `PlateRecognition` 的 77 个可见字符。部署不读取、
不复制外部字符字典。V5 的两份映射分别绑定 Mobile 和 Server，V6 则根据输出
类别数自动选择 tiny 或 full 映射。

V5/V6 库及 C++ 演示程序不会向 stdout/stderr 输出构建、预热、耗时或调试信息；
成功时只输出最终车牌字符串，失败详情由 `LastError()` 返回。LPRNet 与 FastOCR
保持各自官网字符定义，不套用 V5/V6 的 77 字符映射。

## 模型资产

- V5 的两种部署规格已经整理到
  `PlateRecognitionOCRV5/models/{mobile,server}/inference.onnx`。Mobile 模型已转换
  为 TensorRT 8.2.5.1 可接受的 IR 6 / opset 11，Server 模型同为 IR 6 /
  opset 11；来源、SHA-256 和可复现整理脚本见
  `PlateRecognitionOCRV5/models/README.md`。
- V6 的 tiny、small、medium 三种规格已经整理到
  `PlateRecognitionOCRV6/models/{tiny,small,medium}/inference.onnx`，均为 IR 6 /
  opset 11；来源与 SHA-256 见 `PlateRecognitionOCRV6/models/README.md`。
- LPRNet 目录包含官网 `Final_LPRNet_model.pth`、导出的
  `lprnet_official_68.onnx`、可复现导出/验证脚本以及 Apache-2.0 来源记录，详情见
  `PlateRecognition_LPRNet/README.md`。
- FastOCR 的 `models/official` 包含官网 Hub 当前 10 个 ONNX（总计 53,909,920
  字节）和 9 份唯一配置；`models/trt8` 包含脚本派生的 FP32 NCHW、静态 batch 1、
  plate-only、IR 6 / opset 11 部署图。完整哈希和转换说明见
  `PlateRecognitionFastOCR/models/README.md`。

V5/V6 不需要配套部署字典，模型与代码中对应的固化稀疏映射必须配套使用。
FastOCR 的官网 YAML 只用于溯源/验证，运行时 37 类字符表同样固化在 C++ 中。

## CMake

根工程默认构建四个模块，可独立关闭：

```bash
cmake -S . -B build \
  -DPLATE_BUILD_OCRV5=ON \
  -DPLATE_BUILD_OCRV6=ON \
  -DPLATE_BUILD_LPRNET=ON \
  -DPLATE_BUILD_FASTOCR=ON
cmake --build build -j
```

根 `CMakeLists.txt` 中的 `TensorRT_INCLUDE` 与 `TensorRT_LIB` 必须指向项目统一
使用的 TensorRT 8.2.5.1。V5/V6 默认使用动态宽度 profile，FastOCR 使用各部署
图的静态输入；缓存文件名包含 TRT 版本与精度，不能复用其他 TensorRT 版本或
其他 GPU 生成的 engine。通用运行时还会校验相邻 `.meta` 中的 ONNX 指纹、
CUDA/TRT、GPU 型号/规格、精度与 profile，并以临时文件原子发布缓存。
相邻 `.lock` 文件会串行化多个进程对同一 engine 的首次构建。
从 ONNX 自动构建时缓存目录必须可写；只读部署应直接加载已生成的 `.engine`。

## C++ 调用

四个 C++ 类均提供统一的 `Init`、`Run`、`Release`、`LastError`，同时保留
`PlateRecognition<模块名>Init/Run/Release` 风格别名。V5 必须让
`model_variant` 与所选模型规格一致，例如：

```cpp
PlateRecognitionOCRV5Config config;
config.onnx_model_path =
    "PlateRecognitionOCRV5/models/mobile/inference.onnx";
config.model_variant = PlateRecognitionOCRV5ModelVariant::kMobile;

PlateRecognitionOCRV5 recognizer;
if (recognizer.Init(&config) != HZ_SUCCESS) {
    // 读取 recognizer.LastError()
}
std::string plate;
recognizer.Run(bgr_image, plate);
recognizer.Release();
```

使用 Server 时将路径改为 `models/server/inference.onnx`，并将枚举改为
`PlateRecognitionOCRV5ModelVariant::kServer`。V6 无需规格枚举，它会从输出类别数
选择 tiny/full 固化映射。LPRNet 和 FastOCR 另外提供纯 C 头文件
`PlateRecognitionLPRNet_c.h`、`PlateRecognitionFastOCR_c.h`，采用 opaque
handle，可直接从 C 或其他支持 C ABI 的语言调用。

## 模型与识别效果验证

仅使用用户指定的现有 Python 环境，不升级或降级其中已安装的包：

```powershell
$python = 'P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe'

# V5：用 ONNX Runtime CPU 跑 Mobile、Server 两个本地 ONNX
& $python .\PlateRecognitionOCRV5\test\recognition_effect_test.py

# V6：用 ONNX Runtime CPU 跑 tiny、small、medium 三个本地 ONNX
& $python .\PlateRecognitionOCRV6\test\recognition_effect_test.py

# LPRNet：先执行官网 PT 权重，再用 ONNX Runtime 执行 ONNX，比较数值与 argmax
& $python .\PlateRecognition_LPRNet\tools\export_lprnet_onnx.py
& $python .\PlateRecognition_LPRNet\tools\validate_lprnet.py --images <plate.jpg>

# FastOCR：验证全部 10 组官网/部署 ONNX，再跑两张官网测试图
& $python .\PlateRecognitionFastOCR\tools\validate_onnx.py
& $python .\PlateRecognitionFastOCR\test\recognition_effect_test.py
```

因此，V5/V6 当前的识别效果回归结果来自 ONNX Runtime CPU 对 ONNX 的实际推理；
LPRNet 的 PT→ONNX 一致性由 PyTorch 与 ONNX Runtime 共同验证；FastOCR 的 10
组模型均已完成 full checker、官方/部署数值误差及逐槽 argmax 对比。当前 Windows
主机没有可运行的 Linux TensorRT 8.2.5.1 环境，所以这里只能完成模型、解码和
编译侧检查，TensorRT engine 构建、加载及 GPU 识别效果仍须在目标 Linux +
NVIDIA GPU 环境完成最终验收。
