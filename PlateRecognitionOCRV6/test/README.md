# PP-OCRv6 测试说明

本目录把“解码正确”“ONNX 识别效果”和“TensorRT 部署验收”分开测试：

1. `PlateRecognitionOCRV6_unit` 不加载模型，验证 tiny/full 固化 sparse 映射、
   CTC 去重、blank 和 77 字符输出规则。
2. `recognition_effect_test.py` 使用 **ONNX Runtime CPU**，依次运行本地
   tiny、small、medium 三档 ONNX，并对仓库内两张车牌图断言最终公开文字。
3. `PlateRecognitionOCRV6_demo` 使用 C++/TensorRT 8.2.5.1，必须在目标
   Linux/CUDA/GPU 环境做最终精度和性能验收。

## ONNX Runtime 三档识别效果回归

```powershell
& 'P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe' `
  .\PlateRecognitionOCRV6\test\recognition_effect_test.py
```

脚本直接使用 `models/{tiny,small,medium}/inference.onnx`，并解析
`src/OCRV6Text.cpp` 中的 tiny/full C++ 固化数组进行解码，不需要运行时字典。
当前固定结果如下：

| 规格 | 苏A0GY63 | 藏ABT766 |
| --- | --- | --- |
| tiny | `5AGY63` | `ABT766` |
| small | `A0GY63` | `藏ABT766` |
| medium | `苏A0GY63` | `藏ABT766` |

表格反映模型的真实当前效果：tiny/small 并非两张图都全对。测试目的是发现模型、
预处理或类别映射发生非预期变化，不会把错误字符改成标签。脚本报告的 ORT CPU
单次耗时仅用于观察，不设置性能门限，也不能当作 TensorRT 性能数据。

部署代码只固化模型输出中会映射到 PlateRecognition 的 77 个类别，不读取完整
OCR 字典。原始 Q 盘字典仍存在时，可执行一次开发期来源审计：

```powershell
& 'P:\Algorithm\Plate\PlateRecognition_Android\.tools\python311\python.exe' `
  .\PlateRecognitionOCRV6\test\recognition_effect_test.py `
  --source-model-root 'Q:\PPOCRV5\PPOCRV6\models'
```

该选项逐一检查三档固化的全部 sparse class index 与各自源 `dict.txt` 是否一致；
它不是部署时的依赖。
