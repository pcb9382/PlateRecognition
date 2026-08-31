# PlateRecognitionOCRV6

PP-OCRv6 文字识别模块，支持 tiny、small、medium 三档模型，并复用项目公共
TensorRT 8.2.5.1 推理封装。输入为 OpenCV BGR 图像，高度固定 48、宽度动态，
使用 `[-1,1]` 归一化及右侧补零；输出经过 CTC 解码后只保留
`PlateRecognition` 的 77 个可见字符。

- 公共接口：`Init`、`Run`、`Release`、`LastError`
- 同风格别名：`PlateRecognitionOCRV6Init/Run/Release`
- 三档模型、固化映射和导入命令：[`models/README.md`](models/README.md)
- ONNX Runtime 三档验证：`tools/validate_onnx.py`
- 两张车牌的识别效果回归：[`test/README.md`](test/README.md)
- 更完整的统一说明：[`../OCR_BACKENDS.md`](../OCR_BACKENDS.md)

模块不复制、加载或依赖外部 OCR 字典。tiny 的 6,906 类模型使用一套固化稀疏
映射，small/medium 的 18,710 类模型共用另一套；两套映射均只包含
`PlateRecognition` 的 77 个可见字符。库内部不向 stdout/stderr 打印信息；
demo 和 `tools/validate_onnx.py` 成功时仅输出最终识别字符串。

使用指定 Python 环境及 ONNX Runtime CPU 实测，tiny 输出 6,906 类，
small/medium 输出 18,710 类。medium 对仓库车牌图的公开结果为
`苏A0GY63`；三档、两张图片的固定回归结果见测试说明。
