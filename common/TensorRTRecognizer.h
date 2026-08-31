#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace plate_recognition {

struct TensorRTShape {
  int n = 1;
  int c = 3;
  int h = 0;
  int w = 0;
};

struct TensorRTRecognizerConfig {
  std::string onnx_path;
  std::string engine_path;
  int gpu_id = 0;
  bool enable_fp16 = false;
  std::size_t workspace_size = static_cast<std::size_t>(1U) << 30U;
  TensorRTShape min_shape;
  TensorRTShape opt_shape;
  TensorRTShape max_shape;
};

class TensorRTRecognizer {
public:
  TensorRTRecognizer();
  ~TensorRTRecognizer();

  TensorRTRecognizer(const TensorRTRecognizer &) = delete;
  TensorRTRecognizer &operator=(const TensorRTRecognizer &) = delete;

  bool Init(const TensorRTRecognizerConfig &config);
  bool Infer(const std::vector<float> &input, const TensorRTShape &input_shape,
             std::vector<float> &output, std::vector<int> &output_shape);
  void Release();

  bool IsInitialized() const;
  const std::string &LastError() const;
  const std::string &InputName() const;
  const std::string &OutputName() const;
  bool GetInputShape(TensorRTShape &shape) const;

private:
  class Impl;
  Impl *impl_;
};

} // namespace plate_recognition
