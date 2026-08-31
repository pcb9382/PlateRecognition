#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "DataTypes_Plate.h"

// PP-OCRv6 recognition-only configuration. The ONNX models from the tiny,
// small and medium tiers are accepted. Their two sparse class mappings are
// embedded in the module, so no external character dictionary is required.
struct PlateRecognitionOCRV6Config {
  std::string onnx_model_path;
  // Optional TensorRT engine cache.  When empty, a version/profile-specific
  // name is derived next to onnx_model_path.
  std::string engine_cache_path;
  int gpu_id = 0;
  int min_input_width = 160;
  int opt_input_width = 320;
  int max_input_width = 3200;
  std::size_t workspace_size = static_cast<std::size_t>(1) << 30;
  bool enable_fp16 = false;
};

class PlateRecognitionOCRV6 {
public:
  PlateRecognitionOCRV6();
  ~PlateRecognitionOCRV6();

  PlateRecognitionOCRV6(const PlateRecognitionOCRV6 &) = delete;
  PlateRecognitionOCRV6 &operator=(const PlateRecognitionOCRV6 &) = delete;

  HZFLAG Init(const PlateRecognitionOCRV6Config *config);
  HZFLAG Run(const cv::Mat &image, std::string &plate_text);
  HZFLAG Release();

  HZFLAG PlateRecognitionOCRV6Init(const PlateRecognitionOCRV6Config *config) {
    return Init(config);
  }

  HZFLAG PlateRecognitionOCRV6Run(const cv::Mat &image,
                                  std::string &plate_text) {
    return Run(image, plate_text);
  }

  HZFLAG PlateRecognitionOCRV6Release() { return Release(); }

  // Detailed diagnostics are returned to the caller and never printed.
  std::string LastError() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
