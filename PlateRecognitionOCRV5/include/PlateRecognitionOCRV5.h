#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "DataTypes_Plate.h"

enum class PlateRecognitionOCRV5ModelVariant {
  kMobile = 0,
  kServer = 1,
};

struct PlateRecognitionOCRV5Config {
  // Both bundled models use dynamic N/W and expose 18,385 output classes.
  std::string onnx_model_path;
  PlateRecognitionOCRV5ModelVariant model_variant =
      PlateRecognitionOCRV5ModelVariant::kServer;

  // Optional. When empty, a profile-specific path is derived beside the ONNX
  // file. TensorRT engines are not portable across TensorRT versions or GPU
  // platforms.
  std::string engine_cache_path;

  int gpu_id = 0;
  int min_input_width = 20;
  int opt_input_width = 760;
  int max_input_width = 1280;
  std::size_t workspace_size = static_cast<std::size_t>(1) << 30;
  bool enable_fp16 = false;
};

class PlateRecognitionOCRV5 {
public:
  PlateRecognitionOCRV5();
  ~PlateRecognitionOCRV5();

  PlateRecognitionOCRV5(const PlateRecognitionOCRV5 &) = delete;
  PlateRecognitionOCRV5 &operator=(const PlateRecognitionOCRV5 &) = delete;

  HZFLAG Init(const PlateRecognitionOCRV5Config *config);
  HZFLAG Run(const cv::Mat &image, std::string &plate_text);
  HZFLAG Release();

  // Baseline-compatible names retained for callers that use the existing
  // PlateRecognition naming convention.
  HZFLAG PlateRecognitionOCRV5Init(const PlateRecognitionOCRV5Config *config) {
    return Init(config);
  }
  HZFLAG PlateRecognitionOCRV5Run(const cv::Mat &image,
                                  std::string &plate_text) {
    return Run(image, plate_text);
  }
  HZFLAG PlateRecognitionOCRV5Release() { return Release(); }

  // Detailed diagnostics are returned to the caller and are never printed by
  // the library.
  std::string LastError() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
