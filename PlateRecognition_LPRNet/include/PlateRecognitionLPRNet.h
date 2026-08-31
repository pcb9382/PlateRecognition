#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "DataTypes_Plate.h"

#if defined(_WIN32) && defined(PLATE_LPRNET_BUILD_SHARED)
#define PLATE_LPRNET_API __declspec(dllexport)
#elif defined(_WIN32)
#define PLATE_LPRNET_API __declspec(dllimport)
#else
#define PLATE_LPRNET_API __attribute__((visibility("default")))
#endif

struct PlateRecognitionLPRNetConfig {
  // A static-batch-1 ONNX file exported by tools/export_lprnet_onnx.py,
  // or a TensorRT 8.2.5.1 engine built from that ONNX file.
  std::string model_path;

  // Empty means "<onnx stem>_<precision>_trt8.2.5.1.engine".
  // Ignored when model_path already points to an engine.
  std::string engine_cache_path;

  int gpu_id;
  bool enable_fp16;
  std::size_t workspace_size;

  PlateRecognitionLPRNetConfig()
      : gpu_id(0), enable_fp16(false), workspace_size(1U << 30) {}
};

class PLATE_LPRNET_API PlateRecognitionLPRNet {
public:
  static const int kInputChannels = 3;
  static const int kInputHeight = 24;
  static const int kInputWidth = 94;
  static const int kClassCount = 68;
  static const int kTimeSteps = 18;
  static const int kBlankIndex = 67;

  PlateRecognitionLPRNet();
  ~PlateRecognitionLPRNet();

  PlateRecognitionLPRNet(const PlateRecognitionLPRNet &) = delete;
  PlateRecognitionLPRNet &operator=(const PlateRecognitionLPRNet &) = delete;

  HZFLAG Init(const PlateRecognitionLPRNetConfig &config);
  HZFLAG Run(const cv::Mat &image, std::string &plate_text);
  HZFLAG Release();

  // Names matching the existing PlateRecognition class style.
  HZFLAG
  PlateRecognitionLPRNetInit(const PlateRecognitionLPRNetConfig &config) {
    return Init(config);
  }
  HZFLAG PlateRecognitionLPRNetRun(const cv::Mat &image,
                                   std::string &plate_text) {
    return Run(image, plate_text);
  }
  HZFLAG PlateRecognitionLPRNetRelease() { return Release(); }

  // Decodes logits laid out as [68, 18]. The official '-' class (67) is
  // the CTC blank and is never emitted. The other 67 official classes are
  // returned without applying PlateRecognition's character whitelist.
  static HZFLAG DecodeGreedy(const float *logits, std::size_t logits_count,
                             std::string &plate_text);

  const std::string &LastError() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
