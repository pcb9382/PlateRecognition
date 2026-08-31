#include "PlateRecognitionOCRV5.h"

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

#ifndef PLATE_RECOGNITION_OCRV5_DEFAULT_MODEL_ROOT
#define PLATE_RECOGNITION_OCRV5_DEFAULT_MODEL_ROOT "./models"
#endif

namespace {

bool ParseVariant(const std::string &name,
                  PlateRecognitionOCRV5ModelVariant &variant) {
  if (name == "mobile") {
    variant = PlateRecognitionOCRV5ModelVariant::kMobile;
    return true;
  }
  if (name == "server") {
    variant = PlateRecognitionOCRV5ModelVariant::kServer;
    return true;
  }
  return false;
}

const char *VariantDirectory(PlateRecognitionOCRV5ModelVariant variant) {
  return variant == PlateRecognitionOCRV5ModelVariant::kMobile ? "mobile"
                                                               : "server";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 5) {
    return 2;
  }

  PlateRecognitionOCRV5Config config;
  std::string image_path;
  if (argc == 2) {
    config.model_variant = PlateRecognitionOCRV5ModelVariant::kServer;
    image_path = argv[1];
    config.onnx_model_path =
        std::string(PLATE_RECOGNITION_OCRV5_DEFAULT_MODEL_ROOT) + "/server/" +
        "inference.onnx";
  } else {
    if (!ParseVariant(argv[1], config.model_variant)) {
      return 2;
    }
    if (argc == 3) {
      image_path = argv[2];
      config.onnx_model_path =
          std::string(PLATE_RECOGNITION_OCRV5_DEFAULT_MODEL_ROOT) + "/" +
          VariantDirectory(config.model_variant) + "/inference.onnx";
    } else {
      config.onnx_model_path = argv[2];
      image_path = argv[3];
      if (argc == 5) {
        config.engine_cache_path = argv[4];
      }
    }
  }

  PlateRecognitionOCRV5 recognizer;
  HZFLAG status = recognizer.Init(&config);
  if (status != HZ_SUCCESS) {
    return static_cast<int>(status) + 10;
  }

  const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  std::string plate_text;
  status = recognizer.Run(image, plate_text);
  recognizer.Release();
  if (status != HZ_SUCCESS) {
    return static_cast<int>(status) + 20;
  }

  std::cout << plate_text << '\n';
  return 0;
}
