#include "PlateRecognitionOCRV6.h"

#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#ifndef PLATE_RECOGNITION_OCRV6_DEFAULT_MODEL_DIR
#define PLATE_RECOGNITION_OCRV6_DEFAULT_MODEL_DIR "./models/medium"
#endif

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3 && argc != 4) {
    return 2;
  }

  PlateRecognitionOCRV6Config config;
  std::string image_path;
  if (argc == 2) {
    const std::string model_directory =
        PLATE_RECOGNITION_OCRV6_DEFAULT_MODEL_DIR;
    config.onnx_model_path = model_directory + "/inference.onnx";
    image_path = argv[1];
  } else {
    config.onnx_model_path = argv[1];
    image_path = argv[2];
    if (argc == 4) {
      config.engine_cache_path = argv[3];
    }
  }

  PlateRecognitionOCRV6 recognizer;
  HZFLAG status = recognizer.Init(&config);
  if (status != HZ_SUCCESS) {
    return static_cast<int>(status) + 10;
  }

  const cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
  std::string plate_text;
  status = recognizer.Run(image, plate_text);
  recognizer.Release();
  if (status != HZ_SUCCESS) {
    return static_cast<int>(status) + 20;
  }

  std::cout << plate_text << '\n';
  return 0;
}
