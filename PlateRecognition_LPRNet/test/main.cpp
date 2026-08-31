#include "PlateRecognitionLPRNet.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

int main(int argc, char **argv) {
  if (argc < 3 || argc > 5) {
    std::cerr
        << "usage: PlateRecognition_LPRNet_demo <model.onnx|model.engine> "
           "<plate-image> [gpu-id] [fp16:0|1]\n";
    return 2;
  }

  PlateRecognitionLPRNetConfig config;
  config.model_path = argv[1];
  config.gpu_id = argc >= 4 ? std::atoi(argv[3]) : 0;
  config.enable_fp16 = argc >= 5 && std::atoi(argv[4]) != 0;

  PlateRecognitionLPRNet recognizer;
  if (recognizer.Init(config) != HZ_SUCCESS) {
    std::cerr << recognizer.LastError() << '\n';
    return 3;
  }

  const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
  std::string plate;
  if (recognizer.Run(image, plate) != HZ_SUCCESS) {
    std::cerr << recognizer.LastError() << '\n';
    return 4;
  }

  std::cout << plate << '\n';
  return 0;
}
