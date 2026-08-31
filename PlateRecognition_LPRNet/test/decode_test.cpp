#include "PlateRecognitionLPRNet.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main() {
  const int classes = PlateRecognitionLPRNet::kClassCount;
  const int steps = PlateRecognitionLPRNet::kTimeSteps;
  const int blank = PlateRecognitionLPRNet::kBlankIndex;
  std::vector<float> logits(static_cast<std::size_t>(classes * steps), -100.0F);

  const int sequence[PlateRecognitionLPRNet::kTimeSteps] = {
      0,  0,     blank, 41, 41,    blank, 41,    blank, 65,
      65, blank, 66,    66, blank, blank, blank, blank, blank};
  for (int time = 0; time < steps; ++time) {
    logits[static_cast<std::size_t>(sequence[time] * steps + time)] = 100.0F;
  }

  std::string plate;
  if (PlateRecognitionLPRNet::DecodeGreedy(logits.data(), logits.size(),
                                           plate) != HZ_SUCCESS) {
    std::cerr << "DecodeGreedy returned an error\n";
    return 1;
  }
  if (plate != "京AAIO") {
    std::cerr << "Unexpected decode result: " << plate << '\n';
    return 2;
  }

  std::string invalid;
  if (PlateRecognitionLPRNet::DecodeGreedy(logits.data(), logits.size() - 1,
                                           invalid) != HZ_ERROR ||
      !invalid.empty()) {
    std::cerr << "Invalid tensor size was not rejected\n";
    return 3;
  }
  return 0;
}
