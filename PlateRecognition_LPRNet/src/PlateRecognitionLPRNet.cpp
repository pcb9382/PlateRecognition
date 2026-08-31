#include "PlateRecognitionLPRNet.h"

#include "TensorRTRecognizer.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <sstream>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

const char *const kOfficialCharacters[PlateRecognitionLPRNet::kClassCount] = {
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
    "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新", "0",  "1",  "2",  "3",  "4",
    "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",  "E",  "F",  "G",
    "H",  "J",  "K",  "L",  "M",  "N",  "P",  "Q",  "R",  "S",  "T",  "U",
    "V",  "W",  "X",  "Y",  "Z",  "I",  "O",  "-"};

bool FileExists(const std::string &path) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  return stream.good();
}

std::string Lowercase(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return text;
}

bool EndsWith(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string DefaultEnginePath(const PlateRecognitionLPRNetConfig &config) {
  const std::string suffix = config.enable_fp16 ? "_fp16_trt8.2.5.1.engine"
                                                : "_fp32_trt8.2.5.1.engine";
  return config.model_path.substr(0, config.model_path.size() - 5) + suffix;
}

bool OutputShapeIsValid(const std::vector<int> &shape) {
  return shape.size() == 3U && shape[0] == 1 && shape[1] == 68 &&
         shape[2] == 18;
}

} // namespace

const int PlateRecognitionLPRNet::kInputChannels;
const int PlateRecognitionLPRNet::kInputHeight;
const int PlateRecognitionLPRNet::kInputWidth;
const int PlateRecognitionLPRNet::kClassCount;
const int PlateRecognitionLPRNet::kTimeSteps;
const int PlateRecognitionLPRNet::kBlankIndex;

class PlateRecognitionLPRNet::Impl {
public:
  HZFLAG Init(const PlateRecognitionLPRNetConfig &config) {
    Release();
    last_error_.clear();

    if (config.model_path.empty()) {
      return Fail(HZ_WITHOUTMODEL, "LPRNet model_path is empty");
    }
    if (!FileExists(config.model_path)) {
      return Fail(HZ_WITHOUTMODEL,
                  "LPRNet model does not exist: " + config.model_path);
    }
    if (config.gpu_id < 0) {
      return Fail(HZ_INITFAILED, "gpu_id must be non-negative");
    }
    if (config.workspace_size == 0) {
      return Fail(HZ_INITFAILED, "workspace_size must be greater than zero");
    }

    const std::string lowercase_path = Lowercase(config.model_path);
    plate_recognition::TensorRTRecognizerConfig trt_config;
    if (EndsWith(lowercase_path, ".onnx")) {
      trt_config.onnx_path = config.model_path;
      trt_config.engine_path = config.engine_cache_path.empty()
                                   ? DefaultEnginePath(config)
                                   : config.engine_cache_path;
    } else if (EndsWith(lowercase_path, ".engine") ||
               EndsWith(lowercase_path, ".plan")) {
      // The common runtime loads the existing engine first. This placeholder
      // ONNX path is consulted only if that engine is incompatible/corrupt.
      trt_config.onnx_path = config.model_path + ".onnx";
      trt_config.engine_path = config.model_path;
    } else {
      return Fail(HZ_INITFAILED,
                  "model_path must end in .onnx, .engine, or .plan");
    }

    trt_config.gpu_id = config.gpu_id;
    trt_config.enable_fp16 = config.enable_fp16;
    trt_config.workspace_size = config.workspace_size;
    trt_config.min_shape = InputShape();
    trt_config.opt_shape = InputShape();
    trt_config.max_shape = InputShape();

    if (!recognizer_.Init(trt_config)) {
      return Fail(HZ_INITFAILED, recognizer_.LastError());
    }
    if (recognizer_.InputName() != "input" ||
        recognizer_.OutputName() != "logits") {
      recognizer_.Release();
      return Fail(
          HZ_INITFAILED,
          "LPRNet engine must expose input 'input' and output 'logits'");
    }

    input_.assign(
        static_cast<std::size_t>(kInputChannels * kInputHeight * kInputWidth),
        0.0F);
    initialized_ = true;
    return HZ_SUCCESS;
  }

  HZFLAG Run(const cv::Mat &image, std::string &plate_text) {
    plate_text.clear();
    last_error_.clear();
    try {
      if (!initialized_ || !recognizer_.IsInitialized()) {
        return Fail(HZ_INITFAILED, "LPRNet is not initialized");
      }
      if (image.empty()) {
        return Fail(HZ_IMGEMPTY, "Input image is empty");
      }
      if (image.depth() != CV_8U ||
          (image.channels() != 1 && image.channels() != 3 &&
           image.channels() != 4)) {
        return Fail(HZ_IMGFORMATERROR,
                    "Input image must be uint8 with 1, 3, or 4 channels");
      }

      cv::Mat bgr;
      if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
      } else if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
      } else {
        bgr = image;
      }

      cv::Mat resized;
      cv::resize(bgr, resized, cv::Size(kInputWidth, kInputHeight), 0.0, 0.0,
                 cv::INTER_LINEAR);
      const int plane_size = kInputHeight * kInputWidth;
      for (int row = 0; row < kInputHeight; ++row) {
        const cv::Vec3b *pixels = resized.ptr<cv::Vec3b>(row);
        for (int column = 0; column < kInputWidth; ++column) {
          const int offset = row * kInputWidth + column;
          for (int channel = 0; channel < kInputChannels; ++channel) {
            input_[static_cast<std::size_t>(channel * plane_size + offset)] =
                (static_cast<float>(pixels[column][channel]) - 127.5F) *
                0.0078125F;
          }
        }
      }

      std::vector<float> logits;
      std::vector<int> output_shape;
      if (!recognizer_.Infer(input_, InputShape(), logits, output_shape)) {
        return Fail(HZ_ERROR, recognizer_.LastError());
      }
      if (!OutputShapeIsValid(output_shape) ||
          logits.size() != static_cast<std::size_t>(kClassCount * kTimeSteps)) {
        std::ostringstream message;
        message << "Expected LPRNet output [1,68,18], got [";
        for (std::size_t index = 0; index < output_shape.size(); ++index) {
          if (index != 0U) {
            message << ',';
          }
          message << output_shape[index];
        }
        message << ']';
        return Fail(HZ_ERROR, message.str());
      }
      return PlateRecognitionLPRNet::DecodeGreedy(logits.data(), logits.size(),
                                                  plate_text);
    } catch (const cv::Exception &exception) {
      plate_text.clear();
      return Fail(HZ_ERROR,
                  std::string("OpenCV inference error: ") + exception.what());
    } catch (const std::exception &exception) {
      plate_text.clear();
      return Fail(HZ_ERROR,
                  std::string("LPRNet inference error: ") + exception.what());
    } catch (...) {
      plate_text.clear();
      return Fail(HZ_ERROR, "Unknown LPRNet inference error");
    }
  }

  HZFLAG Release() {
    initialized_ = false;
    recognizer_.Release();
    input_.clear();
    last_error_.clear();
    return HZ_SUCCESS;
  }

  const std::string &LastError() const { return last_error_; }

private:
  static plate_recognition::TensorRTShape InputShape() {
    plate_recognition::TensorRTShape shape;
    shape.n = 1;
    shape.c = kInputChannels;
    shape.h = kInputHeight;
    shape.w = kInputWidth;
    return shape;
  }

  HZFLAG Fail(HZFLAG status, const std::string &message) {
    last_error_ = message;
    return status;
  }

  plate_recognition::TensorRTRecognizer recognizer_;
  std::vector<float> input_;
  std::string last_error_;
  bool initialized_ = false;
};

PlateRecognitionLPRNet::PlateRecognitionLPRNet() : impl_(new Impl()) {}

PlateRecognitionLPRNet::~PlateRecognitionLPRNet() = default;

HZFLAG
PlateRecognitionLPRNet::Init(const PlateRecognitionLPRNetConfig &config) {
  return impl_->Init(config);
}

HZFLAG PlateRecognitionLPRNet::Run(const cv::Mat &image,
                                   std::string &plate_text) {
  return impl_->Run(image, plate_text);
}

HZFLAG PlateRecognitionLPRNet::Release() { return impl_->Release(); }

HZFLAG PlateRecognitionLPRNet::DecodeGreedy(const float *logits,
                                            std::size_t logits_count,
                                            std::string &plate_text) {
  plate_text.clear();
  const std::size_t expected =
      static_cast<std::size_t>(kClassCount * kTimeSteps);
  if (logits == nullptr || logits_count != expected) {
    return HZ_ERROR;
  }

  int previous = kBlankIndex;
  for (int time_step = 0; time_step < kTimeSteps; ++time_step) {
    int best_class = 0;
    float best_score = logits[static_cast<std::size_t>(time_step)];
    for (int class_index = 1; class_index < kClassCount; ++class_index) {
      const float score = logits[static_cast<std::size_t>(
          class_index * kTimeSteps + time_step)];
      if (score > best_score) {
        best_score = score;
        best_class = class_index;
      }
    }
    if (best_class != kBlankIndex && best_class != previous) {
      plate_text += kOfficialCharacters[best_class];
    }
    previous = best_class;
  }
  return HZ_SUCCESS;
}

const std::string &PlateRecognitionLPRNet::LastError() const {
  return impl_->LastError();
}
