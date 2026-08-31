#include "PlateRecognitionOCRV6.h"

#include "OCRV6Text.h"
#include "TensorRTRecognizer.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace {

const int kInputHeight = 48;
const int kMinSupportedWidth = 160;
const int kMaxSupportedWidth = 3200;

bool FileExists(const std::string &path) {
  if (path.empty()) {
    return false;
  }
  std::ifstream stream(path.c_str(), std::ios::binary);
  return stream.good();
}

std::string LowercaseExtension(const std::string &path) {
  const std::string::size_type separator = path.find_last_of("/\\");
  const std::string::size_type dot = path.find_last_of('.');
  if (dot == std::string::npos ||
      (separator != std::string::npos && dot < separator)) {
    return std::string();
  }
  std::string extension = path.substr(dot);
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  return extension;
}

std::string StripExtension(const std::string &path) {
  const std::string::size_type separator = path.find_last_of("/\\");
  const std::string::size_type dot = path.find_last_of('.');
  if (dot == std::string::npos ||
      (separator != std::string::npos && dot < separator)) {
    return path;
  }
  return path.substr(0, dot);
}

bool IsEnginePath(const std::string &path) {
  const std::string extension = LowercaseExtension(path);
  return extension == ".engine" || extension == ".plan" || extension == ".trt";
}

std::string DeriveEnginePath(const PlateRecognitionOCRV6Config &config) {
  return StripExtension(config.onnx_model_path) + ".trt8.2.5.1." +
         (config.enable_fp16 ? "fp16" : "fp32") + ".w" +
         std::to_string(config.min_input_width) + "-" +
         std::to_string(config.opt_input_width) + "-" +
         std::to_string(config.max_input_width) + ".engine";
}

bool ReadOutputShape(const std::vector<float> &output,
                     const std::vector<int> &output_shape,
                     std::size_t &time_steps, std::size_t &class_count) {
  time_steps = 0;
  class_count = 0;
  if (output.empty() || output_shape.size() != 3 || output_shape[0] != 1 ||
      output_shape[1] <= 0 || output_shape[2] <= 0) {
    return false;
  }
  time_steps = static_cast<std::size_t>(output_shape[1]);
  class_count = static_cast<std::size_t>(output_shape[2]);
  if (time_steps >
          std::numeric_limits<std::size_t>::max() / class_count ||
      output.size() != time_steps * class_count) {
    time_steps = 0;
    class_count = 0;
    return false;
  }
  return true;
}

plate_recognition::TensorRTShape MakeShape(int height, int width) {
  plate_recognition::TensorRTShape shape;
  shape.n = 1;
  shape.c = 3;
  shape.h = height;
  shape.w = width;
  return shape;
}

} // namespace

class PlateRecognitionOCRV6::Impl {
public:
  Impl() : class_count_(0), initialized_(false) {}

  ~Impl() { ResetUnlocked(); }

  void ResetUnlocked() {
    initialized_ = false;
    recognizer_.Release();
    class_count_ = 0;
    input_.clear();
    output_.clear();
    output_shape_.clear();
  }

  HZFLAG Fail(HZFLAG status, const std::string &message) {
    last_error_ = message;
    return status;
  }

  bool PrepareInput(const cv::Mat &image, int &canvas_width) {
    if (image.empty() || image.dims != 2 || image.depth() != CV_8U) {
      return false;
    }

    cv::Mat bgr;
    try {
      if (image.channels() == 3) {
        bgr = image;
      } else if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
      } else if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
      } else {
        return false;
      }

      const double scaled_width =
          static_cast<double>(kInputHeight) * bgr.cols / bgr.rows;
      const double bounded_width =
          std::min(static_cast<double>(config_.max_input_width),
                   std::ceil(scaled_width));
      int content_width = static_cast<int>(bounded_width);
      content_width = std::max(1, content_width);
      canvas_width = std::max(config_.opt_input_width, content_width);
      canvas_width = std::max(config_.min_input_width,
                              std::min(config_.max_input_width, canvas_width));

      cv::Mat resized;
      cv::resize(bgr, resized, cv::Size(content_width, kInputHeight), 0.0, 0.0,
                 cv::INTER_LINEAR);

      const std::size_t plane =
          static_cast<std::size_t>(kInputHeight) * canvas_width;
      input_.assign(plane * 3, 0.0f);
      for (int row_index = 0; row_index < kInputHeight; ++row_index) {
        const cv::Vec3b *row = resized.ptr<cv::Vec3b>(row_index);
        for (int column = 0; column < content_width; ++column) {
          const std::size_t offset =
              static_cast<std::size_t>(row_index) * canvas_width + column;
          input_[offset] = static_cast<float>(row[column][0]) / 127.5f - 1.0f;
          input_[plane + offset] =
              static_cast<float>(row[column][1]) / 127.5f - 1.0f;
          input_[2 * plane + offset] =
              static_cast<float>(row[column][2]) / 127.5f - 1.0f;
        }
      }
    } catch (const cv::Exception &) {
      return false;
    }
    return true;
  }

  mutable std::mutex mutex_;
  PlateRecognitionOCRV6Config config_;
  plate_recognition::TensorRTRecognizer recognizer_;
  std::vector<float> input_;
  std::vector<float> output_;
  std::vector<int> output_shape_;
  std::string last_error_;
  std::size_t class_count_;
  bool initialized_;
};

PlateRecognitionOCRV6::PlateRecognitionOCRV6() : impl_(new Impl()) {}

PlateRecognitionOCRV6::~PlateRecognitionOCRV6() {}

HZFLAG PlateRecognitionOCRV6::Init(const PlateRecognitionOCRV6Config *config) {
  std::lock_guard<std::mutex> lock(impl_->mutex_);
  impl_->ResetUnlocked();
  impl_->last_error_.clear();
  if (config == NULL) {
    return impl_->Fail(HZ_CONFIGLOADFAILED, "Configuration pointer is null");
  }
  if (config->gpu_id < 0 || config->min_input_width < kMinSupportedWidth ||
      config->min_input_width > config->opt_input_width ||
      config->opt_input_width > config->max_input_width ||
      config->max_input_width > kMaxSupportedWidth ||
      config->workspace_size == 0 || config->onnx_model_path.empty()) {
    return impl_->Fail(HZ_CONFIGLOADFAILED, "Invalid PP-OCRv6 configuration");
  }
  impl_->config_ = *config;

  std::string engine_path = config->engine_cache_path;
  if (engine_path.empty()) {
    engine_path = IsEnginePath(config->onnx_model_path)
                      ? config->onnx_model_path
                      : DeriveEnginePath(*config);
  }
  if (!FileExists(config->onnx_model_path) && !FileExists(engine_path)) {
    impl_->ResetUnlocked();
    return impl_->Fail(HZ_WITHOUTMODEL,
                       "Neither the ONNX model nor TensorRT engine exists");
  }

  plate_recognition::TensorRTRecognizerConfig trt_config;
  trt_config.onnx_path = config->onnx_model_path;
  trt_config.engine_path = engine_path;
  trt_config.gpu_id = config->gpu_id;
  trt_config.enable_fp16 = config->enable_fp16;
  trt_config.workspace_size = config->workspace_size;
  trt_config.min_shape = MakeShape(kInputHeight, config->min_input_width);
  trt_config.opt_shape = MakeShape(kInputHeight, config->opt_input_width);
  trt_config.max_shape = MakeShape(kInputHeight, config->max_input_width);
  if (!impl_->recognizer_.Init(trt_config)) {
    const std::string error = impl_->recognizer_.LastError();
    impl_->ResetUnlocked();
    return impl_->Fail(HZ_INITFAILED, error);
  }

  // Resolve one dynamic output shape at initialization and select the embedded
  // tiny or full sparse mapping before the first user image is processed.
  const plate_recognition::TensorRTShape validation_shape =
      trt_config.opt_shape;
  const std::size_t validation_count =
      static_cast<std::size_t>(validation_shape.n) * validation_shape.c *
      validation_shape.h * validation_shape.w;
  impl_->input_.assign(validation_count, 0.0f);
  if (!impl_->recognizer_.Infer(impl_->input_, validation_shape, impl_->output_,
                                impl_->output_shape_)) {
    const std::string error = impl_->recognizer_.LastError();
    impl_->ResetUnlocked();
    return impl_->Fail(HZ_INITFAILED, error);
  }
  std::size_t validation_steps = 0;
  std::size_t validation_classes = 0;
  if (!ReadOutputShape(impl_->output_, impl_->output_shape_, validation_steps,
                       validation_classes) ||
      !plate_recognition_ocr_v6::detail::IsSupportedClassCount(
          validation_classes)) {
    impl_->ResetUnlocked();
    return impl_->Fail(
        HZ_INITFAILED,
        "Unsupported PP-OCRv6 output classes; expected 6906 or 18710");
  }
  impl_->class_count_ = validation_classes;
  impl_->input_.clear();
  impl_->output_.clear();
  impl_->output_shape_.clear();
  impl_->initialized_ = true;
  return HZ_SUCCESS;
}

HZFLAG PlateRecognitionOCRV6::Run(const cv::Mat &image,
                                  std::string &plate_text) {
  plate_text.clear();
  std::lock_guard<std::mutex> lock(impl_->mutex_);
  impl_->last_error_.clear();
  if (!impl_->initialized_) {
    return impl_->Fail(HZ_INITFAILED, "Recognizer is not initialized");
  }
  if (image.empty()) {
    return impl_->Fail(HZ_IMGEMPTY, "Input image is empty");
  }

  int width = 0;
  if (!impl_->PrepareInput(image, width)) {
    return impl_->Fail(HZ_IMGFORMATERROR,
                       "Input image must be an 8-bit 1, 3, or 4 channel image");
  }
  const plate_recognition::TensorRTShape input_shape =
      MakeShape(kInputHeight, width);
  if (!impl_->recognizer_.Infer(impl_->input_, input_shape, impl_->output_,
                                impl_->output_shape_)) {
    return impl_->Fail(HZ_ERROR, impl_->recognizer_.LastError());
  }

  std::size_t time_steps = 0;
  std::size_t class_count = 0;
  if (!ReadOutputShape(impl_->output_, impl_->output_shape_, time_steps,
                       class_count) ||
      class_count != impl_->class_count_ ||
      !plate_recognition_ocr_v6::detail::DecodeAndFilterCtc(
          impl_->output_.data(), time_steps, class_count, plate_text)) {
    return impl_->Fail(HZ_ERROR,
                       "Model output shape or CTC class count is invalid");
  }
  return HZ_SUCCESS;
}

HZFLAG PlateRecognitionOCRV6::Release() {
  std::lock_guard<std::mutex> lock(impl_->mutex_);
  impl_->ResetUnlocked();
  impl_->last_error_.clear();
  return HZ_SUCCESS;
}

std::string PlateRecognitionOCRV6::LastError() const {
  std::lock_guard<std::mutex> lock(impl_->mutex_);
  return impl_->last_error_;
}
