#include "PlateRecognitionOCRV5.h"

#include "OCRV5Text.h"
#include "TensorRTRecognizer.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

namespace {

const int kInputHeight = 48;
const int kInputChannels = 3;
const int kWidthAlignment = 20;

bool IsKnownModelVariant(PlateRecognitionOCRV5ModelVariant variant) {
  return variant == PlateRecognitionOCRV5ModelVariant::kMobile ||
         variant == PlateRecognitionOCRV5ModelVariant::kServer;
}

plate_recognition_ocr_v5_internal::EmbeddedModelVariant
ToEmbeddedVariant(PlateRecognitionOCRV5ModelVariant variant) {
  return variant == PlateRecognitionOCRV5ModelVariant::kMobile
             ? plate_recognition_ocr_v5_internal::EmbeddedModelVariant::kMobile
             : plate_recognition_ocr_v5_internal::EmbeddedModelVariant::kServer;
}

const char *ModelVariantName(PlateRecognitionOCRV5ModelVariant variant) {
  return variant == PlateRecognitionOCRV5ModelVariant::kMobile ? "mobile"
                                                               : "server";
}

bool FileExists(const std::string &path) {
  std::ifstream file(path.c_str(), std::ios::binary);
  return file.good();
}

std::string DeriveEnginePath(const PlateRecognitionOCRV5Config &config) {
  const std::size_t slash = config.onnx_model_path.find_last_of("/\\");
  const std::size_t dot = config.onnx_model_path.find_last_of('.');
  const bool has_extension =
      dot != std::string::npos && (slash == std::string::npos || dot > slash);
  const std::string base = has_extension ? config.onnx_model_path.substr(0, dot)
                                         : config.onnx_model_path;

  std::ostringstream path;
  path << base << "_" << ModelVariantName(config.model_variant)
       << "_trt8251_" << (config.enable_fp16 ? "fp16" : "fp32") << "_w"
       << config.min_input_width << "_" << config.opt_input_width << "_"
       << config.max_input_width << ".engine";
  return path.str();
}

plate_recognition::TensorRTShape MakeShape(int width) {
  plate_recognition::TensorRTShape shape;
  shape.n = 1;
  shape.c = kInputChannels;
  shape.h = kInputHeight;
  shape.w = width;
  return shape;
}

bool CheckedTensorSize(const std::vector<int> &shape, std::size_t &size) {
  size = 1U;
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] <= 0) {
      return false;
    }
    const std::size_t dimension = static_cast<std::size_t>(shape[index]);
    if (size > std::numeric_limits<std::size_t>::max() / dimension) {
      return false;
    }
    size *= dimension;
  }
  return true;
}

bool ValidateOutputShape(const std::vector<int> &shape, std::size_t output_size,
                         std::size_t &time_steps, std::size_t &class_count,
                         std::string &error) {
  time_steps = 0U;
  class_count = 0U;
  if (shape.size() != 3U || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 1) {
    error = "PP-OCRv5 output must have shape [1, time, classes]";
    return false;
  }

  std::size_t expected_size = 0U;
  if (!CheckedTensorSize(shape, expected_size) ||
      expected_size != output_size) {
    error = "PP-OCRv5 output shape and tensor size do not match";
    return false;
  }
  time_steps = static_cast<std::size_t>(shape[1]);
  class_count = static_cast<std::size_t>(shape[2]);
  return true;
}

bool PreprocessBgr(const cv::Mat &image, int width, std::vector<float> &tensor,
                   std::string &error) {
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(width, kInputHeight), 0.0, 0.0,
             cv::INTER_LINEAR);
  if (resized.empty() || resized.type() != CV_8UC3) {
    error = "OpenCV failed to create a CV_8UC3 recognition input";
    return false;
  }

  const std::size_t plane =
      static_cast<std::size_t>(kInputHeight) * static_cast<std::size_t>(width);
  if (plane > std::numeric_limits<std::size_t>::max() / 3U) {
    error = "PP-OCRv5 input tensor size overflows size_t";
    return false;
  }
  tensor.resize(plane * 3U);
  float *blue = tensor.data();
  float *green = blue + plane;
  float *red = green + plane;
  const float scale = 1.0F / 255.0F;

  for (int row = 0; row < kInputHeight; ++row) {
    const cv::Vec3b *pixels = resized.ptr<cv::Vec3b>(row);
    const std::size_t offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
    for (int column = 0; column < width; ++column) {
      const std::size_t index = offset + static_cast<std::size_t>(column);
      blue[index] = static_cast<float>(pixels[column][0]) * scale;
      green[index] = static_cast<float>(pixels[column][1]) * scale;
      red[index] = static_cast<float>(pixels[column][2]) * scale;
    }
  }
  return true;
}

} // namespace

struct PlateRecognitionOCRV5::Impl {
  mutable std::mutex mutex;
  plate_recognition::TensorRTRecognizer recognizer;
  PlateRecognitionOCRV5Config config;
  plate_recognition_ocr_v5_internal::EmbeddedModelVariant mapping_variant =
      plate_recognition_ocr_v5_internal::EmbeddedModelVariant::kServer;
  std::size_t class_count = 0U;
  std::string last_error;
  bool initialized = false;

  void Reset() {
    recognizer.Release();
    mapping_variant =
        plate_recognition_ocr_v5_internal::EmbeddedModelVariant::kServer;
    class_count = 0U;
    initialized = false;
  }

  bool ValidateConfig(const PlateRecognitionOCRV5Config &candidate) {
    if (candidate.onnx_model_path.empty()) {
      last_error = "ONNX model path must not be empty";
      return false;
    }
    if (!IsKnownModelVariant(candidate.model_variant)) {
      last_error = "PP-OCRv5 model variant must be mobile or server";
      return false;
    }
    if (candidate.gpu_id < 0 || candidate.workspace_size == 0U) {
      last_error = "GPU id and TensorRT workspace configuration are invalid";
      return false;
    }
    if (candidate.min_input_width < kWidthAlignment ||
        candidate.min_input_width > candidate.opt_input_width ||
        candidate.opt_input_width > candidate.max_input_width ||
        candidate.min_input_width % kWidthAlignment != 0 ||
        candidate.opt_input_width % kWidthAlignment != 0 ||
        candidate.max_input_width % kWidthAlignment != 0) {
      last_error = "input widths must be ordered positive multiples of 20";
      return false;
    }
    return true;
  }
};

PlateRecognitionOCRV5::PlateRecognitionOCRV5() : impl_(new Impl()) {}

PlateRecognitionOCRV5::~PlateRecognitionOCRV5() = default;

HZFLAG PlateRecognitionOCRV5::Init(const PlateRecognitionOCRV5Config *config) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->Reset();
  impl_->last_error.clear();

  if (config == NULL || !impl_->ValidateConfig(*config)) {
    if (config == NULL) {
      impl_->last_error = "PlateRecognitionOCRV5Config is null";
    }
    return HZ_CONFIGLOADFAILED;
  }
  impl_->config = *config;
  impl_->mapping_variant = ToEmbeddedVariant(impl_->config.model_variant);

  const std::string engine_path = impl_->config.engine_cache_path.empty()
                                      ? DeriveEnginePath(impl_->config)
                                      : impl_->config.engine_cache_path;
  if (!FileExists(engine_path) && !FileExists(impl_->config.onnx_model_path)) {
    impl_->last_error =
        "PP-OCRv5 ONNX model and TensorRT engine are both missing";
    return HZ_WITHOUTMODEL;
  }

  plate_recognition::TensorRTRecognizerConfig trt_config;
  trt_config.onnx_path = impl_->config.onnx_model_path;
  trt_config.engine_path = engine_path;
  trt_config.gpu_id = impl_->config.gpu_id;
  trt_config.enable_fp16 = impl_->config.enable_fp16;
  trt_config.workspace_size = impl_->config.workspace_size;
  trt_config.min_shape = MakeShape(impl_->config.min_input_width);
  trt_config.opt_shape = MakeShape(impl_->config.opt_input_width);
  trt_config.max_shape = MakeShape(impl_->config.max_input_width);

  try {
    if (!impl_->recognizer.Init(trt_config)) {
      impl_->last_error = impl_->recognizer.LastError();
      impl_->Reset();
      return HZ_INITFAILED;
    }

    // One zero-input inference validates the dynamic profile, output layout,
    // exact native class count and selected embedded mapping up front.
    const plate_recognition::TensorRTShape warm_shape =
        MakeShape(impl_->config.opt_input_width);
    const std::size_t warm_count = static_cast<std::size_t>(warm_shape.n) *
                                   static_cast<std::size_t>(warm_shape.c) *
                                   static_cast<std::size_t>(warm_shape.h) *
                                   static_cast<std::size_t>(warm_shape.w);
    std::vector<float> warm_input(warm_count, 0.0F);
    std::vector<float> warm_output;
    std::vector<int> warm_output_shape;
    if (!impl_->recognizer.Infer(warm_input, warm_shape, warm_output,
                                 warm_output_shape)) {
      impl_->last_error = impl_->recognizer.LastError();
      impl_->Reset();
      return HZ_INITFAILED;
    }

    std::size_t time_steps = 0U;
    std::size_t class_count = 0U;
    if (!ValidateOutputShape(warm_output_shape, warm_output.size(), time_steps,
                             class_count, impl_->last_error)) {
      impl_->Reset();
      return HZ_INITFAILED;
    }
    if (class_count !=
        plate_recognition_ocr_v5_internal::kOCRV5OutputClassCount) {
      impl_->last_error =
          "PP-OCRv5 model must expose exactly 18,385 output classes";
      impl_->Reset();
      return HZ_INITFAILED;
    }
    if (!plate_recognition_ocr_v5_internal::ValidateEmbeddedMapping(
            impl_->mapping_variant, impl_->last_error)) {
      impl_->Reset();
      return HZ_INITFAILED;
    }
    (void)time_steps;
    impl_->class_count = class_count;
    impl_->initialized = true;
    return HZ_SUCCESS;
  } catch (const std::exception &exception) {
    impl_->last_error = exception.what();
    impl_->Reset();
    return HZ_INITFAILED;
  } catch (...) {
    impl_->last_error = "unknown error while initializing PP-OCRv5 TensorRT";
    impl_->Reset();
    return HZ_INITFAILED;
  }
}

HZFLAG PlateRecognitionOCRV5::Run(const cv::Mat &image,
                                  std::string &plate_text) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  plate_text.clear();
  impl_->last_error.clear();

  if (!impl_->initialized || !impl_->recognizer.IsInitialized()) {
    impl_->last_error = "PlateRecognitionOCRV5 is not initialized";
    return HZ_INITFAILED;
  }
  if (image.empty()) {
    impl_->last_error = "input image is empty";
    return HZ_IMGEMPTY;
  }
  if (image.type() != CV_8UC3) {
    impl_->last_error = "PP-OCRv5 expects an 8-bit three-channel BGR image";
    return HZ_IMGFORMATERROR;
  }

  const int input_width = plate_recognition_ocr_v5_internal::ComputeInputWidth(
      image.rows, image.cols, kInputHeight, impl_->config.min_input_width,
      impl_->config.max_input_width);
  if (input_width <= 0) {
    impl_->last_error = "failed to calculate the PP-OCRv5 dynamic input width";
    return HZ_IMGFORMATERROR;
  }

  try {
    std::vector<float> input;
    if (!PreprocessBgr(image, input_width, input, impl_->last_error)) {
      return HZ_IMGFORMATERROR;
    }

    std::vector<float> output;
    std::vector<int> output_shape;
    if (!impl_->recognizer.Infer(input, MakeShape(input_width), output,
                                 output_shape)) {
      impl_->last_error = impl_->recognizer.LastError();
      return HZ_ERROR;
    }

    std::size_t time_steps = 0U;
    std::size_t class_count = 0U;
    if (!ValidateOutputShape(output_shape, output.size(), time_steps,
                             class_count, impl_->last_error) ||
        class_count != impl_->class_count) {
      if (impl_->last_error.empty()) {
        impl_->last_error = "PP-OCRv5 class count changed between inferences";
      }
      return HZ_ERROR;
    }
    if (!plate_recognition_ocr_v5_internal::DecodeCtc(
            output.data(), time_steps, class_count, impl_->mapping_variant,
            plate_text, impl_->last_error)) {
      plate_text.clear();
      return HZ_ERROR;
    }
    return HZ_SUCCESS;
  } catch (const cv::Exception &exception) {
    impl_->last_error = exception.what();
    plate_text.clear();
    return HZ_IMGFORMATERROR;
  } catch (const std::exception &exception) {
    impl_->last_error = exception.what();
    plate_text.clear();
    return HZ_ERROR;
  } catch (...) {
    impl_->last_error = "unknown error during PP-OCRv5 TensorRT inference";
    plate_text.clear();
    return HZ_ERROR;
  }
}

HZFLAG PlateRecognitionOCRV5::Release() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->Reset();
  impl_->last_error.clear();
  return HZ_SUCCESS;
}

std::string PlateRecognitionOCRV5::LastError() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->last_error;
}
