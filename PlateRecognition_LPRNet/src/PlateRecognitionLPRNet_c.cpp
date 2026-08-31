#include "PlateRecognitionLPRNet_c.h"

#include "PlateRecognitionLPRNet.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <exception>
#include <new>
#include <string>

#include <opencv2/core/mat.hpp>

struct PlateRecognitionLPRNetHandle {
  PlateRecognitionLPRNet recognizer;
  bool initialized = false;
  std::string c_api_error;
};

namespace {

int ToCStatus(HZFLAG status, const std::string &error) {
  if (status == HZ_SUCCESS) {
    return PLATE_LPRNET_STATUS_SUCCESS;
  }
  if (status == HZ_IMGEMPTY || status == HZ_IMGFORMATERROR) {
    return PLATE_LPRNET_STATUS_IMAGE_ERROR;
  }
  if (status == HZ_FILEOPENFAILED || status == HZ_WITHOUTMODEL) {
    return PLATE_LPRNET_STATUS_IO_ERROR;
  }
  std::string lowercase_error = error;
  std::transform(lowercase_error.begin(), lowercase_error.end(),
                 lowercase_error.begin(), [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (lowercase_error.find("cuda") != std::string::npos) {
    return PLATE_LPRNET_STATUS_CUDA_ERROR;
  }
  return PLATE_LPRNET_STATUS_TENSORRT_ERROR;
}

} // namespace

PlateRecognitionLPRNetHandle *plate_lprnet_create(void) {
  try {
    return new (std::nothrow) PlateRecognitionLPRNetHandle();
  } catch (...) {
    return nullptr;
  }
}

int plate_lprnet_init(PlateRecognitionLPRNetHandle *handle,
                      const PlateRecognitionLPRNetCConfig *config) {
  if (handle == nullptr || config == nullptr ||
      config->struct_size < sizeof(PlateRecognitionLPRNetCConfig) ||
      config->model_path == nullptr || config->model_path[0] == '\0' ||
      config->gpu_id < 0) {
    return PLATE_LPRNET_STATUS_INVALID_ARGUMENT;
  }

  try {
    handle->c_api_error.clear();
    PlateRecognitionLPRNetConfig cpp_config;
    cpp_config.model_path = config->model_path;
    if (config->engine_cache_path != nullptr) {
      cpp_config.engine_cache_path = config->engine_cache_path;
    }
    cpp_config.gpu_id = config->gpu_id;
    cpp_config.enable_fp16 = config->enable_fp16 != 0;
    cpp_config.workspace_size = config->workspace_size == 0
                                    ? static_cast<size_t>(1U << 30)
                                    : config->workspace_size;

    const HZFLAG status = handle->recognizer.Init(cpp_config);
    handle->initialized = status == HZ_SUCCESS;
    return ToCStatus(status, handle->recognizer.LastError());
  } catch (const std::exception &exception) {
    handle->initialized = false;
    handle->c_api_error = exception.what();
    return PLATE_LPRNET_STATUS_TENSORRT_ERROR;
  } catch (...) {
    handle->initialized = false;
    handle->c_api_error = "Unknown exception during LPRNet initialization";
    return PLATE_LPRNET_STATUS_TENSORRT_ERROR;
  }
}

int plate_lprnet_run(PlateRecognitionLPRNetHandle *handle,
                     const uint8_t *image_data, int32_t width, int32_t height,
                     size_t stride_bytes, char *plate_text,
                     size_t plate_capacity, size_t *plate_length) {
  if (plate_text != nullptr && plate_capacity > 0U) {
    plate_text[0] = '\0';
  }
  if (plate_length != nullptr) {
    *plate_length = 0U;
  }
  if (handle == nullptr || image_data == nullptr || width <= 0 || height <= 0 ||
      stride_bytes < static_cast<size_t>(width) * 3U ||
      plate_length == nullptr) {
    return PLATE_LPRNET_STATUS_INVALID_ARGUMENT;
  }
  if (!handle->initialized) {
    return PLATE_LPRNET_STATUS_NOT_INITIALIZED;
  }

  try {
    handle->c_api_error.clear();
    cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t *>(image_data),
                  stride_bytes);
    std::string result;
    const HZFLAG status = handle->recognizer.Run(image, result);
    if (status != HZ_SUCCESS) {
      *plate_length = 0;
      return ToCStatus(status, handle->recognizer.LastError());
    }

    *plate_length = result.size();
    if (plate_text == nullptr || plate_capacity <= result.size()) {
      return PLATE_LPRNET_STATUS_BUFFER_TOO_SMALL;
    }
    std::memcpy(plate_text, result.data(), result.size());
    plate_text[result.size()] = '\0';
    return PLATE_LPRNET_STATUS_SUCCESS;
  } catch (const std::exception &exception) {
    handle->c_api_error = exception.what();
    *plate_length = 0;
    return PLATE_LPRNET_STATUS_TENSORRT_ERROR;
  } catch (...) {
    handle->c_api_error = "Unknown exception during LPRNet inference";
    *plate_length = 0;
    return PLATE_LPRNET_STATUS_TENSORRT_ERROR;
  }
}

int plate_lprnet_release(PlateRecognitionLPRNetHandle *handle) {
  if (handle == nullptr) {
    return PLATE_LPRNET_STATUS_INVALID_ARGUMENT;
  }
  handle->initialized = false;
  handle->c_api_error.clear();
  return ToCStatus(handle->recognizer.Release(),
                   handle->recognizer.LastError());
}

const char *
plate_lprnet_last_error(const PlateRecognitionLPRNetHandle *handle) {
  if (handle == nullptr) {
    return "Invalid LPRNet handle";
  }
  if (!handle->c_api_error.empty()) {
    return handle->c_api_error.c_str();
  }
  return handle->recognizer.LastError().c_str();
}

void plate_lprnet_destroy(PlateRecognitionLPRNetHandle *handle) {
  delete handle;
}
