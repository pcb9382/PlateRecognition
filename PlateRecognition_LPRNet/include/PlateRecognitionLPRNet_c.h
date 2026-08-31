#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(PLATE_LPRNET_BUILD_SHARED)
#define PLATE_LPRNET_C_API __declspec(dllexport)
#elif defined(_WIN32)
#define PLATE_LPRNET_C_API __declspec(dllimport)
#else
#define PLATE_LPRNET_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PlateRecognitionLPRNetHandle PlateRecognitionLPRNetHandle;

typedef enum PlateRecognitionLPRNetStatus {
  PLATE_LPRNET_STATUS_SUCCESS = 0,
  PLATE_LPRNET_STATUS_INVALID_ARGUMENT = -1,
  PLATE_LPRNET_STATUS_NOT_INITIALIZED = -2,
  PLATE_LPRNET_STATUS_IO_ERROR = -3,
  PLATE_LPRNET_STATUS_TENSORRT_ERROR = -4,
  PLATE_LPRNET_STATUS_CUDA_ERROR = -5,
  PLATE_LPRNET_STATUS_IMAGE_ERROR = -6,
  PLATE_LPRNET_STATUS_BUFFER_TOO_SMALL = -7,
  PLATE_LPRNET_STATUS_OUT_OF_MEMORY = -8
} PlateRecognitionLPRNetStatus;

typedef struct PlateRecognitionLPRNetCConfig {
  size_t struct_size;
  const char *model_path;
  const char *engine_cache_path;
  int32_t gpu_id;
  int32_t enable_fp16;
  size_t workspace_size;
} PlateRecognitionLPRNetCConfig;

PLATE_LPRNET_C_API PlateRecognitionLPRNetHandle *plate_lprnet_create(void);

PLATE_LPRNET_C_API int
plate_lprnet_init(PlateRecognitionLPRNetHandle *handle,
                  const PlateRecognitionLPRNetCConfig *config);

// image_data is a BGR uint8 image. stride_bytes may be larger than width * 3.
// plate_length receives the UTF-8 byte count excluding the trailing NUL. If
// plate_capacity is too small, the required length is still returned.
PLATE_LPRNET_C_API int plate_lprnet_run(PlateRecognitionLPRNetHandle *handle,
                                        const uint8_t *image_data,
                                        int32_t width, int32_t height,
                                        size_t stride_bytes, char *plate_text,
                                        size_t plate_capacity,
                                        size_t *plate_length);

PLATE_LPRNET_C_API int
plate_lprnet_release(PlateRecognitionLPRNetHandle *handle);
PLATE_LPRNET_C_API const char *
plate_lprnet_last_error(const PlateRecognitionLPRNetHandle *handle);
PLATE_LPRNET_C_API void
plate_lprnet_destroy(PlateRecognitionLPRNetHandle *handle);

#ifdef __cplusplus
}
#endif
