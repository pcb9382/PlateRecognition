#pragma once

#include "NvInferVersion.h"

#if NV_TENSORRT_MAJOR != 8 || NV_TENSORRT_MINOR != 2 ||                        \
    NV_TENSORRT_PATCH != 5 || NV_TENSORRT_BUILD != 1
#error "PlateRecognition OCR modules require TensorRT 8.2.5.1"
#endif

namespace plate_recognition {

inline const char *TensorRTRequiredVersion() { return "8.2.5.1"; }

} // namespace plate_recognition
