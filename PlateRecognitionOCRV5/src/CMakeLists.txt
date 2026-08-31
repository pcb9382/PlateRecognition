cmake_minimum_required(VERSION 3.10)

project(PlateRecognitionOCRV5 LANGUAGES CXX)

get_filename_component(OCRV5_REPOSITORY_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(CUDA REQUIRED)

if(NOT TensorRT_INCLUDE OR NOT EXISTS "${TensorRT_INCLUDE}/NvInferVersion.h")
    message(FATAL_ERROR "TensorRT_INCLUDE must point to TensorRT 8.2.5.1 headers")
endif()
if(NOT TensorRT_LIB OR NOT EXISTS "${TensorRT_LIB}")
    message(FATAL_ERROR "TensorRT_LIB must point to TensorRT 8.2.5.1 libraries")
endif()

file(READ "${TensorRT_INCLUDE}/NvInferVersion.h" OCRV5_TRT_VERSION_HEADER)
if(NOT OCRV5_TRT_VERSION_HEADER MATCHES "#define[ \t]+NV_TENSORRT_MAJOR[ \t]+8"
   OR NOT OCRV5_TRT_VERSION_HEADER MATCHES "#define[ \t]+NV_TENSORRT_MINOR[ \t]+2"
   OR NOT OCRV5_TRT_VERSION_HEADER MATCHES "#define[ \t]+NV_TENSORRT_PATCH[ \t]+5"
   OR NOT OCRV5_TRT_VERSION_HEADER MATCHES "#define[ \t]+NV_TENSORRT_BUILD[ \t]+1")
    message(FATAL_ERROR "PlateRecognitionOCRV5 requires TensorRT 8.2.5.1")
endif()

find_library(OCRV5_NVINFER_LIBRARY nvinfer
             PATHS "${TensorRT_LIB}" NO_DEFAULT_PATH)
find_library(OCRV5_NVINFER_PLUGIN_LIBRARY nvinfer_plugin
             PATHS "${TensorRT_LIB}" NO_DEFAULT_PATH)
find_library(OCRV5_NVONNXPARSER_LIBRARY nvonnxparser
             PATHS "${TensorRT_LIB}" NO_DEFAULT_PATH)
if(NOT OCRV5_NVINFER_LIBRARY OR NOT OCRV5_NVINFER_PLUGIN_LIBRARY
   OR NOT OCRV5_NVONNXPARSER_LIBRARY)
    message(FATAL_ERROR "TensorRT 8.2.5.1 nvinfer/parser/plugin libraries were not found")
endif()

set(PLATE_PPOCRV5_TARGET PlateRecognition_PPOCRV5)

add_library(${PLATE_PPOCRV5_TARGET} SHARED
    PlateRecognitionOCRV5.cpp
    OCRV5Text.cpp
    "${OCRV5_REPOSITORY_ROOT}/common/TensorRTRecognizer.cpp"
)

# Backward-compatible alias for the module's original target name.
add_library(PlateRecognitionOCRV5 ALIAS ${PLATE_PPOCRV5_TARGET})

target_compile_features(${PLATE_PPOCRV5_TARGET} PUBLIC cxx_std_11)
if(MSVC)
    target_compile_options(${PLATE_PPOCRV5_TARGET} PRIVATE /W4 /utf-8)
else()
    target_compile_options(${PLATE_PPOCRV5_TARGET} PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_include_directories(${PLATE_PPOCRV5_TARGET}
    PUBLIC
        "${OCRV5_REPOSITORY_ROOT}/PlateRecognitionOCRV5/include"
        "${OCRV5_REPOSITORY_ROOT}/include"
        "${OpenCV_INCLUDE_DIRS}"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${OCRV5_REPOSITORY_ROOT}/common"
        "${TensorRT_INCLUDE}"
        "${CUDA_INCLUDE_DIRS}"
)

target_link_libraries(${PLATE_PPOCRV5_TARGET}
    PRIVATE
        ${OpenCV_LIBS}
        ${CUDA_CUDART_LIBRARY}
        ${OCRV5_NVINFER_LIBRARY}
        ${OCRV5_NVINFER_PLUGIN_LIBRARY}
        ${OCRV5_NVONNXPARSER_LIBRARY}
)

set_target_properties(${PLATE_PPOCRV5_TARGET} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${OCRV5_REPOSITORY_ROOT}/PlateRecognitionOCRV5/lib"
    LIBRARY_OUTPUT_DIRECTORY "${OCRV5_REPOSITORY_ROOT}/PlateRecognitionOCRV5/lib"
    RUNTIME_OUTPUT_DIRECTORY "${OCRV5_REPOSITORY_ROOT}/PlateRecognitionOCRV5/lib"
)
