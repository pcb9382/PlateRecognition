cmake_minimum_required(VERSION 3.10)

project(PlateRecognitionOCRV6 LANGUAGES CXX)

get_filename_component(OCRV6_REPOSITORY_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(OCRV6_MODULE_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(CUDA REQUIRED)

if(NOT TensorRT_INCLUDE OR NOT EXISTS "${TensorRT_INCLUDE}/NvInferVersion.h")
    message(FATAL_ERROR "TensorRT_INCLUDE must point to TensorRT 8.2.5.1 headers")
endif()
if(NOT TensorRT_LIB OR NOT EXISTS "${TensorRT_LIB}")
    message(FATAL_ERROR "TensorRT_LIB must point to TensorRT 8.2.5.1 libraries")
endif()

file(READ "${TensorRT_INCLUDE}/NvInferVersion.h" OCRV6_TRT_VERSION_HEADER)
if(NOT OCRV6_TRT_VERSION_HEADER MATCHES "#define[ \t]+NV_TENSORRT_MAJOR[ \t]+8"
   OR NOT OCRV6_TRT_VERSION_HEADER MATCHES "#define[ \t]+NV_TENSORRT_MINOR[ \t]+2"
   OR NOT OCRV6_TRT_VERSION_HEADER MATCHES "#define[ \t]+NV_TENSORRT_PATCH[ \t]+5"
   OR NOT OCRV6_TRT_VERSION_HEADER MATCHES "#define[ \t]+NV_TENSORRT_BUILD[ \t]+1")
    message(FATAL_ERROR "PlateRecognitionOCRV6 requires TensorRT 8.2.5.1")
endif()

find_library(PlateRecognitionOCRV6_NVINFER
    NAMES nvinfer libnvinfer
    HINTS "${TensorRT_LIB}"
    NO_DEFAULT_PATH)
find_library(PlateRecognitionOCRV6_NVONNXPARSER
    NAMES nvonnxparser libnvonnxparser
    HINTS "${TensorRT_LIB}"
    NO_DEFAULT_PATH)
find_library(PlateRecognitionOCRV6_NVINFER_PLUGIN
    NAMES nvinfer_plugin libnvinfer_plugin
    HINTS "${TensorRT_LIB}"
    NO_DEFAULT_PATH)

if(NOT PlateRecognitionOCRV6_NVINFER OR
   NOT PlateRecognitionOCRV6_NVONNXPARSER OR
   NOT PlateRecognitionOCRV6_NVINFER_PLUGIN)
    message(FATAL_ERROR "TensorRT 8.2.5.1 libraries were not found in ${TensorRT_LIB}")
endif()

set(PLATE_PPOCRV6_TARGET PlateRecognition_PPOCRV6)

add_library(${PLATE_PPOCRV6_TARGET} SHARED
    PlateRecognitionOCRV6.cpp
    OCRV6Text.cpp
    "${OCRV6_REPOSITORY_ROOT}/common/TensorRTRecognizer.cpp")

# Backward-compatible alias for the module's original target name.
add_library(PlateRecognitionOCRV6 ALIAS ${PLATE_PPOCRV6_TARGET})

target_compile_features(${PLATE_PPOCRV6_TARGET} PUBLIC cxx_std_11)

target_include_directories(${PLATE_PPOCRV6_TARGET}
    PUBLIC
        "${OCRV6_MODULE_ROOT}/include"
        "${OCRV6_REPOSITORY_ROOT}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${OCRV6_REPOSITORY_ROOT}/common"
        "${TensorRT_INCLUDE}"
        "${CUDA_INCLUDE_DIRS}"
        ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PLATE_PPOCRV6_TARGET}
    PRIVATE
        ${OpenCV_LIBS}
        ${CUDA_CUDART_LIBRARY}
        ${PlateRecognitionOCRV6_NVINFER}
        ${PlateRecognitionOCRV6_NVONNXPARSER}
        ${PlateRecognitionOCRV6_NVINFER_PLUGIN})

set_target_properties(${PLATE_PPOCRV6_TARGET} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${OCRV6_MODULE_ROOT}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${OCRV6_MODULE_ROOT}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${OCRV6_MODULE_ROOT}/lib")

if(MSVC)
    target_compile_options(${PLATE_PPOCRV6_TARGET} PRIVATE /W4 /utf-8)
else()
    target_compile_options(${PLATE_PPOCRV6_TARGET} PRIVATE -Wall -Wextra -Wpedantic)
endif()
