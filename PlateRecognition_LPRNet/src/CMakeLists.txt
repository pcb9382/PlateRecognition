cmake_minimum_required(VERSION 3.8)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if(NOT TensorRT_INCLUDE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(TensorRT_INCLUDE "/usr/include/aarch64-linux-gnu" CACHE PATH "TensorRT include directory")
    else()
        set(TensorRT_INCLUDE "/home/pcb/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/include" CACHE PATH "TensorRT include directory")
    endif()
endif()

if(NOT TensorRT_LIB)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(TensorRT_LIB "/usr/lib/aarch64-linux-gnu" CACHE PATH "TensorRT library directory")
    else()
        set(TensorRT_LIB "/home/pcb/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/lib" CACHE PATH "TensorRT library directory")
    endif()
endif()

find_path(PLATE_LPRNET_TRT_INCLUDE_DIR NvInfer.h
    HINTS "${TensorRT_INCLUDE}")
find_library(PLATE_LPRNET_NVINFER_LIBRARY nvinfer
    HINTS "${TensorRT_LIB}")
find_library(PLATE_LPRNET_NVONNXPARSER_LIBRARY nvonnxparser
    HINTS "${TensorRT_LIB}")
find_library(PLATE_LPRNET_NVINFER_PLUGIN_LIBRARY nvinfer_plugin
    HINTS "${TensorRT_LIB}")

if(NOT PLATE_LPRNET_TRT_INCLUDE_DIR
   OR NOT PLATE_LPRNET_NVINFER_LIBRARY
   OR NOT PLATE_LPRNET_NVONNXPARSER_LIBRARY
   OR NOT PLATE_LPRNET_NVINFER_PLUGIN_LIBRARY)
    message(FATAL_ERROR "TensorRT 8.2.5.1 headers or libraries were not found")
endif()

add_library(PlateRecognition_LPRNet SHARED
    PlateRecognitionLPRNet.cpp
    PlateRecognitionLPRNet_c.cpp
    "${CMAKE_CURRENT_LIST_DIR}/../../common/TensorRTRecognizer.cpp")

target_compile_features(PlateRecognition_LPRNet PUBLIC cxx_std_11)
target_compile_definitions(PlateRecognition_LPRNet PRIVATE PLATE_LPRNET_BUILD_SHARED)
if(MSVC)
    target_compile_options(PlateRecognition_LPRNet PRIVATE /W4 /utf-8)
else()
    target_compile_options(PlateRecognition_LPRNet PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_include_directories(PlateRecognition_LPRNet
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/../include"
        "${CMAKE_CURRENT_LIST_DIR}/../../include"
        "${OpenCV_INCLUDE_DIRS}"
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../../common"
        "${PLATE_LPRNET_TRT_INCLUDE_DIR}"
        "${CUDA_INCLUDE_DIRS}"
    )

target_link_libraries(PlateRecognition_LPRNet
    PRIVATE
        "${PLATE_LPRNET_NVINFER_LIBRARY}"
        "${PLATE_LPRNET_NVONNXPARSER_LIBRARY}"
        "${PLATE_LPRNET_NVINFER_PLUGIN_LIBRARY}"
        "${CUDA_CUDART_LIBRARY}"
        ${OpenCV_LIBS})

set_target_properties(PlateRecognition_LPRNet PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../lib")
