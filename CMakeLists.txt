
cmake_minimum_required(VERSION 3.10)

project(PlateAlgorithmDemo LANGUAGES C CXX)

include(CTest)

# tensorrt 目录
# tensorrt path
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(_PLATE_DEFAULT_TENSORRT_INCLUDE "/usr/include/aarch64-linux-gnu")
    set(_PLATE_DEFAULT_TENSORRT_LIB "/usr/lib/aarch64-linux-gnu")
else()
    set(_PLATE_DEFAULT_TENSORRT_INCLUDE "/home/pcb/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/include")
    set(_PLATE_DEFAULT_TENSORRT_LIB "/home/pcb/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/lib")
endif()

# Keep the project defaults while allowing toolchains and command-line -D
# values to select another installation of the same required TRT version.
if(NOT DEFINED TensorRT_INCLUDE)
    set(TensorRT_INCLUDE "${_PLATE_DEFAULT_TENSORRT_INCLUDE}" CACHE PATH
        "TensorRT 8.2.5.1 include directory")
else()
    set(TensorRT_INCLUDE "${TensorRT_INCLUDE}" CACHE PATH
        "TensorRT 8.2.5.1 include directory" FORCE)
endif()
if(NOT DEFINED TensorRT_LIB)
    set(TensorRT_LIB "${_PLATE_DEFAULT_TENSORRT_LIB}" CACHE PATH
        "TensorRT 8.2.5.1 library directory")
else()
    set(TensorRT_LIB "${TensorRT_LIB}" CACHE PATH
        "TensorRT 8.2.5.1 library directory" FORCE)
endif()
unset(_PLATE_DEFAULT_TENSORRT_INCLUDE)
unset(_PLATE_DEFAULT_TENSORRT_LIB)

#添加子目录
add_subdirectory("PLateDetection_yolov5")
add_subdirectory("PLateDetection_yolov7")
#add_subdirectory("PLateDetection_yolov8")

# Recognition-only backends.  All three use the same TensorRT 8.2.5.1
# include/library roots defined above and share common/TensorRTRecognizer.cpp.
add_subdirectory("PlateRecognitionOCRV5")
add_subdirectory("PlateRecognitionOCRV6")
add_subdirectory("PlateRecognition_LPRNet")

# The combined detector/recognizer links the recognition targets above, so it
# must be added after those targets have been defined.
add_subdirectory("PlateDetectionRecognition")

