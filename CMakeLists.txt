
cmake_minimum_required (VERSION 2.6)

project (PLateAlgorithmDemo)

# tensorrt 目录
# tensorrt path
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(TensorRT_INCLUDE "/usr/include/aarch64-linux-gnu" CACHE INTERNAL "TensorRT Library include location")
    set(TensorRT_LIB "/usr/lib/aarch64-linux-gnu" CACHE INTERNAL "TensorRT Library lib location")
else()
    set(TensorRT_INCLUDE "/home/pcb/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/include" CACHE INTERNAL "TensorRT Library include location")
    set(TensorRT_LIB "/home/pcb/TensorRT-8.2.5.1/targets/x86_64-linux-gnu/lib" CACHE INTERNAL "TensorRT Library lib location")
endif()

#添加子目录
add_subdirectory("PLateDetection_yolov5")
add_subdirectory("PLateDetection_yolov7")
#add_subdirectory("PLateDetection_yolov8")
add_subdirectory("PlateDetectionRecognition")
add_subdirectory("PlateRecognition")

