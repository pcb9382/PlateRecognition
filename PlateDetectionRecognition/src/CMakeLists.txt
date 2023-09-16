cmake_minimum_required (VERSION 3.8)

project(PLateAlgorithm)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O0 -g")# -rdynamic -g2 -ggdb


SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/PlateDetectionRecognition/lib)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/PLateDetection_yolov5/include)
include_directories(${CMAKE_SOURCE_DIR}/PLateDetection_yolov7/include)
include_directories(${CMAKE_SOURCE_DIR}/PlateRecognition/include)
include_directories(${CMAKE_SOURCE_DIR}/PlateDetectionRecognition/include)
include_directories(${CMAKE_SOURCE_DIR}/common)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

file(GLOB lib_header "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
#file(GLOB common_header "${CMAKE_SOURCE_DIR}/common/*.h")

file(GLOB lib_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
#file(GLOB common_src "${CMAKE_SOURCE_DIR}/common/*.cpp")


include_directories(${TensorRT_INCLUDE})
link_directories(${TensorRT_LIB})

add_library(${PROJECT_NAME} SHARED ${lib_src})

target_link_libraries(${PROJECT_NAME} PLateDetection_yolov5)
target_link_libraries(${PROJECT_NAME} PLateDetection_yolov7)
target_link_libraries(${PROJECT_NAME} PlateRecognition)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
