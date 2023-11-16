# CMakeList.txt: NuctechPersonBagBinding 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)
project(PlateDetectionRecognition_demo)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g")#-Wall 

find_package(OpenCV REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)


#源文件
file(GLOB test_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

#头文件
file(GLOB test_header "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
#头文件搜索目录
find_package(OpenCV REQUIRED)
include_directories(${CMAKE_SOURCE_DIR})
include_directories("${CMAKE_SOURCE_DIR}/PlateDetectionRecognition/include")


# 将源代码添加到此项目的可执行文件。
add_executable(${PROJECT_NAME} ${test_src} ${test_header})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PlateDetectionRecognition)