cmake_minimum_required(VERSION 3.10)

project(PlateRecognitionOCRV6Tests LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

get_filename_component(OCRV6_MODULE_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(OCRV6_REPOSITORY_ROOT
    "${OCRV6_MODULE_ROOT}/.." ABSOLUTE)

add_executable(PlateRecognitionOCRV6_demo main.cpp)
target_compile_features(PlateRecognitionOCRV6_demo PRIVATE cxx_std_11)
target_compile_definitions(PlateRecognitionOCRV6_demo PRIVATE
    PLATE_RECOGNITION_OCRV6_DEFAULT_MODEL_DIR="${OCRV6_MODULE_ROOT}/models/medium")
target_link_libraries(PlateRecognitionOCRV6_demo PRIVATE PlateRecognitionOCRV6 ${OpenCV_LIBS})

add_executable(PlateRecognitionOCRV6_unit
    PlateRecognitionOCRV6_unit.cpp
    "${OCRV6_MODULE_ROOT}/src/OCRV6Text.cpp")
target_compile_features(PlateRecognitionOCRV6_unit PRIVATE cxx_std_11)
target_include_directories(PlateRecognitionOCRV6_unit PRIVATE
    "${OCRV6_MODULE_ROOT}/src"
    "${OCRV6_REPOSITORY_ROOT}/common")

if(MSVC)
    target_compile_options(PlateRecognitionOCRV6_demo PRIVATE /W4 /utf-8)
    target_compile_options(PlateRecognitionOCRV6_unit PRIVATE /W4 /utf-8)
else()
    target_compile_options(PlateRecognitionOCRV6_demo PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(PlateRecognitionOCRV6_unit PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME PlateRecognitionOCRV6_unit COMMAND PlateRecognitionOCRV6_unit)
