cmake_minimum_required(VERSION 3.10)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

get_filename_component(OCRV5_REPOSITORY_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(PlateRecognitionOCRV5_demo main.cpp)
target_compile_features(PlateRecognitionOCRV5_demo PRIVATE cxx_std_11)
target_compile_definitions(PlateRecognitionOCRV5_demo PRIVATE
    PLATE_RECOGNITION_OCRV5_DEFAULT_MODEL_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../models")
target_include_directories(PlateRecognitionOCRV5_demo PRIVATE "${OpenCV_INCLUDE_DIRS}")
target_link_libraries(PlateRecognitionOCRV5_demo PRIVATE PlateRecognitionOCRV5 ${OpenCV_LIBS})

add_executable(PlateRecognitionOCRV5_decode_test
    test_decode.cpp
    "${OCRV5_REPOSITORY_ROOT}/PlateRecognitionOCRV5/src/OCRV5Text.cpp"
)
target_compile_features(PlateRecognitionOCRV5_decode_test PRIVATE cxx_std_11)
target_include_directories(PlateRecognitionOCRV5_decode_test PRIVATE
    "${OCRV5_REPOSITORY_ROOT}/PlateRecognitionOCRV5/src"
    "${OCRV5_REPOSITORY_ROOT}/common"
)
if(MSVC)
    target_compile_options(PlateRecognitionOCRV5_demo PRIVATE /W4 /utf-8)
    target_compile_options(PlateRecognitionOCRV5_decode_test PRIVATE /W4 /utf-8)
else()
    target_compile_options(PlateRecognitionOCRV5_demo PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(PlateRecognitionOCRV5_decode_test PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME PlateRecognitionOCRV5_decode_test
         COMMAND PlateRecognitionOCRV5_decode_test)
