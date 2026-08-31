cmake_minimum_required(VERSION 3.8)

find_package(OpenCV REQUIRED)

add_executable(PlateRecognition_LPRNet_demo main.cpp)
target_compile_features(PlateRecognition_LPRNet_demo PRIVATE cxx_std_11)
target_link_libraries(PlateRecognition_LPRNet_demo PRIVATE PlateRecognition_LPRNet ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
    add_executable(PlateRecognition_LPRNet_decode_test decode_test.cpp)
    target_compile_features(PlateRecognition_LPRNet_decode_test PRIVATE cxx_std_11)
    target_link_libraries(PlateRecognition_LPRNet_decode_test PRIVATE PlateRecognition_LPRNet)
    add_test(NAME PlateRecognition_LPRNet_decode COMMAND PlateRecognition_LPRNet_decode_test)
endif()
