cmake_minimum_required(VERSION 3.10)

add_subdirectory(src)

option(PLATE_RECOGNITION_OCRV6_BUILD_TESTS
       "Build PlateRecognitionOCRV6 demo and decoder tests" ON)

if(PLATE_RECOGNITION_OCRV6_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
