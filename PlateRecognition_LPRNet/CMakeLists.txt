cmake_minimum_required(VERSION 3.8)

project(PlateRecognition_LPRNet LANGUAGES CXX)

option(PLATE_LPRNET_BUILD_TESTS "Build the LPRNet command-line demo and decode test" ON)

add_subdirectory(src)

if(PLATE_LPRNET_BUILD_TESTS)
    add_subdirectory(test)
endif()
