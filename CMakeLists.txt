cmake_minimum_required(VERSION 3.5.2)

project (TestBed)

set (INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

include_directories(${INCLUDE_DIR})

add_subdirectory(src/OpenCVEx)
add_subdirectory(src/OpenCVExTest)
