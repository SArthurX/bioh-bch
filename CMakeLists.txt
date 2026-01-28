cmake_minimum_required(VERSION 3.10)
project(bioh-bch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -fopenmp -pthread")

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/build/install/lib/cmake/opencv4" CACHE PATH "OpenCV config path")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(ncnn_DIR "${CMAKE_SOURCE_DIR}/lib/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "ncnn config path")
find_package(ncnn REQUIRED)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples/arcface)

set(COMMON_SOURCES
    ${SRC_DIR}/base.cpp
    ${SRC_DIR}/mtcnn.cpp
    ${SRC_DIR}/arcface.cpp
    # ${SRC_DIR}/config.cpp
)

set(MAIN_SOURCES
    ${SRC_DIR}/main.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/face_database.cpp
    ${COMMON_SOURCES}
)
add_executable(main ${MAIN_SOURCES})

target_link_libraries(main
    ${OpenCV_LIBS}
    ncnn
    m
)

target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    # ${CMAKE_CURRENT_SOURCE_DIR}/src
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty
)