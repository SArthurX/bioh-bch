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
set(BCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/bch)

# BCH library
add_library(bch_codec STATIC
    ${BCH_DIR}/bch_codec.c
)
target_include_directories(bch_codec PUBLIC ${BCH_DIR})

set(COMMON_SOURCES
    ${SRC_DIR}/base.cpp
    ${SRC_DIR}/mtcnn.cpp
    ${SRC_DIR}/arcface.cpp
    # ${SRC_DIR}/config.cpp
)

# Original main executable
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

# BioHash-BCH demo executable
set(BIOHASH_BCH_SOURCES
    ${SRC_DIR}/arcface_biohash_bch_demo.cpp
    ${COMMON_SOURCES}
)
add_executable(arcface_biohash_bch_demo ${BIOHASH_BCH_SOURCES})

target_link_libraries(arcface_biohash_bch_demo
    ${OpenCV_LIBS}
    ncnn
    bch_codec
    m
)

target_include_directories(arcface_biohash_bch_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${BCH_DIR}
)