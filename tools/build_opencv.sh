#!/bin/bash

set -e
PROJECT_ROOT="$(cd "$(dirname $0)/.." && pwd)"
OPENCV_DIR="${PROJECT_ROOT}/lib/opencv"
BUILD_DIR="${OPENCV_DIR}/build"
INSTALL_DIR="${BUILD_DIR}/install"

echo "=========================================="
echo "Compile OpenCV"
echo "=========================================="

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ${OPENCV_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DWITH_GTK=OFF \
    -DWITH_QT=OFF \
    -DWITH_OPENGL=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_V4L=OFF \
    -DWITH_LIBV4L=OFF \
    -DWITH_IPP=OFF \
    -DWITH_ITT=OFF \
    -DWITH_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_JPEG=ON \
    -DBUILD_PNG=ON \
    -DBUILD_LIST="core,imgproc,imgcodecs" \
    -DCPU_BASELINE="" \
    -DCPU_DISPATCH="" \
    -DCV_ENABLE_INTRINSICS=OFF \
    -DENABLE_PRECOMPILED_HEADERS=OFF \
    

make -j$(nproc)
make install

echo "=========================================="
echo "OpenCV Compilation done！"
echo "Install PATH: ${INSTALL_DIR}"
echo "=========================================="