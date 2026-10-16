cmake_minimum_required(VERSION 3.20)
project(yolo_trt_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorRT 根目录
set(TENSORRT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/TensorRT-10.7.0.23)

# CUDA 根目录（系统装的是 /usr/local/cuda-12.6）
set(CUDA_ROOT /usr/local/cuda-12.6)

include_directories(
    ${TENSORRT_ROOT}/include
    ${CUDA_ROOT}/include       # ← 关键：显式加 CUDA 头文件路径
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(
    ${TENSORRT_ROOT}/lib
    ${CUDA_ROOT}/lib64         # ← 关键：显式加 CUDA 库路径
)

find_package(OpenCV REQUIRED)

add_executable(yolo_trt_demo
    src/main.cpp
    src/trt_engine.cpp
)

target_link_libraries(yolo_trt_demo
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    ${OpenCV_LIBS}
)

target_compile_options(yolo_trt_demo PRIVATE -Wall -O2)