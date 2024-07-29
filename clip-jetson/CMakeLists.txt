cmake_minimum_required(VERSION 2.6)

project(JetsonClip)

add_definitions(-std=c++11)

set(TARGET_NAME "clip")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

include_directories(/usr/include/opencv4)
link_directories(/usr/lib/aarch64-linux-gnu)

FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/lenet.cpp ${PROJECT_SOURCE_DIR}/include/*.h)

add_executable(${TARGET_NAME} ${SRC_FILES})
target_link_libraries(${TARGET_NAME} nvinfer)
target_link_libraries(${TARGET_NAME} cudart)
target_link_libraries(${TARGET_NAME} opencv_core)

add_definitions(-O2 -pthread)