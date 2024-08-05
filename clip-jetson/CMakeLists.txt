cmake_minimum_required(VERSION 2.6)

project(JetsonClip)

add_definitions(-std=c++14)

set(TARGET_NAME "clip")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
include_directories((${OpenCV_INCLUDE_DIRS}))

find_library(NVONNXPARSERS_LIBRARY NAMES nvonnxparser HINTS /usr/lib/aarch64-linux-gnu)
find_library(NVPARSERS_LIBRARY NAMES nvparsers HINTS /usr/lib/aarch64-linux-gnu)


find_package(ZLIB REQUIRED)

set(CNPY_LIBRARY /usr/local/lib/libcnpy.a)
set(CNPY_INCLUDE_DIR /usr/local/include)

FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/sampleOpenclip.cpp ${PROJECT_SOURCE_DIR}/utils/*.cpp   ${PROJECT_SOURCE_DIR}/include/*.h)

add_executable(${TARGET_NAME} ${SRC_FILES})
target_link_libraries(${TARGET_NAME} nvinfer cudart ${OpenCV_LIBS} ${CNPY_LIBRARY} ${ZLIB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} dl ${NVPARSERS_LIBRARY} ${NVONNXPARSERS_LIBRARY})

add_definitions(-pthread -g)