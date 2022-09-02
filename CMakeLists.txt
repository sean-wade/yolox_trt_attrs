cmake_minimum_required(VERSION 3.10)

project(yolox_sgtls_s_725 VERSION 0.0.1)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

# find CUDA
find_package(CUDA)
message("CUDA Libs: ${CUDA_LIBRARIES}")
message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")

# find tensorRT
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)
message("NVINFER: ${NVINFER}")
message("NVPARSERS: ${NVPARSERS}")
message("NVONNXPARSERS: ${NVONNXPARSERS}")

# find CUDNN
find_library(CUDNN_LIBRARY
    NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
    PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES lib lib64 bin
    DOC "CUDNN library."
)
message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(yolox ${PROJECT_SOURCE_DIR}/yolox.cpp)
target_link_libraries(yolox nvinfer)
target_link_libraries(yolox ${CUDA_LIBRARIES})
target_link_libraries(yolox ${OpenCV_LIBS})


add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/detector_yolox.cpp ${PROJECT_SOURCE_DIR}/base_detector.cpp)
target_link_libraries(main nvinfer nvonnxparser)
target_link_libraries(main ${CUDA_LIBRARIES})
target_link_libraries(main ${OpenCV_LIBS})


add_definitions(-O2 -pthread)

