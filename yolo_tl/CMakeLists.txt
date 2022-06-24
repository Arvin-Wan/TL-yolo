cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TL-yolo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(OpenCV_DIR /root/autodl-tmp/TL/opencv/build) 
set(CMAKE_PREFIX_PATH /usr/include/python3.8)
set(CMAKE_PREFIX_PATH /usr/include/python3.8;/root/autodl-tmp/miniconda3/envs/ssrnet2/lib;/root/autodl-tmp/miniconda3/envs/ssrnet2/lib/python3.9/site-packages/numpy/core/include;/root/autodl-tmp/TL/libtorch;/root/autodl-tmp/TL/vision)
find_package(Python3 COMPONENTS Development Python3::python)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
add_executable(yolo_tl detect_yolo.cpp)

target_link_libraries(yolo_tl ${TORCH_LIBRARIES})
target_link_libraries(yolo_tl ${TorchVision_LIBRARIES} TorchVision::TorchVision)
target_link_libraries(yolo_tl ${OpenCV_LIBS})
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
set_property(TARGET yolo_tl PROPERTY CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/include)