cmake_minimum_required(VERSION 2.8)
project(DiffGT2PreForPose)

FIND_PACKAGE(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 11)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(SOURCE_FILES main.cpp jsoncpp.cpp)
include_directories(json)
add_executable(DiffGT2PreForPose ${SOURCE_FILES})
add_executable(DiffGT2PreForPose_output main_output.cpp jsoncpp.cpp)
#set(OPENCV_LIBS  opencv_
target_link_libraries(DiffGT2PreForPose  ${OpenCV_LIBS})
target_link_libraries(DiffGT2PreForPose_output ${OpenCV_LIBS})