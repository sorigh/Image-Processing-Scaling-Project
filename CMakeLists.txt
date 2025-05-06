cmake_minimum_required(VERSION 3.14)
project(ScalingProject)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ScalingProject
        ../ScalingProject/main.cpp
        src/interpolation.cpp
        src/interpolation.h
)

target_link_libraries(ScalingProject ${OpenCV_LIBS})
