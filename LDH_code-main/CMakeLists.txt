cmake_minimum_required(VERSION 3.10)
project(DynamicHomography)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

find_library(LIBFFI_LIBRARIES NAMES ffi HINTS /lib/x86_64-linux-gnu)

add_executable(Optimization RedRiverIndividualCameraAdjustment.cpp main.cpp image_processing_utils.cpp)

target_include_directories(Optimization PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(Optimization PRIVATE ${OpenCV_LIBS} ${CERES_LIBRARIES} ${LIBFFI_LIBRARIES})
