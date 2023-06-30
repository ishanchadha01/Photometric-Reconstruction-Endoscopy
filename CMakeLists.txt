cmake_minimum_required(VERSION 3.10)
project(photometric_reconstruction)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (OpenCV REQUIRED NO_MODULE)
find_package (Ceres REQUIRED PATHS $ENV{CERES_PATH}) # TODO: Add environment variable for ceres path
# target_include_directories(test PRIVATE ${YOUR_DIRECTORY})

add_executable(photometric_reconstruction src/main.cpp)
target_link_libraries(photometric_reconstruction Eigen3::Eigen)
target_link_libraries(photometric_reconstruction ${OpenCV_LIBS})
target_link_libraries(photometric_reconstruction Ceres::ceres)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fpermissive")

set(CMAKE_BUILD_TYPE Debug)