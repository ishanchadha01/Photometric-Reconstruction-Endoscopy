cmake_minimum_required(VERSION 3.10)
project(photometric_reconstruction)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (OpenMP REQUIRED)
find_package (OpenCV REQUIRED NO_MODULE)
find_package (Ceres REQUIRED PATHS $ENV{CERES_PATH}) # TODO: Add environment variable for ceres path
# target_include_directories(test PRIVATE ${YOUR_DIRECTORY})

add_executable(photometric_reconstruction src/main.cpp)
target_link_libraries(photometric_reconstruction Eigen3::Eigen)
target_link_libraries(photometric_reconstruction ${OpenCV_LIBS})
target_link_libraries(photometric_reconstruction Ceres::ceres)
target_link_libraries(photometric_reconstruction OpenMP::OpenMP_CXX)

set(ENV{CC} "gcc-17")
set(ENV{CXX} "gcc-17")

set(CMAKE_CXX_FLAGS "-DCMAKE_C_COMPILER=/usr/local/bin/gcc-17 -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-17 ${OpenMP_CXX_FLAGS} -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -fpermissive")

set(CMAKE_BUILD_TYPE Debug)