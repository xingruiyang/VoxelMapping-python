cmake_minimum_required(VERSION 3.4...3.18)
project(py_vmapping LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(pybind11)

pybind11_add_module(py_vmapping 
    src/py_vmapping.cpp
    voxelization/src/vmapping.cpp
    voxelization/src/image_proc.cu
    voxelization/src/map_struct.cu
    voxelization/src/ray_tracing.cu
    voxelization/src/depth_fusion.cu
    voxelization/src/marching_cubes.cu)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/voxelization/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/voxelization/include/>)

set_target_properties(${PROJECT_NAME}
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CUDA_ARCHITECTURES 75)

target_compile_definitions(${PROJECT_NAME} 
PRIVATE 
VERSION_INFO=${EXAMPLE_VERSION_INFO})

target_link_libraries(${PROJECT_NAME}
PRIVATE
Eigen3::Eigen
${OpenCV_LIBS})