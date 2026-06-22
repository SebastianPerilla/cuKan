cmake_minimum_required(VERSION 3.23)

project(cukan LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Adjust to your GPU if needed. RTX 30xx -> 86 RTX 40xx -> 89 RTX 50xx -> 120
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Torch: ${TORCH_LIBRARIES}")
message(STATUS "Torch CXX flags: ${TORCH_CXX_FLAGS}")

add_subdirectory(src)

enable_testing()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
