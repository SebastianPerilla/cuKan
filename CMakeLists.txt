cmake_minimum_required(VERSION 3.23)
project(
  cuKAN
  VERSION 1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# GTEST
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# Eigen
FetchContent_Declare(
  eigen URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.zip)

FetchContent_MakeAvailable(googletest eigen)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
