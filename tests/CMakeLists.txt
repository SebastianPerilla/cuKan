# tests/CMakeLists.txt

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(test_grid test_grid.cpp ${CMAKE_SOURCE_DIR}/src/grid.cpp)

target_link_libraries(test_grid PRIVATE GTest::gtest_main Eigen3::Eigen)
target_include_directories(test_grid PRIVATE ${CMAKE_SOURCE_DIR}/include
                                             ${CMAKE_SOURCE_DIR}/src)

gtest_discover_tests(test_grid)
