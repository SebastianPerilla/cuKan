find_package(Eigen3 REQUIRED)

add_executable(cuKan)
target_sources(cuKan PRIVATE main.cpp grid.cpp)

target_link_libraries(cuKan PRIVATE Eigen3::Eigen)
