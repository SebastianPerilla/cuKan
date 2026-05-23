add_executable(kan_app main.cpp kernels/cuda/bspline_forward.cu)

target_include_directories(kan_app PRIVATE ${PROJECT_SOURCE_DIR}/include)

set_target_properties(kan_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(Torch REQUIRED)

target_link_libraries(main "${TORCH_LIBRARIES}")

set_property(TARGET main PROPERTY CXX_STANDARD 17)
