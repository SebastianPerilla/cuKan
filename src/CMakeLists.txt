add_executable(kan_app main.cpp kernels/cuda/bspline_forward.cu)

target_include_directories(kan_app PRIVATE ${PROJECT_SOURCE_DIR}/include)

set_target_properties(kan_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
