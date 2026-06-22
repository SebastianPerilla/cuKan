# src/CMakeLists.txt

add_library(cukan MODULE bindings.cpp kernels/cuda/bspline_forward.cu)

target_include_directories(cukan PRIVATE ${PROJECT_SOURCE_DIR}/include)

# ==============================================================================
# THE NUCLEAR OPTION: Purge PyTorch's ABI flags
# ==============================================================================
# 1. Scrub global variables
string(REGEX REPLACE "-D_GLIBCXX_USE_CXX11_ABI=[01]" "" CMAKE_CXX_FLAGS
                     "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-D_GLIBCXX_USE_CXX11_ABI=[01]" "" CMAKE_CUDA_FLAGS
                     "${CMAKE_CUDA_FLAGS}")

# 1. Scrub PyTorch's imported CMake targets
foreach(tgt torch torch_cpu torch_cuda)
  if(TARGET ${tgt})
    # Clean Definitions
    get_target_property(defs ${tgt} INTERFACE_COMPILE_DEFINITIONS)
    if(defs AND NOT defs STREQUAL "defs-NOTFOUND")
      list(REMOVE_ITEM defs "_GLIBCXX_USE_CXX11_ABI=1"
           "_GLIBCXX_USE_CXX11_ABI=0")
      set_target_properties(${tgt} PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                                              "${defs}")
    endif()

    # Clean Options
    get_target_property(opts ${tgt} INTERFACE_COMPILE_OPTIONS)
    if(opts AND NOT opts STREQUAL "opts-NOTFOUND")
      list(REMOVE_ITEM opts "-D_GLIBCXX_USE_CXX11_ABI=1"
           "-D_GLIBCXX_USE_CXX11_ABI=0")
      set_target_properties(${tgt} PROPERTIES INTERFACE_COMPILE_OPTIONS
                                              "${opts}")
    endif()
  endif()
endforeach()

# 1. Now safely enforce the correct Conda ABI (0) without any interference
target_compile_definitions(cukan PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
# ==============================================================================

target_link_libraries(cukan PRIVATE ${TORCH_LIBRARIES} torch_python
                                    Python3::Module pybind11::module)

set_target_properties(
  cukan
  PROPERTIES PREFIX ""
             OUTPUT_NAME "cukan"
             SUFFIX ".so"
             CXX_STANDARD 17
             CUDA_STANDARD 17
             CUDA_SEPARABLE_COMPILATION ON)
