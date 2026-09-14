from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="kan_cuda_backend",
    ext_modules=[
        CUDAExtension(
            name="kan_cuda_backend",
            sources=["csrc/kan_cuda.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
