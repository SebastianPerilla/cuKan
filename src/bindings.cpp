// src/bindings.cpp

#include "bspline_forward.cuh"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

torch::Tensor bspline_forward(torch::Tensor x, torch::Tensor coef, torch::Tensor scale_base, torch::Tensor scale_sp,
                              std::int64_t out_dim, std::int64_t G, std::int64_t k, double t0, double h) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(coef.is_cuda(), "coef must be a CUDA tensor");
    TORCH_CHECK(scale_base.is_cuda(), "scale_base must be a CUDA tensor");
    TORCH_CHECK(scale_sp.is_cuda(), "scale_sp must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(coef.dtype() == torch::kFloat32, "coef must be float32");
    TORCH_CHECK(scale_base.dtype() == torch::kFloat32, "scale_base must be float32");
    TORCH_CHECK(scale_sp.dtype() == torch::kFloat32, "scale_sp must be float32");

    TORCH_CHECK(x.dim() == 2, "x must have shape [B, in_dim]");
    TORCH_CHECK(coef.dim() == 3, "coef must have shape [in_dim, out_dim, n_coef]");
    TORCH_CHECK(scale_base.dim() == 2, "scale_base must have shape [in_dim, out_dim]");
    TORCH_CHECK(scale_sp.dim() == 2, "scale_sp must have shape [in_dim, out_dim]");
    TORCH_CHECK(k == 3, "Current kernel only supports cubic splines (k == 3)");

    const auto B = x.size(0);
    const auto in_dim = x.size(1);
    const auto n_coef = G + k + 1;

    TORCH_CHECK(coef.size(0) == in_dim, "coef.size(0) must equal in_dim");
    TORCH_CHECK(coef.size(1) == out_dim, "coef.size(1) must equal out_dim");
    TORCH_CHECK(coef.size(2) == n_coef, "coef.size(2) must equal G + k + 1");
    TORCH_CHECK(scale_base.size(0) == in_dim && scale_base.size(1) == out_dim,
                "scale_base must have shape [in_dim, out_dim]");
    TORCH_CHECK(scale_sp.size(0) == in_dim && scale_sp.size(1) == out_dim,
                "scale_sp must have shape [in_dim, out_dim]");

    x = x.contiguous();
    // Do NOT call coef = coef.contiguous() here! We want to preserve its original memory layout to use its strides.
    scale_base = scale_base.contiguous();
    scale_sp = scale_sp.contiguous();

    // Extract the actual strides from the PyTorch tensor
    int stride_coef_in = coef.stride(0);
    int stride_coef_out = coef.stride(1);

    c10::cuda::CUDAGuard device_guard(x.device());

    auto out = torch::empty({ B, out_dim }, x.options());

    // Pass the extracted strides to the CUDA launcher
    bspline_forward_cuda(x.data_ptr<float>(), coef.data_ptr<float>(), scale_base.data_ptr<float>(),
                         scale_sp.data_ptr<float>(), out.data_ptr<float>(), static_cast<int>(B),
                         static_cast<int>(in_dim), static_cast<int>(out_dim), static_cast<int>(G), static_cast<int>(k),
                         static_cast<int>(n_coef), static_cast<float>(t0), static_cast<float>(h), stride_coef_in,
                         stride_coef_out); // Added here

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(cukan, m) {
    m.def("bspline_forward", &bspline_forward, py::arg("x"), py::arg("coef"), py::arg("scale_base"),
          py::arg("scale_sp"), py::arg("out_dim"), py::arg("G"), py::arg("k") = 3, py::arg("t0") = 0.0,
          py::arg("h") = 1.0);
}
