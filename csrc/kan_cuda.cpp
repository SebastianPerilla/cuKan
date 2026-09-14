#include "kan_cuda.h"

torch::Tensor kan_forward(torch::Tensor x, torch::Tensor coefs, double grid_min, double grid_max, int64_t G) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(coefs.is_cuda() && coefs.is_contiguous(), "coefs must be a contiguous CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must have shape [N, D_in]");
    TORCH_CHECK(coefs.dim() == 3, "coefs must have shape [D_in, D_out, G+3]");
    TORCH_CHECK(coefs.size(0) == x.size(1), "coefs.size(0) must equal D_in");
    TORCH_CHECK(coefs.size(2) == G + 3, "coefs.size(2) must equal G+3");

    const auto N = x.size(0);
    const auto D_out = coefs.size(1);

    // Kernel launch wired in during forward-pass implementation.
    return torch::zeros({N, D_out}, x.options());
}

std::vector<torch::Tensor> kan_backward(torch::Tensor grad_out, torch::Tensor x, torch::Tensor coefs, double grid_min,
                                        double grid_max, int64_t G) {
    TORCH_CHECK(grad_out.is_cuda() && grad_out.is_contiguous(), "grad_out must be a contiguous CUDA tensor");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(coefs.is_cuda() && coefs.is_contiguous(), "coefs must be a contiguous CUDA tensor");

    // Kernel launch wired in during backward-pass implementation.
    return {torch::zeros_like(x), torch::zeros_like(coefs)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kan_forward", &kan_forward, "Localized U*M*C cubic B-spline forward");
    m.def("kan_backward", &kan_backward, "Localized U*M*C cubic B-spline backward");
}
