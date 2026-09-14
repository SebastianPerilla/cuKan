#include "kan_cuda.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

void kan_forward_cuda(const float* x, const float* coefs, float* out, int64_t N, int64_t D_in, int64_t D_out,
                      int64_t G, float grid_min, float grid_len, cudaStream_t stream);

void kan_backward_cuda(const float* grad_out, const float* x, const float* coefs, float* grad_x, float* grad_coefs,
                       int64_t N, int64_t D_in, int64_t D_out, int64_t G, float grid_min, float grid_len,
                       cudaStream_t stream);

torch::Tensor kan_forward(torch::Tensor x, torch::Tensor coefs, double grid_min, double grid_max, int64_t G) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(coefs.is_cuda() && coefs.is_contiguous(), "coefs must be a contiguous CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(coefs.dtype() == torch::kFloat32, "coefs must be float32");
    TORCH_CHECK(x.dim() == 2, "x must have shape [N, D_in]");
    TORCH_CHECK(coefs.dim() == 3, "coefs must have shape [D_in, D_out, G+3]");
    TORCH_CHECK(coefs.size(0) == x.size(1), "coefs.size(0) must equal D_in");
    TORCH_CHECK(coefs.size(2) == G + 3, "coefs.size(2) must equal G+3");

    const auto N = x.size(0);
    const auto D_in = x.size(1);
    const auto D_out = coefs.size(1);
    const float grid_len = static_cast<float>(grid_max - grid_min) / static_cast<float>(G);

    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty({N, D_out}, x.options());

    kan_forward_cuda(x.data_ptr<float>(), coefs.data_ptr<float>(), out.data_ptr<float>(), N, D_in, D_out, G,
                     static_cast<float>(grid_min), grid_len, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

std::vector<torch::Tensor> kan_backward(torch::Tensor grad_out, torch::Tensor x, torch::Tensor coefs, double grid_min,
                                        double grid_max, int64_t G) {
    TORCH_CHECK(grad_out.is_cuda() && grad_out.is_contiguous(), "grad_out must be a contiguous CUDA tensor");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
    TORCH_CHECK(coefs.is_cuda() && coefs.is_contiguous(), "coefs must be a contiguous CUDA tensor");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "grad_out must be float32");

    const auto N = x.size(0);
    const auto D_in = x.size(1);
    const auto D_out = coefs.size(1);
    const float grid_len = static_cast<float>(grid_max - grid_min) / static_cast<float>(G);

    c10::cuda::CUDAGuard device_guard(x.device());
    auto grad_x = torch::empty_like(x);
    auto grad_coefs = torch::zeros_like(coefs);

    kan_backward_cuda(grad_out.data_ptr<float>(), x.data_ptr<float>(), coefs.data_ptr<float>(),
                      grad_x.data_ptr<float>(), grad_coefs.data_ptr<float>(), N, D_in, D_out, G,
                      static_cast<float>(grid_min), grid_len, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_x, grad_coefs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kan_forward", &kan_forward, "Localized U*M*C cubic B-spline forward");
    m.def("kan_backward", &kan_backward, "Localized U*M*C cubic B-spline backward");
}
