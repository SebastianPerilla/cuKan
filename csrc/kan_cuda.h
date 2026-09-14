#pragma once

#include <torch/extension.h>

// Full pykan activation: out = scale_base*silu(x) + scale_sp*(U*M*C).
// x: [N, D_in], coefs: [D_in, D_out, G+3], scale_base/scale_sp: [D_in, D_out] -> out: [N, D_out]
torch::Tensor kan_forward(torch::Tensor x, torch::Tensor coefs, torch::Tensor scale_base, torch::Tensor scale_sp,
                          double grid_min, double grid_max, int64_t G);

// Returns {grad_x, grad_coefs, grad_scale_base, grad_scale_sp}.
std::vector<torch::Tensor> kan_backward(torch::Tensor grad_out, torch::Tensor x, torch::Tensor coefs,
                                        torch::Tensor scale_base, torch::Tensor scale_sp, double grid_min,
                                        double grid_max, int64_t G);
