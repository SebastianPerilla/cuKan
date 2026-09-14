#pragma once

#include <torch/extension.h>

// Localized U*M*C cubic B-spline evaluation.
// x: [N, D_in], coefs: [D_in, D_out, G+3] -> out: [N, D_out]
torch::Tensor kan_forward(torch::Tensor x, torch::Tensor coefs, double grid_min, double grid_max, int64_t G);

// Returns {grad_x, grad_coefs}.
std::vector<torch::Tensor> kan_backward(torch::Tensor grad_out, torch::Tensor x, torch::Tensor coefs, double grid_min,
                                        double grid_max, int64_t G);
