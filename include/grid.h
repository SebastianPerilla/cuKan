#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>

Eigen::Tensor<float, 1> build_grid(int G, int k, std::pair<float, float> grid_range);
