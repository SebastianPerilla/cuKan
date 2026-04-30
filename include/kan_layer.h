#pragma once

#include "Eigen/Core"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/Tensor.h>

using Eigen::Tensor;

template <typename T, typename V> class _KAN_FUNCTION {
    private:
    public:
        _KAN_FUNCTION() {};

        ~_KAN_FUNCTION();

        _KAN_FUNCTION& operator=(const _KAN_FUNCTION& other) {};

        Eigen::MatrixXd forward(T ctx, Tensor<float, 2> x, Tensor<float, 2> coef, Tensor<float, 2> scale_base,
                                Tensor<float, 2> scale_sp, float t0, float h, V in_dim, V out_dim, V G, V k, V n_coef) {
            int B = x.dimension(0);
            Eigen::Tensor<float, 2> out(B, out_dim);
        };
};
