// src/kernels/cuda/bspline_forward.cu

#include "bspline_forward.cuh"
#include <cuda_runtime.h>

namespace Constants {
    // cubic basis constants
    constexpr float M00 = 1.0f / 6.0f;
    constexpr float M01 = 4.0f / 6.0f;
    constexpr float M02 = 1.0f / 6.0f;
    constexpr float M03 = 0.0f;
    constexpr float M10 = -3.0f / 6.0f;
    constexpr float M11 = 0.0f;
    constexpr float M12 = 3.0f / 6.0f;
    constexpr float M13 = 0.0f;
    constexpr float M20 = 3.0f / 6.0f;
    constexpr float M21 = -6.0f / 6.0f;
    constexpr float M22 = 3.0f / 6.0f;
    constexpr float M23 = 0.0f;
    constexpr float M30 = -1.0f / 6.0f;
    constexpr float M31 = 3.0f / 6.0f;
    constexpr float M32 = -3.0f / 6.0f;
    constexpr float M33 = 1.0f / 6.0f;
} // namespace Constants

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void bspline_forward_kernel(const float* x, const float* coef, const float* scale_base,
                                       const float* scale_sp, float* out, int B, int in_dim, int out_dim, int G, int k,
                                       int n_coef, float t0, float h) {

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * out_dim;
    if (pid >= total) {
        return;
    }

    int b_idx = pid / out_dim;
    int j_idx = pid % out_dim;

    float acc = 0.0f;

    for (int i = 0; i < in_dim; ++i) {
        float x_val = x[b_idx * in_dim + i];

        float span_f = (x_val - t0) / h;
        int span = (int)floorf(span_f);
        span = max(span, k);
        span = min(span, G + k - 1);

        float grid_span = t0 + (float)span * h;
        float u = (x_val - grid_span) / h;

        float u2 = u * u;
        float u3 = u2 * u;

        float w0 = Constants::M00 + u * Constants::M10 + u2 * Constants::M20 + u3 * Constants::M30;
        float w1 = Constants::M01 + u * Constants::M11 + u2 * Constants::M21 + u3 * Constants::M31;
        float w2 = Constants::M02 + u * Constants::M12 + u2 * Constants::M22 + u3 * Constants::M32;
        float w3 = Constants::M03 + u * Constants::M13 + u2 * Constants::M23 + u3 * Constants::M33;

        int seg_start = span - k;
        int base = i * out_dim * n_coef + j_idx * n_coef + seg_start;

        float c0 = coef[base + 0];
        float c1 = coef[base + 1];
        float c2 = coef[base + 2];
        float c3 = coef[base + 3];

        float spline_val = w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3;

        float sb = scale_base[i * out_dim + j_idx];
        float sp = scale_sp[i * out_dim + j_idx];

        acc += sb * silu(x_val) + sp * spline_val;
    }

    out[b_idx * out_dim + j_idx] = acc;
}

void bspline_forward_cuda(const float* x, const float* coef, const float* scale_base, const float* scale_sp, float* out,
                          int B, int in_dim, int out_dim, int G, int k, int n_coef, float t0, float h) {
    int total = B * out_dim;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    bspline_forward_kernel<<<gridSize, blockSize>>>(x, coef, scale_base, scale_sp, out, B, in_dim, out_dim, G, k,
                                                    n_coef, t0, h);
}
