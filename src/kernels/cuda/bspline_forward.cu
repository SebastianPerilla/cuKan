// src/kernels/cuda/bspline_forward.cu

#include "bspline_forward.cuh"
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void bspline_forward_kernel(const float* __restrict__ x, const float* __restrict__ coef,
                                       const float* __restrict__ scale_base, const float* __restrict__ scale_sp,
                                       float* __restrict__ out, int B, int in_dim, int out_dim, int G, int k,
                                       int n_coef, float t0, float h, int stride_coef_in, int stride_coef_out) {

    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * out_dim;
    if (pid >= total) {
        return;
    }

    int b_idx = pid / out_dim;
    int j_idx = pid % out_dim;

    float acc = 0.0f;
    float inv_h = 1.0f / h;

    // Cache the row base offset for x to save integer multiplications inside the loop
    int x_row_base = b_idx * in_dim;

    for (int i = 0; i < in_dim; ++i) {
        // Coalesced style evaluation using registers
        float x_val = x[x_row_base + i];

        // O(1) span lookup optimized with hardware intrinsics (no branching)
        float span_f = (x_val - t0) * inv_h;
        int span = (int)floorf(span_f);

        // Native hardware clamping
        span = __float2int_rd(fmaxf((float)k, fminf((float)span, (float)(G + k - 1))));

        float grid_span = t0 + (float)span * h;
        float u = (x_val - grid_span) * inv_h;

        // Optimized evaluation of standard cubic B-Spline blending functions
        // Reduces arithmetic instructions compared to raw matrix-coefficient processing
        float u_inv = 1.0f - u;
        float w0 = 1.0f / 6.0f * (u_inv * u_inv * u_inv);
        float w1 = 1.0f / 6.0f * (3.0f * u * u * u - 6.0f * u * u + 4.0f);
        float w2 = 1.0f / 6.0f * (-3.0f * u * u * u + 3.0f * u * u + 3.0f * u + 1.0f);
        float w3 = 1.0f / 6.0f * (u * u * u);

        int seg_start = span - k;
        int coef_base = i * stride_coef_in + j_idx * stride_coef_out + seg_start;

        // Linear memory access
        float c0 = coef[coef_base + 0];
        float c1 = coef[coef_base + 1];
        float c2 = coef[coef_base + 2];
        float c3 = coef[coef_base + 3];

        float spline_val = w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3;
        float silu_val = silu(x_val);

        int sb_idx = i * out_dim + j_idx;
        float sb = scale_base[sb_idx];
        float sp = scale_sp[sb_idx];

        acc += sb * silu_val + sp * spline_val;
    }

    out[pid] = acc;
}

void bspline_forward_cuda(const float* x, const float* coef, const float* scale_base, const float* scale_sp, float* out,
                          int B, int in_dim, int out_dim, int G, int k, int n_coef, float t0, float h,
                          int stride_coef_in, int stride_coef_out) {
    int total = B * out_dim;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    bspline_forward_kernel<<<gridSize, blockSize>>>(x, coef, scale_base, scale_sp, out, B, in_dim, out_dim, G, k,
                                                    n_coef, t0, h, stride_coef_in, stride_coef_out);
}
