#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Uniform cubic B-spline blend weights: [w0..w3] = U * M, U = [1, u, u^2, u^3].
__device__ __forceinline__ void blend_weights(float u, float w[4]) {
    float u2 = u * u;
    float u3 = u2 * u;
    w[0] = (1.0f - 3.0f * u + 3.0f * u2 - u3) / 6.0f;
    w[1] = (4.0f - 6.0f * u2 + 3.0f * u3) / 6.0f;
    w[2] = (1.0f + 3.0f * u + 3.0f * u2 - 3.0f * u3) / 6.0f;
    w[3] = u3 / 6.0f;
}

// d(blend_weights)/du.
__device__ __forceinline__ void blend_weights_grad(float u, float dw[4]) {
    float u2 = u * u;
    dw[0] = (-3.0f + 6.0f * u - 3.0f * u2) / 6.0f;
    dw[1] = (-12.0f * u + 9.0f * u2) / 6.0f;
    dw[2] = (3.0f + 6.0f * u - 9.0f * u2) / 6.0f;
    dw[3] = (3.0f * u2) / 6.0f;
}

__device__ __forceinline__ int locate_span(float x_val, float grid_min, float grid_len, int64_t G) {
    int i = static_cast<int>(floorf((x_val - grid_min) / grid_len));
    return max(0, min(static_cast<int>(G) - 1, i));
}

// pykan's base-residual activation: silu(x) = x * sigmoid(x).
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// d(silu)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
__device__ __forceinline__ float silu_grad(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

__global__ void kan_forward_kernel(const float* __restrict__ x, const float* __restrict__ coefs,
                                   const float* __restrict__ scale_base, const float* __restrict__ scale_sp,
                                   float* __restrict__ out, int64_t N, int64_t D_in, int64_t D_out, int64_t G,
                                   float grid_min, float grid_len) {
    int64_t d_out = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || d_out >= D_out) {
        return;
    }

    const int64_t n_coef = G + 3;
    float acc = 0.0f;

    for (int64_t d_in = 0; d_in < D_in; ++d_in) {
        float x_val = x[n * D_in + d_in];
        int i = locate_span(x_val, grid_min, grid_len, G);
        float u = (x_val - (grid_min + i * grid_len)) / grid_len;

        float w[4];
        blend_weights(u, w);

        const float* c = coefs + (d_in * D_out + d_out) * n_coef + i;
        float spline_val = w[0] * c[0] + w[1] * c[1] + w[2] * c[2] + w[3] * c[3];
        float silu_val = silu(x_val);

        int64_t s_idx = d_in * D_out + d_out;
        acc += scale_base[s_idx] * silu_val + scale_sp[s_idx] * spline_val;
    }

    out[n * D_out + d_out] = acc;
}

__global__ void kan_backward_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                    const float* __restrict__ coefs, const float* __restrict__ scale_base,
                                    const float* __restrict__ scale_sp, float* __restrict__ grad_x,
                                    float* __restrict__ grad_coefs, float* __restrict__ grad_scale_base,
                                    float* __restrict__ grad_scale_sp, int64_t N, int64_t D_in, int64_t D_out,
                                    int64_t G, float grid_min, float grid_len) {
    int64_t d_in = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || d_in >= D_in) {
        return;
    }

    const int64_t n_coef = G + 3;
    float x_val = x[n * D_in + d_in];
    int i = locate_span(x_val, grid_min, grid_len, G);
    float u = (x_val - (grid_min + i * grid_len)) / grid_len;

    float w[4];
    float dw[4];
    blend_weights(u, w);
    blend_weights_grad(u, dw);

    float inv_h = 1.0f / grid_len;
    float silu_val = silu(x_val);
    float dsilu_val = silu_grad(x_val);
    float grad_x_acc = 0.0f;

    for (int64_t d_out = 0; d_out < D_out; ++d_out) {
        float go = grad_out[n * D_out + d_out];
        int64_t s_idx = d_in * D_out + d_out;
        float sb = scale_base[s_idx];
        float sp = scale_sp[s_idx];

        const float* c = coefs + (d_in * D_out + d_out) * n_coef + i;
        float* gc = grad_coefs + (d_in * D_out + d_out) * n_coef + i;

        float spline_val = w[0] * c[0] + w[1] * c[1] + w[2] * c[2] + w[3] * c[3];
        atomicAdd(&grad_scale_base[s_idx], go * silu_val);
        atomicAdd(&grad_scale_sp[s_idx], go * spline_val);

        float go_sp = go * sp;
        float du_acc = 0.0f;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            du_acc += dw[m] * c[m];
            atomicAdd(&gc[m], go_sp * w[m]);
        }
        grad_x_acc += go * (sb * dsilu_val + sp * du_acc * inv_h);
    }

    grad_x[n * D_in + d_in] = grad_x_acc;
}

}  // namespace

void kan_forward_cuda(const float* x, const float* coefs, const float* scale_base, const float* scale_sp, float* out,
                      int64_t N, int64_t D_in, int64_t D_out, int64_t G, float grid_min, float grid_len,
                      cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((D_out + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    kan_forward_kernel<<<grid, block, 0, stream>>>(x, coefs, scale_base, scale_sp, out, N, D_in, D_out, G, grid_min,
                                                   grid_len);
}

void kan_backward_cuda(const float* grad_out, const float* x, const float* coefs, const float* scale_base,
                       const float* scale_sp, float* grad_x, float* grad_coefs, float* grad_scale_base,
                       float* grad_scale_sp, int64_t N, int64_t D_in, int64_t D_out, int64_t G, float grid_min,
                       float grid_len, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((D_in + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    kan_backward_kernel<<<grid, block, 0, stream>>>(grad_out, x, coefs, scale_base, scale_sp, grad_x, grad_coefs,
                                                    grad_scale_base, grad_scale_sp, N, D_in, D_out, G, grid_min,
                                                    grid_len);
}
