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

__global__ void kan_forward_kernel(const float* __restrict__ x, const float* __restrict__ coefs,
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
        acc += w[0] * c[0] + w[1] * c[1] + w[2] * c[2] + w[3] * c[3];
    }

    out[n * D_out + d_out] = acc;
}

__global__ void kan_backward_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                    const float* __restrict__ coefs, float* __restrict__ grad_x,
                                    float* __restrict__ grad_coefs, int64_t N, int64_t D_in, int64_t D_out,
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
    float grad_x_acc = 0.0f;

    for (int64_t d_out = 0; d_out < D_out; ++d_out) {
        float go = grad_out[n * D_out + d_out];
        const float* c = coefs + (d_in * D_out + d_out) * n_coef + i;
        float* gc = grad_coefs + (d_in * D_out + d_out) * n_coef + i;

        float du_acc = 0.0f;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            du_acc += dw[m] * c[m];
            atomicAdd(&gc[m], go * w[m]);
        }
        grad_x_acc += go * du_acc * inv_h;
    }

    grad_x[n * D_in + d_in] = grad_x_acc;
}

}  // namespace

void kan_forward_cuda(const float* x, const float* coefs, float* out, int64_t N, int64_t D_in, int64_t D_out,
                      int64_t G, float grid_min, float grid_len, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((D_out + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    kan_forward_kernel<<<grid, block, 0, stream>>>(x, coefs, out, N, D_in, D_out, G, grid_min, grid_len);
}

void kan_backward_cuda(const float* grad_out, const float* x, const float* coefs, float* grad_x, float* grad_coefs,
                       int64_t N, int64_t D_in, int64_t D_out, int64_t G, float grid_min, float grid_len,
                       cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((D_in + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    kan_backward_kernel<<<grid, block, 0, stream>>>(grad_out, x, coefs, grad_x, grad_coefs, N, D_in, D_out, G,
                                                    grid_min, grid_len);
}
