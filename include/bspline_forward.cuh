#pragma once

void bspline_forward_cuda(const float* x, const float* coef, const float* scale_base, const float* scale_sp, float* out,
                          int B, int in_dim, int out_dim, int G, int k, int n_coef, float t0, float h);
