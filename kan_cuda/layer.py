import torch
import torch.nn as nn

import kan_cuda_backend


class KANCUDALayer(nn.Module):
    """Full pykan-formula KAN layer backed by the CUDA kernel.

    out[n, j] = sum_i  scale_base[i,j]*silu(x[n,i]) + scale_sp[i,j]*spline(x[n,i]; coef[i,j,:])
    """

    def __init__(self, in_dim, out_dim, num_intervals=5, grid_min=-1.0, grid_max=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_intervals = num_intervals
        self.grid_min = grid_min
        self.grid_max = grid_max

        n_coef = num_intervals + 3
        # Scale by 1/sqrt(in_dim): the layer sums one term per input dimension
        # (silu and spline branches alike), so unscaled init blows up for wide
        # layers (e.g. in_dim=784). Mirrors pykan's own scale_base init.
        fan_in_scale = 1.0 / (in_dim ** 0.5)
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, n_coef) * 0.1 * fan_in_scale)
        self.scale_base = nn.Parameter(torch.ones(in_dim, out_dim) * fan_in_scale)
        self.scale_sp = nn.Parameter(torch.ones(in_dim, out_dim) * fan_in_scale)

    def forward(self, x):
        x = x.contiguous().float()
        coef = self.coef.contiguous()
        scale_base = self.scale_base.contiguous()
        scale_sp = self.scale_sp.contiguous()
        return kan_cuda_backend.kan_forward(
            x, coef, scale_base, scale_sp, self.grid_min, self.grid_max, self.num_intervals
        )
