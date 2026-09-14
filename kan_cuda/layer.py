import torch
import torch.nn as nn

import kan_cuda_backend


class KANCUDALayer(nn.Module):
    """Localized U*M*C cubic B-spline KAN layer backed by the CUDA kernel."""

    def __init__(self, in_dim, out_dim, num_intervals=5, grid_min=-1.0, grid_max=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_intervals = num_intervals
        self.grid_min = grid_min
        self.grid_max = grid_max

        n_coef = num_intervals + 3
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, n_coef) * 0.5)

    def forward(self, x):
        x = x.contiguous().float()
        coef = self.coef.contiguous()
        return kan_cuda_backend.kan_forward(x, coef, self.grid_min, self.grid_max, self.num_intervals)
