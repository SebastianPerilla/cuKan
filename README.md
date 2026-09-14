# cuKan

A custom CUDA/PyTorch C++ extension implementing the spline evaluation core of
Kolmogorov-Arnold Networks (KAN). It replaces the per-edge B-spline evaluation
in `pykan` with a fused CUDA kernel driven by a localized matrix formulation,
yielding a measured **291.9x** forward+backward speedup over the reference
`pykan` implementation.

## Overview

A KAN layer replaces the fixed activation + linear weight of an MLP with a
learnable univariate spline on every edge of the computation graph. For a
layer mapping `D_in` inputs to `D_out` outputs, each `(d_in, d_out)` edge
carries an independent cubic B-spline over a uniform grid, and the layer
output is the sum of these spline evaluations across the input dimension.

This project reimplements that evaluation as a single dense CUDA kernel: for
every scalar input, only the four B-spline control points local to that
input's grid interval are ever touched, and no dense basis matrix is
materialized. The kernel is threaded so that CUDA occupancy scales with batch
size and output width, with the input dimension folded into a per-thread
accumulation loop.

## Mathematical Formulation

For a scalar input `x` on a uniform grid with `G` intervals spanning
`[x_min, x_max]`, let `Δx = (x_max - x_min) / G`. The active interval index
and local coordinate are:

```
i = clamp(floor((x - x_min) / Δx), 0, G - 1)
u = (x - (x_min + i·Δx)) / Δx
```

The four B-spline control points relevant to `x` are `C[i : i+4]`. The cubic
value is computed as a localized `U · M · C` product:

```
U = [1, u, u^2, u^3]

        ┌  1   4   1   0 ┐
    M = │ -3   0   3   0 │ / 6
        │  3  -6   3   0 │
        └ -1   3  -3   1 ┘

spline(x) = (U · M) · C[i : i+4]
```

`U · M` is the closed-form uniform cubic B-spline basis — computed directly
in-kernel from powers of `u` rather than an explicit matrix-vector multiply.
The full layer output sums this over the input dimension:

```
Y[n, d_out] = Σ_{d_in} spline(x[n, d_in]; C[d_in, d_out, :])
```

**Forward kernel.** A 2D thread grid covers `(D_out, N)`; each thread loops
over `D_in`, resolving the local interval and blend weights once per input
and accumulating into a single output scalar.

**Backward kernel.** Gradients are analytical, not autodiff-traced through
the blend weights:

```
dU/du = [0, 1, 2u, 3u^2]        du/dx = 1 / Δx

dY/dx[n, d_in]      = Σ_{d_out} grad_out[n, d_out] · ((dU/du · M) · C[i:i+4]) / Δx
dY/dC[d_in, d_out, i:i+4] = Σ_n grad_out[n, d_out] · (U · M)
```

Since multiple batch elements can land in the same grid interval for a given
edge, gradient contributions to `C` are accumulated with `atomicAdd`. The
`x`-gradient has no such collision (one thread owns each `(n, d_in)` output)
and is written directly.

Both kernels are exposed through a `torch::autograd::Function`
(`KanSplineFunction`) registered in `csrc/kan_cuda.cpp`, so `kan_forward` is
differentiable end-to-end without any Python-side autograd wrapper —
verified against `torch.autograd.gradcheck` on float64 inputs.

## Project Layout

```
csrc/
  kan_cuda.h            declarations: kan_forward / kan_backward
  kan_cuda.cpp          pybind11 bindings, autograd::Function, host checks
  kan_cuda_kernel.cu    forward/backward CUDA kernels
kan_cuda/
  layer.py              KANCUDALayer(nn.Module)
tests/
  test_01_build.py      extension import & shape checks
  test_02_forward.py    forward parity vs. a pure-Python reference
  test_03_backward.py   gradcheck on the analytical backward kernel
  test_04_module.py     toy-function convergence via nn.Module
  test_05_benchmark.py  latency comparison against pykan.KAN
```

## Installation

Environment and build are managed entirely through [Pixi](https://pixi.sh).

```bash
pixi install          # resolve and create the conda/pip environment
pixi run build         # compile the CUDA extension (setup.py build_ext --inplace)
```

Requires an NVIDIA GPU with a CUDA 12.1-compatible driver. The build targets
`sm_86` by default (Ampere); adjust `TORCH_CUDA_ARCH_LIST` for other
architectures.

## Running

```bash
pixi run test          # pytest tests/  (build, forward, backward, module)
pixi run benchmark     # tests/test_05_benchmark.py  (latency vs. pykan.KAN)
```

## Benchmark Results

Measured on an NVIDIA GeForce RTX 3070 Laptop GPU, single forward + backward
pass, `N = 8192`, `D_in = 32`, `D_out = 32`, `G = 10` grid intervals:

| Implementation       | Total Time (fwd + bwd) | Speedup |
|-----------------------|------------------------:|--------:|
| `pykan.KAN`            |               605.23 ms |      1x |
| `KANCUDALayer` (cuKan) |                 2.07 ms | **291.9x** |

This exceeds the project's 10x-30x target range. The gap is largely
architectural: `pykan` evaluates the full dense B-spline basis per edge
through a sequence of PyTorch ops, while `kan_forward` fuses interval lookup,
blend-weight computation, and the local dot product into one kernel launch
per layer, touching only the four control points that matter for each input.

Reproduce with:

```bash
pixi run benchmark
```
