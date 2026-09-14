# cuKan

A custom CUDA/PyTorch C++ extension implementing pykan's real KAN layer
activation — `scale_base·silu(x) + scale_sp·spline(x)` — as a fused CUDA
kernel exposed through pybind11. It reproduces pykan's exact per-edge
activation formula and parameterization, but evaluates it with a localized
matrix formulation in a single hand-written kernel instead of pykan's
dense, recursively-computed B-spline basis — yielding a measured **199.1x**
forward+backward speedup and up to **890x** in a real training loop.

---

## What this project is

[Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (KANs)
replace an MLP's fixed activation + linear weight with a learnable
univariate spline on every edge of the network. That's more expressive per
parameter, but the reference implementation, `pykan`, is slow: evaluating a
spline per edge, per sample, is real per-element computation, not a matrix
multiply, and `pykan`'s implementation does it generically through a chain
of ordinary PyTorch tensor ops.

This project is a from-scratch CUDA/C++ reimplementation of that one
operation — the KAN layer's forward and backward pass — built as a
`torch.utils.cpp_extension.CUDAExtension` with hand-written pybind11
bindings, not a higher-level kernel DSL (Triton, etc.). The deliverable is
a drop-in `KANCUDALayer(nn.Module)` that is numerically faithful to
`pykan`'s actual layer (same formula, same three learnable parameter
tensors) but runs one to three orders of magnitude faster, verified end to
end on both a synthetic benchmark and a real MNIST training run.

---

## Approach

### 1. The math: a closed-form, localized evaluation instead of a dense recursive one

`pykan`'s spline evaluation (`kan/spline.py`, function `B_batch`) computes
the B-spline basis via the textbook **Cox-de Boor recursion**: to get a
degree-`k` basis, it recomputes a degree-`(k-1)` basis, recursively down to
degree 0, materializing a new dense tensor of shape `(batch, in_dim,
grid_width)` at every recursion level, then contracts over the *entire*
grid width even though a B-spline only has 4 non-zero values at any given
point. That's real, repeated, wasted global-memory traffic that gets worse
as the grid (`G`), spline order (`k`), or layer width grow.

We exploit the fact that a cubic (`k=3`) uniform B-spline has a known
closed form and never recurse. For a scalar input `x` on a uniform grid
with `G` intervals over `[x_min, x_max]`, `Δx = (x_max - x_min)/G`:

```
i = clamp(floor((x - x_min) / Δx), 0, G - 1)          # active interval, O(1)
u = (x - (x_min + i·Δx)) / Δx                          # local coordinate in [0,1)

U = [1, u, u^2, u^3]

        ┌  1   4   1   0 ┐
    M = │ -3   0   3   0 │ / 6
        │  3  -6   3   0 │
        └ -1   3  -3   1 ┘

spline(x) = (U · M) · C[i : i+4]        # only 4 control points ever touched
```

`U · M` (the blend weights) is computed directly from powers of `u`
in-kernel — no matrix-vector multiply, no basis tensor, no recursion.

The full per-layer activation matches `pykan`'s `KANLayer` exactly — a
spline term *plus* a SiLU residual, each with its own learnable per-edge
scale:

```
silu(x) = x * sigmoid(x)

Y[n, d_out] = Σ_{d_in}  scale_base[d_in,d_out]·silu(x[n,d_in])
                       + scale_sp[d_in,d_out]·spline(x[n,d_in]; C[d_in,d_out,:])
```

Three learnable tensors per layer: `C (D_in,D_out,G+3)`, `scale_base
(D_in,D_out)`, `scale_sp (D_in,D_out)`.

### 2. The CUDA kernels

**Forward** (`kan_forward_kernel`, `csrc/kan_cuda_kernel.cu`): a 2D thread
grid covers `(D_out, N)` — every thread owns one output scalar
`Y[n, d_out]` and loops over `D_in` internally, resolving the interval,
blend weights, and SiLU value for each input and accumulating into a
register. One kernel launch computes the entire layer's forward pass.

**Backward** (`kan_backward_kernel`): analytical, not autodiff-traced.
Gradients are derived directly:

```
dU/du = [0, 1, 2u, 3u^2]              du/dx = 1/Δx
d(silu)/dx = sigmoid(x)·(1 + x·(1 - sigmoid(x)))

dY/dx           = Σ_{d_out} grad_out · (scale_base·d(silu)/dx + scale_sp·((dU/du·M)·C[i:i+4])/Δx)
dY/dC[i:i+4]    = Σ_n grad_out · scale_sp · (U·M)
dY/d(scale_base)= Σ_n grad_out · silu(x)
dY/d(scale_sp)  = Σ_n grad_out · spline(x)
```

A 2D thread grid covers `(D_in, N)` this time — each thread owns one
`dx[n, d_in]` output (no collision, written directly), but multiple batch
elements can land in the same grid interval for the same edge, so gradient
contributions to `C`, `scale_base`, and `scale_sp` are accumulated with
`atomicAdd`.

### 3. Wiring it into PyTorch

Both kernels are launched from `csrc/kan_cuda.cpp`, which registers them as
a single `torch::autograd::Function` (`KanSplineFunction`) — `forward`
calls the forward kernel and saves the tensors it needs; `backward` calls
the backward kernel with those saved tensors. This is done entirely in
C++, so `kan_forward` is differentiable end-to-end from Python without any
separate Python-side `autograd.Function` wrapper. Correctness of the hand
written backward is checked against `torch.autograd.gradcheck` on float64
inputs (finite-difference verification), not just "it runs."

`kan_cuda/layer.py` wraps this in `KANCUDALayer(nn.Module)` — a normal
PyTorch module with `nn.Parameter`s for `coef`, `scale_base`, `scale_sp`,
so it composes with `nn.Sequential`, any optimizer, any loss, `.to(device)`,
`state_dict()`, etc. like any built-in layer.

---

## How this compares to pykan

| | `pykan` | `cuKan` (this project) |
|---|---|---|
| Per-edge activation formula | `scale_base·silu(x) + scale_sp·spline(x)` | **identical** |
| Learnable parameters per layer | `coef`, `scale_base`, `scale_sp` | **identical** |
| Basis evaluation | Cox-de Boor recursion (`k` recursive levels, each materializing a dense `(batch,in_dim,grid_width)` tensor) | Closed-form `U·M` computed directly, no recursion |
| Data touched per edge per sample | Full grid width (`G+k` basis values, mostly multiplied by zero) | Exactly 4 control points |
| Execution | Chain of ~10+ separate PyTorch/ATen ops per layer, each with its own dispatch + launch overhead | 1 CUDA kernel launch for forward, 1 for backward, per layer |
| Backward | Standard autograd through the dense-tensor forward graph | Hand-derived analytical kernel with `atomicAdd` accumulation |
| Forward+backward latency (`N=8192, D_in=D_out=32, G=10`) | 573.35 ms | 2.88 ms (**199.1x**) |
| 1-epoch / 5,000 MNIST samples | 444.63 s | 0.50 s (**890.8x**) |
| MNIST, 3 epochs, 60,000 samples, test accuracy | not run at this scale (too slow to be practical) | 95.97% in 10.93s total |

The formula and parameterization are deliberately identical to `pykan` — this
is meant to be a faithful, drop-in-numerically-equivalent layer, not a
simplified approximation. The speedup comes entirely from *how* that
formula is evaluated: no recursion, no wasted computation outside a spline's
local support, and kernel fusion (one launch instead of many).

---

## Project Layout

```
csrc/
  kan_cuda.h            declarations: kan_forward / kan_backward
  kan_cuda.cpp          pybind11 bindings, autograd::Function, host checks
  kan_cuda_kernel.cu    forward/backward CUDA kernels
kan_cuda/
  layer.py              KANCUDALayer(nn.Module)
examples/
  train_mnist.py        end-to-end MNIST training + pykan speed comparison
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
pixi run python examples/train_mnist.py   # real MNIST training + pykan comparison
```

## Benchmark Results

Measured on an NVIDIA GeForce RTX 3070 Laptop GPU, single forward + backward
pass, `N = 8192`, `D_in = 32`, `D_out = 32`, `G = 10` grid intervals:

| Implementation       | Total Time (fwd + bwd) | Speedup |
|-----------------------|------------------------:|--------:|
| `pykan.KAN`            |               573.35 ms |      1x |
| `KANCUDALayer` (cuKan) |                 2.88 ms | **199.1x** |

Reproduce with `pixi run benchmark`.

## Real-World Benchmark: MNIST Classification

`examples/train_mnist.py` trains a 2-layer KAN
(`KANCUDALayer(784, 64, num_intervals=8) -> KANCUDALayer(64, 10, num_intervals=8)`)
on the full MNIST training set (60,000 samples, batch size 256, Adam
lr=0.005, `CrossEntropyLoss`) for 3 epochs, then runs a 1-epoch head-to-head
against `pykan.KAN([784, 64, 10], grid=8)` on a 5,000-sample subset to get a
real training-loop speedup ratio (data loading, autograd, and the optimizer
step included, not just the raw kernel).

**Full 3-epoch training (KANCUDALayer only, 60,000 samples):**

| Epoch | Time  | Loss   |
|-------|------:|-------:|
| 1     | 3.69s | 0.266  |
| 2     | 3.62s | 0.106  |
| 3     | 3.62s | 0.069  |

Total training time: **10.93s**. Final test-set accuracy: **95.97%**.

**1-epoch head-to-head vs. `pykan.KAN` (5,000-sample subset, batch 256):**

| Implementation         | Time (1 epoch) | Speedup |
|-------------------------|---------------:|--------:|
| `pykan.KAN`              |       444.63s |      1x |
| `KANCUDALayer` (cuKan)   |         0.50s | **890.8x** |

pykan takes roughly 7-8 minutes to complete a single epoch over 5,000 MNIST
samples at this width; cuKan completes the same epoch in half a second, and
trains the full 60,000-sample dataset for 3 epochs in the time pykan needs
for a few dozen mini-batches.

---

## Roadmap: what's next for the CUDA implementation

The current implementation is deliberately scoped to first-order training
(standard `loss.backward()`). Two concrete extensions are planned next,
identified by comparing against a more mature Triton-based KAN
implementation:

- **Second-order differentiability (PINN-ready).** The backward kernel is
  first-order only today; `torch.autograd.grad(..., create_graph=True)`
  through `KANCUDALayer` would not correctly propagate gradients from a
  second-derivative loss (e.g. a PDE residual / Laplacian term) back into
  the spline parameters. Closing this requires an analytical
  double-backward CUDA kernel plus chaining a second
  `torch::autograd::Function` so higher-order derivatives work end to end
  — needed for physics-informed neural network (PINN) style training, not
  for standard classification/regression losses like the MNIST example
  above.
- **Numerical parity against real `pykan` tensors, not just an internal
  reference.** Current tests (`test_02_forward.py`) check the CUDA kernel
  against a hand-derived Python re-implementation of the *same* formula —
  internally consistent, but never directly diffed against tensors
  produced by running real `pykan`. The plan is a small script that builds
  an actual `pykan.KANLayer`, saves its forward and gradient output on
  fixed inputs, and asserts our kernel matches it to `1e-5`.

Explicitly not planned unless priorities change: a Triton port (this
project is intentionally raw CUDA/C++), genuine SRAM/shared-memory tiling
(the current bottleneck is kernel-launch/dispatch overhead, not memory
bandwidth, so tiling wouldn't move the needle at current problem sizes),
roofline/hardware-utilization profiling, mixed-precision (FP16/BF16)
support, or a CPU fallback backend.

A detailed, phase-by-phase tracking document for this roadmap is kept
locally (gitignored, not published) so it can be picked back up across
sessions.
