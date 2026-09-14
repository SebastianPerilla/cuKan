# KAN CUDA Project Plan & State Tracking

## Project Objective
Implement a custom C++/CUDA PyTorch extension for Kolmogorov-Arnold Networks (KAN) utilizing the localized matrix formulation ($U \cdot M \cdot C$) for B-spline evaluation. Achieve a verified 10x-30x speedup over the standard PyTorch `pykan` implementation on synthetic financial/mathematical target functions.

## Architectural Guidelines
1. **Spline Evaluation Formula**:
   $$u = \frac{x - (x_{\text{min}} + i \cdot \Delta x)}{\Delta x}, \quad U = [1, u, u^2, u^3]$$
   $$Y[n, d_{\text{out}}] = \sum_{d_{\text{in}}} \left( U \cdot M \right) \cdot C[d_{\text{in}}, d_{\text{out}}, i : i+4]$$
2. **CUDA Threading**: Block size (16, 16) over (Batch N, Output Dim $D_{\text{out}}$). Internal loop over Input Dim $D_{\text{in}}$.
3. **Backward Gradient Accumulation**: Analytical derivative for input $x$; `atomicAdd` for control point tensor $C$.
4. **Environment**: Executed via `pixi run`.

---

## Sprint Task Checklist

### Phase 0: Environment & Verification
- [x] Initialize Pixi environment (`pixi install`).
- [x] Confirm `torch.cuda.is_available()` is True inside `pixi run`.

### Phase 1: Extension Scaffolding & Build System
- [x] Create `setup.py` using `torch.utils.cpp_extension.CUDAExtension`.
- [x] Create `csrc/kan_cuda.h` declaring `kan_forward` and `kan_backward`.
- [x] Create `csrc/kan_cuda.cpp` binding functions via `pybind11` under module `kan_cuda_backend`.
- [x] Create `tests/test_01_build.py` to verify C++ module import and zero-tensor output shapes.
- [x] Run `pixi run pytest tests/test_01_build.py` and pass.

### Phase 2: CUDA Forward Kernel ($U \cdot M \cdot C$)
- [x] Write `kan_forward_kernel` in `csrc/kan_cuda_kernel.cu`.
- [x] Implement local interval lookup, clamped power vector $U$, cubic blend matrix $M$, and coefficient dot product.
- [x] Hook host launch routine into `csrc/kan_cuda.cpp`.
- [x] Create `tests/test_02_forward.py` comparing output against pure Python `pykan` baseline.
- [x] Run `pixi run pytest tests/test_02_forward.py` and pass (`atol=1e-4`).

### Phase 3: CUDA Backward Kernel & Analytical Gradients
- [x] Derive $dU/du = [0, 1, 2u, 3u^2]$ and $du/dx = 1 / \Delta x$.
- [x] Write `kan_backward_kernel` in `csrc/kan_cuda_kernel.cu`.
- [x] Apply `atomicAdd` to avoid gradient updates race conditions on control tensor $dC$.
- [x] Create `tests/test_03_backward.py` running `torch.autograd.gradcheck` on float64 input tensors.
- [x] Run `pixi run pytest tests/test_03_backward.py` and pass.

### Phase 4: `torch.nn.Module` Integration
- [x] Create `kan_cuda/layer.py` wrapping custom `torch.autograd.Function` into `KANCUDALayer(nn.Module)`.
- [x] Initialize parameters ($C$) and default grid boundaries.
- [x] Create `tests/test_04_module.py` fitting target function $y = \sin(\pi x_0) + x_1^2$.
- [x] Run `pixi run pytest tests/test_04_module.py` and confirm >80% loss reduction in 100 epochs.

### Phase 5: Verification & `pykan` Speedup Benchmark
- [x] Create `tests/test_05_benchmark.py` comparing runtime latency and memory of `KANCUDALayer` vs `pykan.KAN`.
- [x] Verify 10x to 30x forward/backward pass execution speedup on batch size $N=8192$.
- [x] Run `pixi run python tests/test_05_benchmark.py` and generate report.

**Result:** 291.9x speedup measured (PyKAN: 605.23 ms, KANCUDALayer: 2.07 ms) for N=8192, D_in=32, D_out=32, G=10 forward+backward pass. Exceeds the 10x-30x target range.
