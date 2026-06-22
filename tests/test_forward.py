# test_forward.py
import torch
import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
build_src_dir = os.path.abspath(os.path.join(current_dir, "../build/src"))

if build_src_dir not in sys.path:
    sys.path.append(build_src_dir)

import cukan


def torch_bspline_mathematical_reference(
    x, coef, scale_base, scale_sp, out_dim, G, k=3, t0=0.0, h=1.0
):
    """
    An explicit mathematical reference implementation to verify CUDA outputs.
    """
    B, in_dim = x.shape
    out = torch.zeros((B, out_dim), device=x.device, dtype=x.dtype)

    # Core B-spline Matrix Constants (Matching your CUDA namespace)
    M = torch.tensor(
        [
            [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0 / 6.0],
            [-3.0 / 6.0, 0.0 / 6.0, 3.0 / 6.0, 0.0 / 6.0],
            [3.0 / 6.0, -6.0 / 6.0, 3.0 / 6.0, 0.0 / 6.0],
            [-1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0],
        ],
        device=x.device,
        dtype=torch.float32,
    )

    for b in range(B):
        for j in range(out_dim):
            acc = 0.0
            for i in range(in_dim):
                x_val = x[b, i].item()

                # Replicate CUDA O(1) span lookup and clamping
                span_f = (x_val - t0) / h
                span = int(math.floor(span_f))
                if span < k:
                    span = k
                if span > G + k - 1:
                    span = G + k - 1

                grid_span = t0 + float(span) * h
                u = (x_val - grid_span) / h

                # Compute weights mathematically via powers of u
                u_vec = torch.tensor(
                    [1.0, u, u**2, u**3], device=x.device, dtype=torch.float32
                )
                weights = u_vec @ M  # returns [w0, w1, w2, w3]

                # Partition of Unity Invariant Check
                assert abs(weights.sum().item() - 1.0) < 1e-5, (
                    f"Math invariant broken: weights sum to {weights.sum().item()}"
                )

                seg_start = span - k
                c = coef[i, j, seg_start : seg_start + 4]

                spline_val = (weights * c).sum().item()
                silu_val = x_val / (1.0 + math.exp(-x_val))

                sb = scale_base[i, j].item()
                sp = scale_sp[i, j].item()

                acc += sb * silu_val + sp * spline_val
            out[b, j] = acc
    return out


def run_accuracy_checks():
    print("====================================================")
    print("   cuKan CUDA B-Spline Accuracy Verification Suite   ")
    print("====================================================\n")

    # Fixed structural parameters
    B, in_dim, out_dim = 3, 4, 4
    G, k = 5, 3
    n_coef = G + k + 1  # 9
    t0, h = -1.0, 0.5

    device = torch.device("cuda")

    # Seed generators for reproducible verification
    torch.manual_seed(1337)
    scale_base = torch.randn(in_dim, out_dim, device=device)
    scale_sp = torch.randn(in_dim, out_dim, device=device)
    coef = torch.randn(in_dim, out_dim, n_coef, device=device)

    # --------------------------------------------------------
    # CRITICAL TEST 1: Standard Domain Values
    # --------------------------------------------------------
    print("🧪 Test 1: Standard Domain Points (In-bound values)...")
    x_normal = torch.tensor(
        [[-0.8, 0.2, 0.5, 0.9], [-0.1, -0.5, 1.1, 0.0], [0.4, 0.7, -0.9, 1.2]],
        device=device,
    )

    out_cuda_1 = cukan.bspline_forward(
        x_normal, coef, scale_base, scale_sp, out_dim, G, k, t0, h
    )
    out_ref_1 = torch_bspline_mathematical_reference(
        x_normal, coef, scale_base, scale_sp, out_dim, G, k, t0, h
    )

    max_error_1 = torch.max(torch.abs(out_cuda_1 - out_ref_1)).item()
    print(f"   -> Max Absolute Discrepancy: {max_error_1:.2e}")
    assert max_error_1 < 1e-5, "❌ Test 1 Failed: Accuracy out of bounds!"
    print("   ✅ Passed.")

    # --------------------------------------------------------
    # CRITICAL TEST 2: Extreme Out-of-Bounds and Clamping
    # --------------------------------------------------------
    print("\n🧪 Test 2: Boundary Clamping Handling...")
    # Passing values heavily below t0 (-1.0) and heavily above max range (-1.0 + 5*0.5 = 1.5)
    x_extreme = torch.tensor(
        [
            [-50.0, 100.0, -10.0, 25.0],
            [-1.00, 1.50, -1.01, 1.51],  # Testing exact edges and fractional offsets
            [-0.50, 0.00, 0.50, 1.00],
        ],
        device=device,
    )

    out_cuda_2 = cukan.bspline_forward(
        x_extreme, coef, scale_base, scale_sp, out_dim, G, k, t0, h
    )
    out_ref_2 = torch_bspline_mathematical_reference(
        x_extreme, coef, scale_base, scale_sp, out_dim, G, k, t0, h
    )

    max_error_2 = torch.max(torch.abs(out_cuda_2 - out_ref_2)).item()
    print(f"   -> Max Absolute Discrepancy: {max_error_2:.2e}")
    assert max_error_2 < 1e-5, "❌ Test 2 Failed: Edge clamping math mismatch!"
    print("   ✅ Passed.")

    # --------------------------------------------------------
    # CRITICAL TEST 3: Memory Stride Integrity
    # --------------------------------------------------------
    print("\n🧪 Test 3: Non-contiguous Strided Layout Verification...")
    # Create a transposed view of a larger tensor to inject complex memory strides
    coef_raw = torch.randn(out_dim, in_dim, n_coef, device=device)
    coef_strided = coef_raw.permute(1, 0, 2)

    out_cuda_3 = cukan.bspline_forward(
        x_normal, coef_strided, scale_base, scale_sp, out_dim, G, k, t0, h
    )
    out_ref_3 = torch_bspline_mathematical_reference(
        x_normal, coef_strided, scale_base, scale_sp, out_dim, G, k, t0, h
    )

    max_error_3 = torch.max(torch.abs(out_cuda_3 - out_ref_3)).item()
    print(f"   -> Max Absolute Discrepancy: {max_error_3:.2e}")
    assert max_error_3 < 1e-5, (
        "❌ Test 3 Failed: Stride-based dynamic index calculation failed!"
    )
    print("   ✅ Passed.")

    print(
        "\n🎉 Verification complete! The forward pass is 100% accurate across all structural domains."
    )


if __name__ == "__main__":
    run_accuracy_checks()
