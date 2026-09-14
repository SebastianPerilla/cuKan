import torch


def python_reference_kan_forward(x, coefs, scale_base, scale_sp, grid_min, grid_max, G):
    N, D_in = x.shape
    _, D_out, _ = coefs.shape
    grid_len = (grid_max - grid_min) / G
    M = (
        torch.tensor(
            [[1, 4, 1, 0], [-3, 0, 3, 0], [3, -6, 3, 0], [-1, 3, -3, 1]],
            dtype=torch.float32,
            device=x.device,
        )
        / 6.0
    )

    out = torch.zeros((N, D_out), device=x.device, dtype=torch.float32)
    for n in range(N):
        for dout in range(D_out):
            val_sum = 0.0
            for din in range(D_in):
                val = x[n, din].item()
                i = int(max(0, min(G - 1, (val - grid_min) // grid_len)))
                u = (val - (grid_min + i * grid_len)) / grid_len
                U = torch.tensor([1.0, u, u**2, u**3], device=x.device)
                c_local = coefs[din, dout, i : i + 4]
                blend = torch.matmul(U, M)
                spline_val = torch.dot(blend, c_local).item()
                silu_val = val / (1.0 + torch.exp(torch.tensor(-val)).item())
                val_sum += (
                    scale_base[din, dout].item() * silu_val
                    + scale_sp[din, dout].item() * spline_val
                )
            out[n, dout] = val_sum
    return out


def test_forward_cuda_parity():
    import kan_cuda_backend

    N, D_in, D_out, G = 16, 8, 4, 5
    torch.manual_seed(42)
    x = torch.empty(N, D_in, device="cuda").uniform_(-0.8, 0.8)
    coefs = torch.randn(D_in, D_out, G + 3, device="cuda", dtype=torch.float32)
    scale_base = torch.randn(D_in, D_out, device="cuda", dtype=torch.float32)
    scale_sp = torch.randn(D_in, D_out, device="cuda", dtype=torch.float32)

    py_out = python_reference_kan_forward(x, coefs, scale_base, scale_sp, -1.0, 1.0, G)
    cuda_out = kan_cuda_backend.kan_forward(x, coefs, scale_base, scale_sp, -1.0, 1.0, G)

    assert torch.allclose(py_out, cuda_out, atol=1e-4, rtol=1e-4)
