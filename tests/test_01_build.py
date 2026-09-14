import torch


def test_backend_import():
    import kan_cuda_backend

    assert hasattr(kan_cuda_backend, "kan_forward")
    assert hasattr(kan_cuda_backend, "kan_backward")


def test_backend_dummy_shapes():
    import kan_cuda_backend

    N, D_in, D_out, G = 32, 16, 8, 10
    x = torch.randn(N, D_in, device="cuda", dtype=torch.float32)
    coefs = torch.randn(D_in, D_out, G + 3, device="cuda", dtype=torch.float32)
    scale_base = torch.randn(D_in, D_out, device="cuda", dtype=torch.float32)
    scale_sp = torch.randn(D_in, D_out, device="cuda", dtype=torch.float32)

    out = kan_cuda_backend.kan_forward(x, coefs, scale_base, scale_sp, -1.0, 1.0, G)
    assert out.shape == (N, D_out)

    grad_out = torch.randn(N, D_out, device="cuda", dtype=torch.float32)
    dx, dc, dsb, dsp = kan_cuda_backend.kan_backward(
        grad_out, x, coefs, scale_base, scale_sp, -1.0, 1.0, G
    )
    assert dx.shape == (N, D_in)
    assert dc.shape == (D_in, D_out, G + 3)
    assert dsb.shape == (D_in, D_out)
    assert dsp.shape == (D_in, D_out)
