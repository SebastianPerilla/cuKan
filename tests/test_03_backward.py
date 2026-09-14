import torch


def test_cuda_gradcheck():
    import kan_cuda_backend

    N, D_in, D_out, G = 4, 3, 2, 4
    x = (
        torch.empty(N, D_in, device="cuda", dtype=torch.float64)
        .uniform_(-0.7, 0.7)
        .requires_grad_(True)
    )
    coefs = torch.randn(
        D_in, D_out, G + 3, device="cuda", dtype=torch.float64
    ).requires_grad_(True)
    scale_base = torch.randn(
        D_in, D_out, device="cuda", dtype=torch.float64
    ).requires_grad_(True)
    scale_sp = torch.randn(
        D_in, D_out, device="cuda", dtype=torch.float64
    ).requires_grad_(True)

    def func(x_in, c_in, sb_in, sp_in):
        return kan_cuda_backend.kan_forward(
            x_in.float(), c_in.float(), sb_in.float(), sp_in.float(), -1.0, 1.0, G
        ).double()

    assert torch.autograd.gradcheck(
        func, (x, coefs, scale_base, scale_sp), eps=1e-3, atol=1e-2
    )
