import torch
import torch.nn as nn
from kan_cuda.layer import KANCUDALayer


def test_toy_dataset_convergence():
    torch.manual_seed(42)
    X = torch.empty(512, 2, device="cuda").uniform_(-0.8, 0.8)
    Y = torch.sin(3.14159 * X[:, 0:1]) + (X[:, 1:2] ** 2)

    model = nn.Sequential(
        KANCUDALayer(2, 16, num_intervals=8), KANCUDALayer(16, 1, num_intervals=8)
    ).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    initial_loss = criterion(model(X), Y).item()
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.15, (
        f"Loss failed to drop sufficiently: Initial {initial_loss}, Final {final_loss}"
    )
