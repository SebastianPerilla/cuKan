import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kan_cuda.layer import KANCUDALayer

DEVICE = "cuda"
DATA_ROOT = "./data"


def load_mnist():
    # Flatten to 784 and rescale to [-1, 1] to match the default spline grid domain.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1) * 2.0 - 1.0)]
    )
    train_set = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    return train_set, test_set


def build_cuda_kan():
    return nn.Sequential(
        KANCUDALayer(784, 64, num_intervals=8),
        KANCUDALayer(64, 10, num_intervals=8),
    ).to(DEVICE)


def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def train_cuda_kan(train_set, test_set, epochs=3, batch_size=256, lr=0.005):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    model = build_cuda_kan()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_times = []
    total_start = time.time()
    for epoch in range(epochs):
        torch.cuda.synchronize()
        epoch_start = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"[cuKan] Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s - loss {loss.item():.4f}")
    total_time = time.time() - total_start

    accuracy = evaluate_accuracy(model, test_loader)
    return epoch_times, total_time, accuracy


def run_pykan_speed_comparison(train_set, subset_size=5000, batch_size=256, lr=0.005):
    from kan import KAN as PyKAN

    subset = Subset(train_set, range(subset_size))
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    pykan_model = PyKAN(width=[784, 64, 10], grid=8, k=3).to(DEVICE)
    pykan_optimizer = torch.optim.Adam(pykan_model.parameters(), lr=lr)

    torch.cuda.synchronize()
    start = time.time()
    for x, y in subset_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pykan_optimizer.zero_grad()
        loss = criterion(pykan_model(x), y)
        loss.backward()
        pykan_optimizer.step()
    torch.cuda.synchronize()
    pykan_time = time.time() - start

    cuda_model = build_cuda_kan()
    cuda_optimizer = torch.optim.Adam(cuda_model.parameters(), lr=lr)

    torch.cuda.synchronize()
    start = time.time()
    for x, y in subset_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        cuda_optimizer.zero_grad()
        loss = criterion(cuda_model(x), y)
        loss.backward()
        cuda_optimizer.step()
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    return pykan_time, cuda_time


def main():
    torch.manual_seed(42)
    train_set, test_set = load_mnist()

    print("=== cuKan Full Training (3 epochs, 60000 samples) ===")
    epoch_times, total_time, accuracy = train_cuda_kan(train_set, test_set)
    print(f"Per-epoch times: {[f'{t:.2f}s' for t in epoch_times]}")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Final test accuracy: {accuracy * 100:.2f}%")

    print("\n=== 1-Epoch Speed Comparison vs pykan (5000-sample subset) ===")
    pykan_time, cuda_time = run_pykan_speed_comparison(train_set)
    speedup = pykan_time / cuda_time
    print(f"pykan.KAN:    {pykan_time:.2f}s")
    print(f"KANCUDALayer: {cuda_time:.2f}s")
    print(f"Speedup:      {speedup:.1f}x")


if __name__ == "__main__":
    main()
