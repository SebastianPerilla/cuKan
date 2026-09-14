import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kan import KAN as PyKAN
from kan_cuda.layer import KANCUDALayer


def run_benchmark():
    N, D_in, D_out, G = 8192, 32, 32, 10
    print(
        f"\n--- Running Benchmark (Batch Size N={N}, D_in={D_in}, D_out={D_out}, Grid G={G}) ---"
    )

    X = torch.empty(N, D_in, device="cuda").uniform_(-0.8, 0.8)

    # 1. Official PyKAN Baseline
    pykan_model = PyKAN(width=[D_in, D_out], grid=G, k=3).to("cuda")

    torch.cuda.synchronize()
    start_pykan = time.time()
    out_pykan = pykan_model(X)
    loss_pykan = out_pykan.sum()
    loss_pykan.backward()
    torch.cuda.synchronize()
    pykan_time = (time.time() - start_pykan) * 1000.0

    # 2. Custom CUDA KAN Layer
    cuda_kan = KANCUDALayer(D_in, D_out, num_intervals=G).to("cuda")

    # Warmup
    for _ in range(5):
        cuda_kan(X).sum().backward()

    torch.cuda.synchronize()
    start_cuda = time.time()
    out_cuda = cuda_kan(X)
    loss_cuda = out_cuda.sum()
    loss_cuda.backward()
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_cuda) * 1000.0

    speedup = pykan_time / cuda_time
    print(f"Official PyKAN Total Time: {pykan_time:.2f} ms")
    print(f"Custom CUDA KAN Total Time: {cuda_time:.2f} ms")
    print(f"--> Achieved Speedup: {speedup:.1f}x")

    assert speedup >= 10.0, (
        f"Speedup requirement failed: achieved {speedup:.1f}x, target >= 10x"
    )


if __name__ == "__main__":
    run_benchmark()
