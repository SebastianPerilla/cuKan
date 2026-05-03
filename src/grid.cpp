#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>

Eigen::Tensor<float, 1> build_grid(int G, int k, std::pair<float, float> grid_range) {

    auto [a, b] = grid_range;                  // Structured Bindings Added in C++17
    float h = (b - a) / static_cast<float>(G); // Cast the G into a float (better than implicit type conversion)
    int total_size = G + 2 * k + 1;

    Eigen::Tensor<float, 1> grid(total_size); // Allocate Memory on the Grid to N size

    // Left pad
    for (int i = 0; i < k; ++i) {
        grid(i) = a - (k - i) * h;
    }

    // Interior (linspace equivalent)
    for (int i = 0; i <= G; ++i) {
        grid(k + i) = a + i * h;
    }

    // Right pad
    for (int i = 0; i < k; ++i) {
        grid(k + G + 1 + i) = b + (i + 1) * h;
    }

    return grid;
}
