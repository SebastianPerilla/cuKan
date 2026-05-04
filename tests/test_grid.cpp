#include "grid.h"
#include "unsupported/Eigen/CXX11/src/Tensor/Tensor.h"
#include <array>
#include <cassert>
#include <gtest/gtest.h>
#include <unsupported/Eigen/CXX11/Tensor>

TEST(TestBuildGrid, shape_default) {
    int G = 5;
    int k = 3;
    auto grid { build_grid(G, k, { -1.0f, 1.0f }) };

    EXPECT_EQ(grid.size(), 12); // 5 + 2*3 + 1 = 12
}

TEST(TestBuildGrid, shape_various) {
    std::array<std::pair<int, int>, 4> shapeVarious { { { 3, 2 }, { 10, 3 }, { 20, 4 }, { 100, 3 } } };

    for (const auto& [G, k] : shapeVarious) {
        auto grid = build_grid(G, k, { -1.0f, -1.0f });

        EXPECT_EQ(grid.dimension(0), G + 2 * k + 1);
    }
}

TEST(TestBuildGrid, interior_knots_uniform) {
    const int G = 5;
    const int k = 3;

    auto grid = build_grid(G, k, { -1.0f, 1.0f });
    const int N = G + 1;
    Eigen::Tensor<float, 1> expected(N);

    for (int i = 0; i < N; ++i) {
        expected(i) = -1.0f + i * (2.0f / G);
    }

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(grid(k + i), expected(i), 1e-6f);
    }
}

// def test_interior_knots_uniform(self):
//     """Interior knots (indices k..k+G) must be linspace(a, b, G+1)."""
//     G, k = 5, 3
//     grid = build_grid(G=G, k=k, grid_range=(-1.0, 1.0))
//     interior = grid[k : k + G + 1]
//     expected = torch.linspace(-1.0, 1.0, G + 1)
//     assert torch.allclose(interior, expected, atol=1e-6)
