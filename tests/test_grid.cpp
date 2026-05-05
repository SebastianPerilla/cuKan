#include "grid.h"
#include "unsupported/Eigen/CXX11/src/Tensor/Tensor.h"
#include <array>
#include <cassert>
#include <cstdlib>
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

TEST(TestBuildGrid, extension_spacing) {
    const int G { 5 };
    const int k { 3 };

    const float a { -1.0 };
    const float b { 1.0 };

    auto h { (b - a) / G };

    auto grid { build_grid(G, k, std::pair { a, b }) };

    // Left padding: a - k*h, a - (k-1)*h, ..., a - h
    for (size_t i { 0 }; i < k; i++) {
        float expected { a - (k - i) * h };

        std::cout << "\nLeft Padding:\n";
        std::cout << "Expected: " << expected << "\n";
        std::cout << "Grid(i): " << grid(i) << "\n";

        EXPECT_NEAR(grid(i), expected, 1e-6f);
    }

    // Right padding: b + h, b + 2h, ..., b + k*h
    for (size_t i { 0 }; i < k; i++) {
        float expected { b + (i + 1) * h };
        auto idx { k + G + 1 + i };

        std::cout << "\nRight Padding:\n";
        std::cout << "Expected: " << expected << "\n";
        std::cout << "IDX: " << grid(idx) << "\n";
        EXPECT_NEAR(grid(idx), expected, 1e-6f);
    }
}
