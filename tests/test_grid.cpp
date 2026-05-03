#include "grid.h"
#include <cassert>
#include <gtest/gtest.h>
#include <unsupported/Eigen/CXX11/Tensor>

// assuming build_grid(...) is declared somewhere

TEST(TestBuildGrid, ShapeDefault) {
    int G = 5;
    int k = 3;
    auto grid = build_grid(G, k, { -1.0f, 1.0f });

    EXPECT_EQ(grid.size(), 12); // 5 + 2*3 + 1 = 12
}
