#include <Eigen/Dense>
#include <cmath>
#include <unistd.h>

// Same cubic basis matrix constants as the forward kernel.

namespace g_BsplineBackwards_Constants {
    constexpr long double M00 { 1.0 / 6.0 };
    constexpr long double M01 { 4.0 / 6.0 };
    constexpr long double M02 { 1.0 / 6.0 };
    constexpr long double M03 { 0.0 };

    constexpr long double M10 { -3.0 / 6.0 };
    constexpr long double M11 { 0.0 };
    constexpr long double M12 { 3.0 / 6.0 };
    constexpr long double M13 { 0.0 };

    constexpr long double M20 { 3.0 / 6.0 };
    constexpr long double M21 { -6.0 / 6.0 };
    constexpr long double M22 { 3.0 / 6.0 };
    constexpr long double M23 { 0.0 };

    constexpr long double M30 { -1.0 / 6.0 };
    constexpr long double M31 { 3.0 / 6.0 };
    constexpr long double M32 { -3.0 / 6.0 };
    constexpr long double M33 { 1.0 / 6.0 };

} // namespace g_BsplineBackwards_Constants

namespace g_BsplineBackwards_Parameters {}

int bspline_backward_kernel(int x_ptr, int coef_ptr, double scale_base_ptr, double scale_sp_ptr, double dout_ptr,
                            double dx_ptr, double d_coef_ptr, double dscale_base_ptr, double dscale_sp_ptr, int t0,
                            int h, double in_dim, double out_dim, int G, int k, int n_coef) {

    /* Backwards Pass:
     * Compute Gradients for X, coef, scale_base, scale_sp
     */

    pid_t pid { getpid() };
    auto b_idx = std::floor(pid / in_dim);
    auto i_idx = pid % in_dim;

    auto x_val = 0;

    return 0;
}
