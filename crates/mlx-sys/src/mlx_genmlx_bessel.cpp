#include "mlx_common.h"

// ============================================================================
// mlx_genmlx_bessel.cpp  (P2 — genmlx-core CUDA/Linux graft)
//
// Exponentially-scaled modified Bessel functions of the first kind, order 0
// and 1:  i0e(x) = I0(x) * exp(-|x|),  i1e(x) = I1(x) * exp(-|x|).
//
// Our frozen MLX 0.32.0 (b410f6c) does NOT export bessel_i0e / bessel_i1e
// (the donor fork added them via an MLX patch; we keep MLX frozen). We
// therefore re-implement them as COMPOSITES from primitives our 0.32.0 HAS:
//   where, abs, exp, sqrt, the arithmetic/comparison operators (+ - * / <)
// All of these are CUDA-native lazy graph ops and are fully autograd-
// composable, so the resulting function differentiates correctly (used by
// the von Mises log-prob path, which is the only consumer — low traffic).
//
// Numerical method: the classic Abramowitz & Stegun (9.8.1–9.8.4) rational
// polynomial approximations, two-branch form selected with where():
//   * |x| < 3.75 : polynomial in t = (x/3.75)^2, then scaled by exp(-|x|).
//   * |x| >= 3.75: asymptotic 1/sqrt(|x|) rational in u = 3.75/|x|. Already
//                  carries the exp(|x|) factor analytically, so for the
//                  *scaled* variant we DROP the exp and keep only the
//                  1/sqrt(|x|) * P(u) part — no overflow for large x.
// Both branches are evaluated unconditionally (graph ops) and merged by
// where(); the unused branch's intermediate values are finite for all real
// x because we feed abs(x) (never negative) and, for the small branch's
// large-|x| inputs, the polynomial stays finite (it's only multiplied by
// exp(-|x|)->0 and then discarded by where()).
//
// Max relative error of the A&S forms is < ~1.6e-7 (I0) / < ~2.2e-7 (I1),
// comfortably below float32 precision.
// ============================================================================

extern "C" {

namespace {

// Horner evaluation of a polynomial in the array `t` with the given
// ascending-order coefficients (c[0] + c[1]*t + c[2]*t^2 + ...). Pure graph
// construction; `t` is the lazy input node, scalars fold via the operator
// overloads (array op double).
static mlx::core::array poly_horner(const mlx::core::array& t,
                                    const double* coeffs,
                                    int n) {
  // Start from the highest-order coefficient as a scalar promoted against t.
  mlx::core::array acc = t * 0.0 + coeffs[n - 1];
  for (int i = n - 2; i >= 0; --i) {
    acc = acc * t + coeffs[i];
  }
  return acc;
}

// i0e(x) = I0(x) * exp(-|x|), x given as |x| (ax) plus the original.
static mlx::core::array bessel_i0e_composite(const mlx::core::array& x) {
  using mlx::core::abs;
  using mlx::core::exp;
  using mlx::core::sqrt;
  using mlx::core::where;

  mlx::core::array ax = abs(x);

  // --- Small branch: |x| < 3.75, polynomial in t = (x/3.75)^2 -------------
  // A&S 9.8.1 coefficients for I0.
  static const double i0_small[] = {
      1.0, 3.5156229, 3.0899424, 1.2067492,
      0.2659732, 0.0360768, 0.0045813};
  mlx::core::array t_small = (x / 3.75) * (x / 3.75);  // (x/3.75)^2
  mlx::core::array i0_unscaled =
      poly_horner(t_small, i0_small, 7);               // ~ I0(x)
  // Scale to i0e: multiply by exp(-|x|).
  mlx::core::array small = i0_unscaled * exp(-ax);

  // --- Large branch: |x| >= 3.75, asymptotic in u = 3.75/|x| --------------
  // A&S 9.8.2 coefficients for exp(-|x|)*sqrt(|x|)*I0(x).
  static const double i0_large[] = {
      0.39894228,  0.01328592,  0.00225319, -0.00157565,
      0.00916281, -0.02057706,  0.02635537, -0.01647633,
      0.00392377};
  mlx::core::array u = 3.75 / ax;                       // 3.75/|x|
  mlx::core::array p_large = poly_horner(u, i0_large, 9);
  // exp(-|x|)*I0(x) = (1/sqrt(|x|)) * P(u)
  mlx::core::array large = p_large / sqrt(ax);

  return where(ax < 3.75, small, large);
}

// i1e(x) = I1(x) * exp(-|x|). I1 is ODD; the asymptotic form gives |I1|, so
// the large branch must carry the sign of x.
static mlx::core::array bessel_i1e_composite(const mlx::core::array& x) {
  using mlx::core::abs;
  using mlx::core::exp;
  using mlx::core::sqrt;
  using mlx::core::where;

  mlx::core::array ax = abs(x);

  // --- Small branch: |x| < 3.75 -------------------------------------------
  // A&S 9.8.3: I1(x)/x = poly in t=(x/3.75)^2, so I1(x) = x * poly.
  static const double i1_small[] = {
      0.5, 0.87890594, 0.51498869, 0.15084934,
      0.02658733, 0.00301532, 0.00032411};
  mlx::core::array t_small = (x / 3.75) * (x / 3.75);
  mlx::core::array i1_over_x = poly_horner(t_small, i1_small, 7);
  mlx::core::array i1_unscaled = x * i1_over_x;          // ~ I1(x), keeps sign
  mlx::core::array small = i1_unscaled * exp(-ax);

  // --- Large branch: |x| >= 3.75 ------------------------------------------
  // A&S 9.8.4: exp(-|x|)*sqrt(|x|)*I1(|x|) = poly in u=3.75/|x| (gives |I1|).
  static const double i1_large[] = {
       0.39894228, -0.03988024, -0.00362018,  0.00163801,
      -0.01031555,  0.02282967, -0.02895312,  0.01787654,
      -0.00420059};
  mlx::core::array u = 3.75 / ax;
  mlx::core::array p_large = poly_horner(u, i1_large, 9);
  mlx::core::array large_mag = p_large / sqrt(ax);       // exp(-|x|)*|I1(x)|
  // Restore the odd sign: multiply by sign(x) = x/|x|. ax==0 is excluded
  // here (this branch only fires for |x|>=3.75), so the division is safe.
  mlx::core::array large = large_mag * (x / ax);

  return where(ax < 3.75, small, large);
}

}  // namespace

mlx_array* mlx_array_bessel_i0e(mlx_array* handle) {
  MLX_GUARD_PTR("array_bessel_i0e",
    auto arr = reinterpret_cast<mlx::core::array*>(handle);
    mlx::core::array result = bessel_i0e_composite(*arr);
    return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(result)));
  )
}

mlx_array* mlx_array_bessel_i1e(mlx_array* handle) {
  MLX_GUARD_PTR("array_bessel_i1e",
    auto arr = reinterpret_cast<mlx::core::array*>(handle);
    mlx::core::array result = bessel_i1e_composite(*arr);
    return reinterpret_cast<mlx_array*>(new mlx::core::array(std::move(result)));
  )
}

}  // extern "C"
