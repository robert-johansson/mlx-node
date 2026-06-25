// mlx_genmlx_gamma.cpp — composite lgamma / digamma for the GenMLX CUDA port.
//
// WHY THIS FILE EXISTS
// --------------------
// The robert-johansson/mlx-node fork PATCHED its (older, 0.31.2-based) MLX to
// add `mlx::core::lgamma` and `mlx::core::digamma` as native primitives, and
// its mlx-sys shims simply called them. Our build target keeps MLX FROZEN at
// 0.32.0 @ b410f6c, which has NEITHER op (verified: 0 grep hits). Rather than
// patch a frozen MLX, we re-implement both as COMPOSITES built purely from
// primitives our 0.32.0 ops.h already exports (log, sin, cos, where, square,
// abs, astype, full, plus the scalar +/-/* operator overloads).
//
// Consequences of the composite approach, all of them desirable here:
//   * CUDA-native: every sub-op (log/sin/cos/where/...) has a CUDA kernel in
//     0.32.0, so the whole composite runs on the Jetson Blackwell GPU with no
//     host fallback.
//   * Autograd-composable: every sub-op is a differentiable MLX primitive with a
//     registered VJP, so reverse-mode autodiff threads through the composite
//     automatically. No custom VJP / mlx_custom_function is required. This matters
//     because lgamma/digamma sit on the gradient-critical path of the beta, gamma,
//     and (log-factorial of the) Poisson log-probabilities.
//
// The branch-free idiom: MLX has no host control flow inside a graph, so each
// shim evaluates BOTH the "primary" branch and the "reflected" branch over the
// whole array and selects per-element with `where`. The reflection argument is
// constructed so the primary path never sees a value where it would diverge.
//
// GRADIENT-SEAM ROBUSTNESS (the "double where" trick)
// ---------------------------------------------------
// MLX's Select (where) VJP routes the cotangent into branch x as
// `multiply(cond_as_float, g)` and into branch y as `multiply(!cond, g)` — it
// does NOT prune the unselected subgraph. So even for an element where `where`
// discards the reflection branch, autograd still flows a ZERO cotangent BACK
// THROUGH that branch's nodes. The reflection branch contains
// `log(|sin(pi*x)|)`, whose local derivative `pi*cos(pi*x)/sin(pi*x)` blows up
// to +/-inf at integer x. `0 * inf = NaN`, which then ADDS into the total
// gradient w.r.t. x and poisons it — at exactly the integer arguments that
// lgamma(n+1) (binomial/Poisson log-norm) and integer-valued beta/gamma shape
// params legitimately take.
//
// Fix: feed the reflection-ONLY transcendentals (`sin`/`cos`/`log|sin|`) a
// SAFE argument `x_safe = where(is_reflect, x, k)` with a benign constant k
// (0.25) on the non-reflect elements. For non-reflect elements the dead branch
// then evaluates sin(pi*0.25) != 0, so its local derivative is finite and the
// 0-cotangent yields a clean 0 instead of NaN. The final `where(is_reflect,...)`
// still selects the primary branch for those elements, so VALUES are unchanged;
// for reflect elements x_safe == x, so reflect values/grads are unchanged too.
// This is the standard JAX-style guard for where-based piecewise functions.
//
// Symbols brought in via mlx_common.h's anonymous namespace (using-decls):
//   log, sin, cos, floor, where, square, abs, astype, full. Both anonymous
//   namespaces in this TU are the SAME unnamed namespace, so those using-decls
//   are visible in the block below. For subtract/multiply/divide we use the
//   scalar operator overloads (operator-, operator*, operator/) declared in
//   ops.h, found by ADL on mlx::core::array. Scalar constants are spelled as
//   plain `double` literals so the `array op T` / `T op array` templates apply.

#include "mlx_common.h"

using mlx::core::array;

namespace {

// Force a real-valued working dtype. lgamma/digamma are float functions; if the
// caller hands us an integer array (e.g. a Poisson count cast), promote to
// float32 so the transcendental sub-ops are well-defined. float32 matches MLX's
// no-float64 reality (mlx.cljs aliases float64->float32 anyway).
inline array to_float(const array& a) {
  if (mlx::core::issubdtype(a.dtype(), mlx::core::floating)) {
    return a;
  }
  return astype(a, mlx::core::float32);
}

// ---------------------------------------------------------------------------
// lgamma — log Gamma(x) for x>0, and via Euler reflection for x<0.5.
//
// METHOD: Lanczos approximation, g = 7, with the canonical 9-term coefficient
// set (the "g=7, n=9" Lanczos coefficients popularised by Numerical Recipes /
// Boost). For Re(x) >= 0.5 we use directly:
//
//   Gamma(x) = sqrt(2*pi) * t^(x-0.5) * exp(-t) * A_g(x),   t = x - 0.5 + g
//   A_g(x)   = c0 + sum_{k=1}^{8} c_k / (x - 1 + k)
//   lgamma(x) = 0.5*log(2*pi) + (x-0.5)*log(t) - t + log(A_g(x))
//
// For x < 0.5 we use Euler's reflection formula:
//
//   lgamma(x) = log(pi) - log|sin(pi*x)| - lgamma(1 - x)
//
// and note that (1 - x) > 0.5 there, so the Lanczos branch is evaluated on the
// reflected argument xr = where(x < 0.5, 1 - x, x), which is always >= 0.5.
//
// ACCURACY: the g=7/n=9 Lanczos set gives ~15 significant decimal digits in
// double precision over the right half-plane; evaluated in float32 here the
// effective accuracy is bounded by float32 (~1e-6..1e-7 relative) which is far
// below the noise floor of the log-prob gradients that consume it. Verified
// against known values (double eval): lgamma(0.5)=0.5723649,
// lgamma(1)=0, lgamma(5)=3.178054, lgamma(-0.5)=1.265512 — all match to <4e-7.
// Branch-free + gradient-seam-safe (see the double-where note in the header).
// ---------------------------------------------------------------------------
mlx::core::array lgamma_composite(const array& x_in) {
  array x = to_float(x_in);

  // Lanczos g = 7, n = 9 coefficients (Boost / Numerical Recipes set).
  static const double g = 7.0;
  static const double C0 = 0.99999999999980993;
  static const double C[8] = {
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.13857109526572012,
      9.9843695780195716e-6,
      1.5056327351493116e-7,
  };

  const double LOG_SQRT_2PI = 0.91893853320467274178;  // 0.5*log(2*pi)
  const double LOG_PI = 1.1447298858494001741;          // log(pi)
  const double PI = 3.14159265358979323846;

  // Reflected argument: for x < 0.5 evaluate Lanczos at (1 - x) which is > 0.5.
  array is_reflect = (x < 0.5);                  // bool array
  array xr = where(is_reflect, 1.0 - x, x);      // always >= 0.5

  // Series A_g(xr) = C0 + sum_k C[k] / (xr + k), k = 0..7.
  array series = full(xr.shape(), C0, xr.dtype());
  for (int k = 0; k < 8; ++k) {
    array denom = xr + static_cast<double>(k);   // xr + k, always >= 0.5 > 0
    series = series + (C[k] / denom);
  }

  // t = xr - 0.5 + g  (always >= g = 7 > 0).
  array t = (xr - 0.5) + g;

  // lgamma_pos = LOG_SQRT_2PI + (xr - 0.5)*log(t) - t + log(series).
  // This branch is universally finite/differentiable: xr>=0.5 => t>=7,
  // series>0, so log(t) and log(series) have finite local derivatives for ALL
  // input elements (both reflect and non-reflect). No seam guard needed here.
  array lg_pos =
      ((xr - 0.5) * log(t)) - t + log(series) + LOG_SQRT_2PI;

  // Reflection: lgamma(x) = log(pi) - log|sin(pi*x)| - lgamma(1 - x).
  // The sin/log|sin| terms blow up (and have inf local derivative) at integer
  // x. Feed them a SAFE argument so the dead-branch 0-cotangent stays finite
  // for non-reflect elements (the double-where trick — see header).
  // NOTE: `where`'s third arg must be an array (the array(double) ctor is
  // explicit, so a bare 0.25 would NOT compile); build the benign constant with
  // full() so it broadcasts against x's shape.
  array safe_const = full(x.shape(), 0.25, x.dtype());
  array x_safe = where(is_reflect, x, safe_const);
  array sin_pix = sin(PI * x_safe);
  array log_abs_sin = log(abs(sin_pix));
  array lg_reflect = (LOG_PI - log_abs_sin) - lg_pos;

  return where(is_reflect, lg_reflect, lg_pos);
}

// ---------------------------------------------------------------------------
// digamma — psi(x) = d/dx log Gamma(x).
//
// METHOD: recurrence-shift + asymptotic series, with reflection for x < 0.5.
//
//   Reflection (x < 0.5): psi(x) = psi(1 - x) - pi / tan(pi*x)
//   working on the reflected argument xr = where(x < 0.5, 1 - x, x) (>= 0.5).
//
//   Recurrence to push the (>= 0.5) argument into the asymptotic-accurate
//   region: psi(z) = psi(z + 1) - 1/z. Shift up by S = 6 steps:
//     psi(xr) = psi(xr + S) - sum_{j=0}^{S-1} 1/(xr + j)
//
//   Asymptotic (for w = xr + S, w large):
//     psi(w) ~ log(w) - 1/(2w)
//              - 1/(12 w^2) + 1/(120 w^4) - 1/(252 w^6) + 1/(240 w^8)
//
//   Then add the reflection correction for the x<0.5 elements:
//     psi(x) = psi(xr) - pi*cos(pi*x)/sin(pi*x)   (branch-free, avoids a tan op).
//
// ACCURACY: shifting up by 6 makes w >= 6.5, where the truncated asymptotic
// series (through w^-8) is accurate well past float32. Verified (double eval):
// digamma(1)=-0.5772157, digamma(0.5)=-1.963510, digamma(-0.5)=0.03648997 —
// all match to <5e-6. Branch-free + gradient-seam-safe (double-where on the
// cot term, same rationale as lgamma).
// ---------------------------------------------------------------------------
mlx::core::array digamma_composite(const array& x_in) {
  array x = to_float(x_in);

  const double PI = 3.14159265358979323846;
  const int S = 6;  // number of recurrence shifts

  array is_reflect = (x < 0.5);
  array xr = where(is_reflect, 1.0 - x, x);   // always >= 0.5

  // Recurrence subtraction: sum_{j=0}^{S-1} 1/(xr + j). xr>=0.5 => denoms > 0.
  array shift_sum = full(xr.shape(), 0.0, xr.dtype());
  for (int j = 0; j < S; ++j) {
    array denom = xr + static_cast<double>(j);
    shift_sum = shift_sum + (1.0 / denom);
  }

  // Asymptotic series at w = xr + S (>= 6.5 > 0).
  array w = xr + static_cast<double>(S);
  array inv = 1.0 / w;             // 1/w
  array inv2 = square(inv);        // 1/w^2

  // psi(w) ~ log(w) - 1/(2w)
  //          - (1/12)/w^2 + (1/120)/w^4 - (1/252)/w^6 + (1/240)/w^8
  // Horner in inv2:
  //   corr = inv2 * ( -1/12 + inv2*( 1/120 + inv2*( -1/252 + inv2*(1/240) ) ) )
  const double B2 = -1.0 / 12.0;
  const double B4 = 1.0 / 120.0;
  const double B6 = -1.0 / 252.0;
  const double B8 = 1.0 / 240.0;

  array corr = (B8 * inv2 + B6);
  corr = corr * inv2 + B4;
  corr = corr * inv2 + B2;
  corr = corr * inv2;  // multiply by leading inv2 to make all terms even powers

  array psi_w = log(w) - (0.5 * inv) + corr;

  // psi(xr) = psi(w) - shift_sum. Universally finite/differentiable for ALL
  // elements (xr>=0.5 => w>=6.5, denoms>0). No seam guard needed here.
  array psi_xr = psi_w - shift_sum;

  // Reflection correction for x < 0.5: psi(x) = psi(xr) - pi*cos(pi*x)/sin(pi*x).
  // cos/sin blow up (inf local derivative) at integer x; feed a SAFE argument
  // so the dead-branch 0-cotangent stays finite for non-reflect elements.
  // See lgamma: `where`'s y arg must be an array, so build the safe constant.
  array safe_const = full(x.shape(), 0.25, x.dtype());
  array x_safe = where(is_reflect, x, safe_const);
  array sin_pix = sin(PI * x_safe);
  array cos_pix = cos(PI * x_safe);
  array cot_term = (PI * cos_pix) / sin_pix;
  array psi_reflect = psi_xr - cot_term;

  return where(is_reflect, psi_reflect, psi_xr);
}

}  // namespace

extern "C" {

// log Gamma — composite Lanczos (g=7,n=9) + reflection. CUDA-native &
// autograd-composable. See lgamma_composite above for method/accuracy.
mlx_array* mlx_array_lgamma(mlx_array* handle) {
  MLX_GUARD_PTR("array_lgamma",
    auto arr = reinterpret_cast<array*>(handle);
    array result = lgamma_composite(*arr);
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// digamma psi(x) — composite recurrence-shift asymptotic + reflection.
// CUDA-native & autograd-composable. See digamma_composite above.
mlx_array* mlx_array_digamma(mlx_array* handle) {
  MLX_GUARD_PTR("array_digamma",
    auto arr = reinterpret_cast<array*>(handle);
    array result = digamma_composite(*arr);
    return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

}  // extern "C"
