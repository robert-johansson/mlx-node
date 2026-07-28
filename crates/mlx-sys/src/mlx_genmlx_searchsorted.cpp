#include "mlx_common.h"

// =============================================================================
// mlx_array_searchsorted — composite re-implementation (genmlx CUDA/Linux port,
// phase P2). Upstream MLX still does NOT ship mlx::core::searchsorted (checked
// again at pin a27ddcaef; the donor fork @49503b65 patched one in), so we
// synthesize it from primitives our headers DO have.
//
// Signature matches the fork's lib.rs extern decl and genmlx-core's
// sys::mlx_array_searchsorted call:
//     mlx_array* mlx_array_searchsorted(mlx_array* sorted, mlx_array* values,
//                                       bool right)
//
// Semantics (numpy/mlx searchsorted on a 1-D sorted array):
//   left  (right == false, default): for each value v, the index where v would
//         be inserted to keep `sorted` sorted, inserting BEFORE equal elements
//         => count of sorted elements strictly less than v
//         (sorted[i-1] < v <= sorted[i]). Predicate: sorted < v.
//   right (right == true): insert AFTER equal elements
//         => count of sorted elements <= v
//         (sorted[i-1] <= v < sorted[i]). Predicate: sorted <= v.
//
// TWO PATHS, one meaning (genmlx-l5m2)
// ------------------------------------
// The original implementation had only the BROADCAST path: values[:,None]
// against sorted[None,:], predicate to a boolean [N,M], sum over the sorted
// axis. Simple and fast for small inputs — but it materializes N*M elements,
// so cost is quadratic in BOTH time and memory. That is invisible at test
// scale and lethal at real scale: N=1e5 queries x M=6e5 sorted materialized a
// ~60GB mask per call on the Thor (tabular/bfs, genmlx-69wh) and OOM-rebooted
// the box. The systematic-resampling callers are the ones that matter here —
// src/genmlx/vectorized.cljs:31 and src/genmlx/inference/compiled_smc.cljs:50
// pass N particles against an M=N cumsum, i.e. exactly N^2 per resample step.
//
// So above a size cap we take the BINARY-SEARCH path instead: run the standard
// lo/hi bisection over the sorted axis for every query AT ONCE, ceil(log2(M+1))
// iterations of gather + compare + select. O(N) memory and O(N log M) work, no
// [N,M] temporary ever built. At N=M=1e5 that is ~17 steps on 1e5-element
// arrays instead of a 1e10-element mask.
//
// Why keep the broadcast path at all: below the cap it is ONE kernel against
// the bisection's ~6 per step, and small-N searchsorted sits inside SMC hot
// loops where per-eval launch latency is already the measured bottleneck on
// discrete cards (genmlx-6qz4). The cap is where the mask stops being cheap,
// not a correctness boundary — both paths compute the same counts, and
// searchsorted_test asserts that across the crossover.
//
// PRECONDITION divergence, stated honestly: the broadcast path counts elements
// satisfying the predicate, so it returns a sensible count even for an
// UNSORTED `sorted` argument; bisection assumes the predicate is monotone
// along the axis, as numpy/mlx searchsorted requires. Passing an unsorted
// array was always a precondition violation; above the cap it now yields a
// different wrong answer rather than the same wrong answer. Every in-tree
// caller passes a cumsum or an explicitly sorted array.
// NaN behaviour is unchanged and agrees between the paths: NaN compares false
// everywhere, so a NaN query keeps the predicate false at every step and lands
// at index 0, exactly as the mask-sum did.
//
// CRITICAL — int32 result. On this MLX, the sum of a bool/comparison array
// comes back float32. The downstream systematic-resampling callers feed this
// directly into integer gather/take:
//   * src/genmlx/inference/compiled_smc.cljs:50-52 — `ancestors` (no astype)
//     flows straight into `mx/take-idx`, which needs an int32 index array.
//   * src/genmlx/vectorized.cljs:31-33 — `indices` is later astype'd to int32,
//     but only after an `mx/minimum` against an int32 scalar.
// So we cast the result to int32 HERE; both call sites then behave identically
// to the donor's native primitive. The bisection path is int32 throughout.
// =============================================================================

namespace {

// Element cap for the broadcast path: at most this many mask elements may be
// materialized. 4Mi bool elements is ~4 MiB — the point where "one cheap fused
// kernel" stops being true. For the square SMC case (M == N) this puts the
// crossover at N ~ 2048 particles.
constexpr int64_t kBroadcastMaxElems = 4LL << 20;

// Bisection steps needed to resolve an answer in [0, m]: the smallest k with
// 2^k >= m+1. m=1 -> 1, m=2 -> 2, m=6e5 -> 20.
int bisect_steps(int64_t m) {
  int k = 0;
  while ((int64_t(1) << k) < (m + 1)) ++k;
  return k;
}

}  // namespace

extern "C" {

mlx_array* mlx_array_searchsorted(mlx_array* sorted_handle,
                                  mlx_array* values_handle,
                                  bool right) {
  MLX_GUARD_PTR("array_searchsorted",
  auto sorted = reinterpret_cast<array*>(sorted_handle);
  auto values = reinterpret_cast<array*>(values_handle);

  // N-D values (genmlx-fqqx): the [N,1]x[1,M] broadcast trick is 1-D-only —
  // a [2,2] values array became [2,1,2] and could not broadcast against
  // [1,M]. Flatten first, reshape the index result back at the end
  // (numpy/mlx searchsorted preserves the values shape). `sorted` is flattened
  // on both paths so they agree on any input shape.
  const Shape values_shape = values->shape();
  array values_flat = reshape(*values, {-1});
  array sorted_flat = reshape(*sorted, {-1});

  const int64_t n = static_cast<int64_t>(values_flat.size());
  const int64_t m = static_cast<int64_t>(sorted_flat.size());

  array result = zeros(values_flat.shape(), mlx::core::int32);

  if (m > 0 && n > 0) {
    if (n * m <= kBroadcastMaxElems) {
      // ---- broadcast path -------------------------------------------------
      // values[:, None] -> [N, 1] ; sorted[None, :] -> [1, M].
      array values_col = expand_dims(values_flat, 1);  // varies along axis 0
      array sorted_row = expand_dims(sorted_flat, 0);  // varies along axis 1

      // Predicate broadcasts to [N, M]; sum over the sorted axis (1) -> [N].
      // left  => sorted <  v  (strictly less)
      // right => sorted <= v
      array mask = right ? less_equal(sorted_row, values_col)
                         : less(sorted_row, values_col);

      // Sum the boolean counts along the sorted axis, then cast to int32 so
      // the result is a valid gather/take index array (see header note).
      result = astype(sum(mask, /*axis=*/1, /*keepdims=*/false),
                      mlx::core::int32);
    } else {
      // ---- bisection path -------------------------------------------------
      // Invariant per query: the answer lies in [lo, hi]. Each step probes
      // mid = floor((lo+hi)/2); if the predicate still holds there the answer
      // is above it (lo = mid+1), else at or below it (hi = mid). After
      // bisect_steps(m) iterations lo == hi == the count we want.
      const array one = array(1, mlx::core::int32);
      const array two = array(2, mlx::core::int32);
      // mid can only reach m once lo == hi == m (the all-predicate-true,
      // already-converged state), where a gather at m would be out of bounds.
      // Clamping to m-1 keeps that state fixed: the predicate holds at m-1, so
      // lo = (m-1)+1 = m and hi = m, unchanged. Every unconverged step has
      // mid <= m-1 already, so the clamp is invisible to them.
      const array last = array(static_cast<int32_t>(m - 1), mlx::core::int32);

      array lo = zeros(values_flat.shape(), mlx::core::int32);
      array hi = full(values_flat.shape(), static_cast<int32_t>(m),
                      mlx::core::int32);

      const int steps = bisect_steps(m);
      for (int step = 0; step < steps; ++step) {
        array mid = minimum(floor_divide(add(lo, hi), two), last);
        array probe = take(sorted_flat, mid, /*axis=*/0);
        array holds = right ? less_equal(probe, values_flat)
                            : less(probe, values_flat);
        lo = where(holds, add(mid, one), lo);
        hi = where(holds, hi, mid);
      }
      result = lo;
    }
  }

  if (values_shape.size() != 1) {
    result = reshape(result, values_shape);
  }

  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

}  // extern "C"
