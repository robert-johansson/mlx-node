#include "mlx_common.h"

// =============================================================================
// mlx_array_searchsorted — COMPOSITE re-implementation (genmlx CUDA/Linux port,
// phase P2). MLX 0.32.0 @ b410f6c does NOT ship mlx::core::searchsorted (the
// donor fork @49503b65 patched it in; we keep MLX frozen and synthesize it from
// primitives our headers DO have: expand_dims, less/less_equal, sum, astype).
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
// Implementation: broadcast values[:,None] ([N,1]) against sorted[None,:]
// ([1,M]), evaluate the predicate to a boolean [N,M], then sum over the sorted
// axis (axis 1). The boolean sum is the per-value count == the insertion index.
//
// CRITICAL — int32 result. On this MLX, the sum of a bool/comparison array
// comes back float32. The downstream systematic-resampling callers feed this
// directly into integer gather/take:
//   * src/genmlx/inference/compiled_smc.cljs:50-52 — `ancestors` (no astype)
//     flows straight into `mx/take-idx`, which needs an int32 index array.
//   * src/genmlx/vectorized.cljs:31-33 — `indices` is later astype'd to int32,
//     but only after an `mx/minimum` against an int32 scalar.
// So we cast the result to int32 HERE; both call sites then behave identically
// to the donor's native primitive.
// =============================================================================

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
  // (numpy/mlx searchsorted preserves the values shape).
  const Shape values_shape = values->shape();
  array values_flat = reshape(*values, {-1});

  // values[:, None] -> [N, 1] ; sorted[None, :] -> [1, M].
  array values_col = expand_dims(values_flat, 1); // values varies along axis 0
  array sorted_row = expand_dims(*sorted, 0);  // sorted varies along axis 1

  // Predicate broadcasts to [N, M]; sum over the sorted axis (1) -> [N].
  // left  => sorted <  v  (strictly less)
  // right => sorted <= v
  array mask = right ? less_equal(sorted_row, values_col)
                     : less(sorted_row, values_col);

  // Sum the boolean counts along the sorted axis, then cast to int32 so the
  // result is a valid gather/take index array (see header note above).
  array counts = sum(mask, /*axis=*/1, /*keepdims=*/false);
  array result = astype(counts, mlx::core::int32);
  if (values_shape.size() != 1) {
    result = reshape(result, values_shape);
  }

  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

}  // extern "C"
