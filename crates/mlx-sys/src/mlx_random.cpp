#include "mlx_common.h"
#include "mlx/random.h"

namespace rng = mlx::core::random;

// All PRNG functions call result.eval() to materialize the random values.
// This ensures random arrays are concrete constants — not lazy graph nodes.
// Without eval(), random ops inside autograd callbacks would create RandomBits
// nodes in the computation graph, and MLX's VJP can't differentiate through them.
// With eval(), random values are treated as constants in the backward pass,
// matching node-mlx's behavior where random ops produce evaluated results.

extern "C" {

// ============================================================================
// Key-based PRNG (functional, no global state)
// ============================================================================

mlx_array* mlx_random_key(uint64_t seed) {
  array result = rng::key(seed);
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

void mlx_random_split(mlx_array* key_handle,
                      mlx_array** k1_out, mlx_array** k2_out) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto [k1, k2] = rng::split(*key);
  k1.eval();
  k2.eval();
  *k1_out = reinterpret_cast<mlx_array*>(new array(std::move(k1)));
  *k2_out = reinterpret_cast<mlx_array*>(new array(std::move(k2)));
}

mlx_array* mlx_random_split_n(mlx_array* key_handle, int n) {
  auto key = reinterpret_cast<array*>(key_handle);
  array result = rng::split(*key, n);
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// --- Key-based sampling functions ---

mlx_array* mlx_random_uniform_key(mlx_array* key_handle,
                                   const int64_t* shape, size_t ndim,
                                   float low, float high, int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  array lo = mlx::core::astype(array(low), dt);
  array hi = mlx::core::astype(array(high), dt);
  array result = rng::uniform(lo, hi, sh, dt, std::optional<array>(*key));
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_normal_key(mlx_array* key_handle,
                                  const int64_t* shape, size_t ndim,
                                  int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  array result = rng::normal(sh, dt, std::optional<array>(*key));
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_bernoulli_key(mlx_array* key_handle,
                                     float prob,
                                     const int64_t* shape, size_t ndim) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  array result = rng::bernoulli(prob, sh, std::optional<array>(*key));
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_categorical_key(mlx_array* key_handle,
                                       mlx_array* logits_handle,
                                       int axis) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto logits = reinterpret_cast<array*>(logits_handle);
  array result = rng::categorical(*logits, axis, std::optional<array>(*key));
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_randint_key(mlx_array* key_handle,
                                   int low, int high,
                                   const int64_t* shape, size_t ndim,
                                   int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  array result = rng::randint(low, high, sh, dt, std::optional<array>(*key));
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_gumbel_key(mlx_array* key_handle,
                                  const int64_t* shape, size_t ndim,
                                  int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  array result = rng::gumbel(sh, dt, std::optional<array>(*key));
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_laplace_key(mlx_array* key_handle,
                                   const int64_t* shape, size_t ndim,
                                   int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  array result = rng::laplace(sh, dt, std::optional<array>(*key));
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_truncated_normal_key(mlx_array* key_handle,
                                            mlx_array* lower_handle,
                                            mlx_array* upper_handle,
                                            const int64_t* shape, size_t ndim,
                                            int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto lower = reinterpret_cast<array*>(lower_handle);
  auto upper = reinterpret_cast<array*>(upper_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  array result = rng::truncated_normal(*lower, *upper, sh, dt,
                                        std::optional<array>(*key));
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_multivariate_normal_key(mlx_array* key_handle,
                                               mlx_array* mean_handle,
                                               mlx_array* cov_handle,
                                               const int64_t* shape,
                                               size_t ndim,
                                               int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto mean = reinterpret_cast<array*>(mean_handle);
  auto cov = reinterpret_cast<array*>(cov_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  array result = rng::multivariate_normal(*mean, *cov, sh, dt,
                                           std::optional<array>(*key));
  result.eval();
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

}  // extern "C"
