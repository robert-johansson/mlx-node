#include "mlx_common.h"
#include "mlx/random.h"

namespace rng = mlx::core::random;

// Key management (key, split, split_n): EAGER — eval() materializes keys
// immediately, breaking reference chains that would retain the entire key
// derivation graph in GPU memory. Keys are tiny (2 uint32), so eval is cheap.
//
// Sampling functions (normal, uniform, ...): LAZY — no eval(). The sampled
// array references an already-evaluated key (constant), so no chain growth.
// stop_gradient prevents autograd from differentiating through RandomBits.
// Evaluation is deferred to the caller's eval boundary, where it can be
// batched with other operations for a single GPU sync.

extern "C" {

// ============================================================================
// Key-based PRNG (functional, no global state)
// ============================================================================

mlx_array* mlx_random_key(uint64_t seed) {
  array result = rng::key(seed);
  result.eval();
  result = mlx::core::stop_gradient(result);
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
  result = mlx::core::stop_gradient(result);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// --- Key-based sampling functions (LAZY — no eval) ---

mlx_array* mlx_random_uniform_key(mlx_array* key_handle,
                                   const int64_t* shape, size_t ndim,
                                   float low, float high, int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  array lo = mlx::core::astype(array(low), dt);
  array hi = mlx::core::astype(array(high), dt);
  auto result = mlx::core::stop_gradient(
      rng::uniform(lo, hi, sh, dt, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_normal_key(mlx_array* key_handle,
                                  const int64_t* shape, size_t ndim,
                                  int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::normal(sh, dt, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_bernoulli_key(mlx_array* key_handle,
                                     float prob,
                                     const int64_t* shape, size_t ndim) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto result = mlx::core::stop_gradient(
      rng::bernoulli(prob, sh, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_categorical_key(mlx_array* key_handle,
                                       mlx_array* logits_handle,
                                       int axis) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto logits = reinterpret_cast<array*>(logits_handle);
  auto result = mlx::core::stop_gradient(
      rng::categorical(*logits, axis, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_randint_key(mlx_array* key_handle,
                                   int low, int high,
                                   const int64_t* shape, size_t ndim,
                                   int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::randint(low, high, sh, dt, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_gumbel_key(mlx_array* key_handle,
                                  const int64_t* shape, size_t ndim,
                                  int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::gumbel(sh, dt, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_random_laplace_key(mlx_array* key_handle,
                                   const int64_t* shape, size_t ndim,
                                   int32_t dtype) {
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::laplace(sh, dt, std::optional<array>(*key)));
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
  auto result = mlx::core::stop_gradient(
      rng::truncated_normal(*lower, *upper, sh, dt,
                            std::optional<array>(*key)));
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
  auto result = mlx::core::stop_gradient(
      rng::multivariate_normal(*mean, *cov, sh, dt,
                               std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

}  // extern "C"
