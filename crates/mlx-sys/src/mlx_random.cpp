// GenMLX CUDA/Linux port — P2 mlx-sys FFI surface: keyed-PRNG shims.
//
// 12 functional (key-threaded, no global state) PRNG shims consumed by
// genmlx-core (genmlx.rs `mlx_sys::mlx_random_*`) and declared as
// `unsafe extern "C-unwind"` in crates/mlx-sys/src/lib.rs (lines 587-655).
//
// Ported from the robert-johansson/mlx-node fork (mlx-sys @ MLX 0.31.2 +
// 9 patches) and RECONCILED, signature by signature, against OUR FROZEN
// MLX 0.32.0 @ b410f6c headers (crates/mlx-sys/mlx/mlx/random.h). All 12
// mlx::core::random::* entry points used here are present in 0.32.0 with
// the SAME arg order / stream / dtype shapes the fork relied on — NO drift
// (the random surface did not change between 0.31.2 and 0.32.0). The
// reconciled overload picked for each call is noted inline below.
//
// Infrastructure pulled from mlx_common.h (P1): make_shape, to_mlx_dtype
// (decodes our BridgeDType i32 codes Float32=0..Int8=6), and the MLX_GUARD_*
// macros (record native error + return sentinel on any MLX exception, so a
// throw becomes a catchable napi error via sys::mlx_take_last_error()
// instead of an uncaught unwind across the extern "C" frame).

#include "mlx_common.h"
#include "mlx/random.h"

namespace rng = mlx::core::random;

// Key management (key, split, split_n): EAGER on CPU stream — eval()
// materializes keys immediately, breaking reference chains that would
// retain the key derivation graph. Threefry2x32 on 4 uint32 values is
// pure integer arithmetic — CPU stream avoids the Metal command buffer
// round-trip entirely.
//
// Sampling functions (normal, uniform, ...): LAZY — no eval(). The sampled
// array references an already-evaluated key (constant), so no chain growth.
// stop_gradient prevents autograd from differentiating through RandomBits.
// Evaluation is deferred to the caller's eval boundary, where it can be
// batched with other operations for a single GPU sync.

static mlx::core::Stream cpu_stream() {
  return mlx::core::default_stream(mlx::core::Device::cpu);
}

extern "C" {

// ============================================================================
// Key-based PRNG (functional, no global state)
// ============================================================================

// random.h:39  MLX_API array key(uint64_t seed);  — verbatim.
mlx_array* mlx_random_key(uint64_t seed) {
  MLX_GUARD_PTR("random_key",
  array result = rng::key(seed);
  result.eval();
  result = mlx::core::stop_gradient(result);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// random.h:58  MLX_API std::pair<array,array> split(const array& key,
//                                                   StreamOrDevice s = {});
//
// Called on EVERY GenMLX sample (key threading), so this is the highest-
// frequency Metal allocation site in inference loops. Unguarded, a malloc
// throw here (e.g. "[metal::malloc] Resource limit exceeded" under buffer-
// count pressure in long SBC sweeps) escaped through the extern "C" frame
// and aborted the process (genmlx-8w48). Out-params are nulled first so the
// Rust side surfaces the failure as a catchable napi error.
void mlx_random_split(mlx_array* key_handle,
                      mlx_array** k1_out, mlx_array** k2_out) {
  *k1_out = nullptr;
  *k2_out = nullptr;
  MLX_GUARD_VOID("random_split",
  auto key = reinterpret_cast<array*>(key_handle);
  auto [k1, k2] = rng::split(*key, cpu_stream());
  k1.eval();
  k2.eval();
  *k1_out = reinterpret_cast<mlx_array*>(new array(std::move(k1)));
  *k2_out = reinterpret_cast<mlx_array*>(new array(std::move(k2)));
  )
}

// random.h:61  MLX_API array split(const array& key, int num,
//                                   StreamOrDevice s = {});  — verbatim.
mlx_array* mlx_random_split_n(mlx_array* key_handle, int n) {
  MLX_GUARD_PTR("random_split_n",
  auto key = reinterpret_cast<array*>(key_handle);
  array result = rng::split(*key, n, cpu_stream());
  result.eval();
  result = mlx::core::stop_gradient(result);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// --- Key-based sampling functions (LAZY — no eval) ---

// random.h:64  MLX_API array uniform(const array& low, const array& high,
//   const Shape& shape, Dtype dtype = float32,
//   const std::optional<array>& key = std::nullopt, StreamOrDevice s = {});
// low/high are pre-cast to dtype arrays (matches the array,array overload).
mlx_array* mlx_random_uniform_key(mlx_array* key_handle,
                                   const int64_t* shape, size_t ndim,
                                   float low, float high, int32_t dtype) {
  MLX_GUARD_PTR("random_uniform_key",
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  array lo = mlx::core::astype(array(low), dt);
  array hi = mlx::core::astype(array(high), dt);
  auto result = mlx::core::stop_gradient(
      rng::uniform(lo, hi, sh, dt, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// random.h:124  inline array normal(const Shape& shape, const Dtype dtype,
//   const std::optional<array>& key = std::nullopt, StreamOrDevice s = {});
// (std-normal, loc/scale defaulted) — the fork's call resolves to THIS
// overload, distinct from the 0.32.0 (shape,dtype,loc,scale,key,s) primitive.
mlx_array* mlx_random_normal_key(mlx_array* key_handle,
                                  const int64_t* shape, size_t ndim,
                                  int32_t dtype) {
  MLX_GUARD_PTR("random_normal_key",
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::normal(sh, dt, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// random.h:186 template<T> array bernoulli(T p, const Shape& shape,
//   const std::optional<array>& key = std::nullopt, StreamOrDevice s = {});
// prob (float) selects the templated scalar-p overload — array(p) internally.
mlx_array* mlx_random_bernoulli_key(mlx_array* key_handle,
                                     float prob,
                                     const int64_t* shape, size_t ndim) {
  MLX_GUARD_PTR("random_bernoulli_key",
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto result = mlx::core::stop_gradient(
      rng::bernoulli(prob, sh, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// random.h:234  MLX_API array categorical(const array& logits, int axis = -1,
//   const std::optional<array>& key = std::nullopt, StreamOrDevice s = {});
// Caller passes an explicit axis; this is the no-shape / no-num_samples
// overload (one sample per row).
mlx_array* mlx_random_categorical_key(mlx_array* key_handle,
                                       mlx_array* logits_handle,
                                       int axis) {
  MLX_GUARD_PTR("random_categorical_key",
  auto key = reinterpret_cast<array*>(key_handle);
  auto logits = reinterpret_cast<array*>(logits_handle);
  auto result = mlx::core::stop_gradient(
      rng::categorical(*logits, axis, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// random.h:156 template<T,U> array randint(T low, U high, const Shape& shape,
//   Dtype dtype = int32, const std::optional<array>& key = std::nullopt,
//   StreamOrDevice s = {});  — int low/high select the templated overload.
mlx_array* mlx_random_randint_key(mlx_array* key_handle,
                                   int low, int high,
                                   const int64_t* shape, size_t ndim,
                                   int32_t dtype) {
  MLX_GUARD_PTR("random_randint_key",
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::randint(low, high, sh, dt, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// random.h:214  MLX_API array gumbel(const Shape& shape, Dtype dtype = float32,
//   const std::optional<array>& key = std::nullopt, StreamOrDevice s = {});
mlx_array* mlx_random_gumbel_key(mlx_array* key_handle,
                                  const int64_t* shape, size_t ndim,
                                  int32_t dtype) {
  MLX_GUARD_PTR("random_gumbel_key",
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::gumbel(sh, dt, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// random.h:256  inline array laplace(const Shape& shape, const Dtype dtype,
//   const std::optional<array>& key = std::nullopt, StreamOrDevice s = {});
// (loc=0,scale=1 defaulted) — distinct from the (shape,dtype,loc,scale,...)
// primitive at random.h:241.
mlx_array* mlx_random_laplace_key(mlx_array* key_handle,
                                   const int64_t* shape, size_t ndim,
                                   int32_t dtype) {
  MLX_GUARD_PTR("random_laplace_key",
  auto key = reinterpret_cast<array*>(key_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::laplace(sh, dt, std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// random.h:199  MLX_API array truncated_normal(const array& lower,
//   const array& upper, const Shape& shape, Dtype dtype = float32,
//   const std::optional<array>& key = std::nullopt, StreamOrDevice s = {});
mlx_array* mlx_random_truncated_normal_key(mlx_array* key_handle,
                                            mlx_array* lower_handle,
                                            mlx_array* upper_handle,
                                            const int64_t* shape, size_t ndim,
                                            int32_t dtype) {
  MLX_GUARD_PTR("random_truncated_normal_key",
  auto key = reinterpret_cast<array*>(key_handle);
  auto lower = reinterpret_cast<array*>(lower_handle);
  auto upper = reinterpret_cast<array*>(upper_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::truncated_normal(*lower, *upper, sh, dt,
                            std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// random.h:139  MLX_API array multivariate_normal(const array& mean,
//   const array& cov, const Shape& shape, Dtype dtype,
//   const std::optional<array>& key = std::nullopt, StreamOrDevice s = {});
mlx_array* mlx_random_multivariate_normal_key(mlx_array* key_handle,
                                               mlx_array* mean_handle,
                                               mlx_array* cov_handle,
                                               const int64_t* shape,
                                               size_t ndim,
                                               int32_t dtype) {
  MLX_GUARD_PTR("random_multivariate_normal_key",
  auto key = reinterpret_cast<array*>(key_handle);
  auto mean = reinterpret_cast<array*>(mean_handle);
  auto cov = reinterpret_cast<array*>(cov_handle);
  auto sh = make_shape(shape, ndim);
  auto dt = to_mlx_dtype(dtype);
  auto result = mlx::core::stop_gradient(
      rng::multivariate_normal(*mean, *cov, sh, dt,
                               std::optional<array>(*key)));
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

}  // extern "C"
