#include "mlx_common.h"
#include "mlx/linalg.h"

namespace linalg = mlx::core::linalg;

// Linalg ops are CPU-only in MLX. Use a CPU stream explicitly.
static mlx::core::Stream cpu_stream() {
  return default_stream(Device::cpu);
}

extern "C" {

// ============================================================================
// Linear Algebra Operations (CPU stream, exception-safe)
// ============================================================================

mlx_array* mlx_linalg_cholesky(mlx_array* handle, bool upper) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::cholesky(*arr, upper, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_linalg_solve(mlx_array* a_handle, mlx_array* b_handle) {
  auto a = reinterpret_cast<array*>(a_handle);
  auto b = reinterpret_cast<array*>(b_handle);
  array result = linalg::solve(*a, *b, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_linalg_solve_triangular(mlx_array* a_handle,
                                        mlx_array* b_handle,
                                        bool upper) {
  auto a = reinterpret_cast<array*>(a_handle);
  auto b = reinterpret_cast<array*>(b_handle);
  array result = linalg::solve_triangular(*a, *b, upper, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_linalg_inv(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::inv(*arr, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_linalg_tri_inv(mlx_array* handle, bool upper) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::tri_inv(*arr, upper, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_linalg_cholesky_inv(mlx_array* handle, bool upper) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::cholesky_inv(*arr, upper, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

void mlx_linalg_qr(mlx_array* handle,
                    mlx_array** q_out, mlx_array** r_out) {
  auto arr = reinterpret_cast<array*>(handle);
  auto [q, r] = linalg::qr(*arr, cpu_stream());
  *q_out = reinterpret_cast<mlx_array*>(new array(std::move(q)));
  *r_out = reinterpret_cast<mlx_array*>(new array(std::move(r)));
}

void mlx_linalg_svd(mlx_array* handle,
                     mlx_array** u_out, mlx_array** s_out, mlx_array** vt_out) {
  auto arr = reinterpret_cast<array*>(handle);
  auto results = linalg::svd(*arr, cpu_stream());
  *u_out = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
  *s_out = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
  *vt_out = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
}

void mlx_linalg_eigh(mlx_array* handle, const char* uplo,
                      mlx_array** eigvals_out, mlx_array** eigvecs_out) {
  auto arr = reinterpret_cast<array*>(handle);
  auto [eigvals, eigvecs] = linalg::eigh(*arr, std::string(uplo), cpu_stream());
  *eigvals_out = reinterpret_cast<mlx_array*>(new array(std::move(eigvals)));
  *eigvecs_out = reinterpret_cast<mlx_array*>(new array(std::move(eigvecs)));
}

mlx_array* mlx_linalg_eigvalsh(mlx_array* handle, const char* uplo) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::eigvalsh(*arr, std::string(uplo), cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

// Norm can run on GPU — no CPU stream needed
mlx_array* mlx_linalg_norm(mlx_array* handle, double ord) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::norm(*arr, ord);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

mlx_array* mlx_linalg_norm_default(mlx_array* handle) {
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::norm(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
}

}  // extern "C"
