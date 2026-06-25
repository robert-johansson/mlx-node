// GenMLX CUDA/Linux port — P2 mlx-sys FFI surface.
//
// Linear-algebra shims ported from the robert-johansson/mlx-node fork
// (mlx-sys/src/mlx_linalg.cpp, written against MLX 0.31.2 @49503b65) and
// reconciled against OUR frozen MLX 0.32.0 @b410f6c headers
// (crates/mlx-sys/mlx/mlx/linalg.h). Every signature below was checked
// against that header — see the signature-reconciliation note in the task
// output; no drift was found (the linalg API is identical between the two
// MLX revisions for the ops we expose).
//
// CPU-STREAM CONTRACT: cholesky / cholesky_inv / inv / tri_inv / solve /
// solve_triangular / qr / svd / eigh / eigvalsh are CPU-backend-only in MLX
// on BOTH Metal and CUDA. We therefore pass an explicit CPU stream
// (default_stream(Device::cpu)) so these ops run on the CPU backend of our
// CUDA build — exactly as the fork did. norm runs on the default stream
// (it has a GPU kernel), matching the fork.
//
// Headers (mlx_common.h) supply: `array`, `Device`, `Stream`,
// `default_stream` (via using-decls), the mlx_array opaque-handle struct,
// and the MLX_GUARD_* exception-to-sentinel macros. mlx_array* <-> array*
// is the same reinterpret_cast handle convention used across this crate
// (see mlx_fused_ops.cpp).

#include "mlx_common.h"
#include "mlx/linalg.h"

namespace linalg = mlx::core::linalg;

// Linalg factorizations/solves are CPU-only in MLX. Use a CPU stream
// explicitly so they run on the CPU backend under our CUDA build.
static mlx::core::Stream cpu_stream() {
  return default_stream(Device::cpu);
}

extern "C" {

// ============================================================================
// Linear Algebra Operations (CPU stream, exception-safe)
// ============================================================================

mlx_array* mlx_linalg_cholesky(mlx_array* handle, bool upper) {
  MLX_GUARD_PTR("linalg_cholesky",
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::cholesky(*arr, upper, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_linalg_solve(mlx_array* a_handle, mlx_array* b_handle) {
  MLX_GUARD_PTR("linalg_solve",
  auto a = reinterpret_cast<array*>(a_handle);
  auto b = reinterpret_cast<array*>(b_handle);
  array result = linalg::solve(*a, *b, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_linalg_solve_triangular(mlx_array* a_handle,
                                        mlx_array* b_handle,
                                        bool upper) {
  MLX_GUARD_PTR("linalg_solve_triangular",
  auto a = reinterpret_cast<array*>(a_handle);
  auto b = reinterpret_cast<array*>(b_handle);
  array result = linalg::solve_triangular(*a, *b, upper, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_linalg_inv(mlx_array* handle) {
  MLX_GUARD_PTR("linalg_inv",
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::inv(*arr, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_linalg_tri_inv(mlx_array* handle, bool upper) {
  MLX_GUARD_PTR("linalg_tri_inv",
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::tri_inv(*arr, upper, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_linalg_cholesky_inv(mlx_array* handle, bool upper) {
  MLX_GUARD_PTR("linalg_cholesky_inv",
  auto arr = reinterpret_cast<array*>(handle);
  array result = linalg::cholesky_inv(*arr, upper, cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// The void out-param decompositions null their out-params before the guarded
// body: validation throws (non-square input) and backend allocation throws
// both surface as null handles -> Rust check_handle -> catchable napi error.
void mlx_linalg_qr(mlx_array* handle,
                    mlx_array** q_out, mlx_array** r_out) {
  *q_out = nullptr;
  *r_out = nullptr;
  MLX_GUARD_VOID("linalg_qr",
  auto arr = reinterpret_cast<array*>(handle);
  // 0.32.0: qr -> std::pair<array,array> (header linalg.h:64). Matches fork.
  auto [q, r] = linalg::qr(*arr, cpu_stream());
  *q_out = reinterpret_cast<mlx_array*>(new array(std::move(q)));
  *r_out = reinterpret_cast<mlx_array*>(new array(std::move(r)));
  )
}

void mlx_linalg_svd(mlx_array* handle,
                     mlx_array** u_out, mlx_array** s_out, mlx_array** vt_out) {
  *u_out = nullptr;
  *s_out = nullptr;
  *vt_out = nullptr;
  MLX_GUARD_VOID("linalg_svd",
  auto arr = reinterpret_cast<array*>(handle);
  // 0.32.0: svd(a, stream) is the compute_uv=true overload (header
  // linalg.h:68) returning std::vector<array> = {U, S, Vt}. Matches fork.
  auto results = linalg::svd(*arr, cpu_stream());
  *u_out = reinterpret_cast<mlx_array*>(new array(std::move(results[0])));
  *s_out = reinterpret_cast<mlx_array*>(new array(std::move(results[1])));
  *vt_out = reinterpret_cast<mlx_array*>(new array(std::move(results[2])));
  )
}

void mlx_linalg_eigh(mlx_array* handle, const char* uplo,
                      mlx_array** eigvals_out, mlx_array** eigvecs_out) {
  *eigvals_out = nullptr;
  *eigvecs_out = nullptr;
  MLX_GUARD_VOID("linalg_eigh",
  auto arr = reinterpret_cast<array*>(handle);
  // 0.32.0: eigh(a, std::string UPLO, stream) -> std::pair<array,array>
  // (header linalg.h:112). Matches fork.
  auto [eigvals, eigvecs] = linalg::eigh(*arr, std::string(uplo), cpu_stream());
  *eigvals_out = reinterpret_cast<mlx_array*>(new array(std::move(eigvals)));
  *eigvecs_out = reinterpret_cast<mlx_array*>(new array(std::move(eigvecs)));
  )
}

mlx_array* mlx_linalg_eigvalsh(mlx_array* handle, const char* uplo) {
  MLX_GUARD_PTR("linalg_eigvalsh",
  auto arr = reinterpret_cast<array*>(handle);
  // 0.32.0: eigvalsh(a, std::string UPLO, stream) -> array (header
  // linalg.h:109). Matches fork.
  array result = linalg::eigvalsh(*arr, std::string(uplo), cpu_stream());
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// Norm can run on GPU — use the default stream (no CPU pin), matching fork.
mlx_array* mlx_linalg_norm(mlx_array* handle, double ord) {
  MLX_GUARD_PTR("linalg_norm",
  auto arr = reinterpret_cast<array*>(handle);
  // 0.32.0: norm(a, double ord, axis=nullopt, keepdims=false, stream={})
  // (header linalg.h:26). ord is double — exact match for our f64 FFI arg.
  array result = linalg::norm(*arr, ord);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_linalg_norm_default(mlx_array* handle) {
  MLX_GUARD_PTR("linalg_norm_default",
  auto arr = reinterpret_cast<array*>(handle);
  // 0.32.0: norm(a, axis=nullopt, keepdims=false, stream={}) — the
  // no-ord overload, 2-norm of flatten(a) (header linalg.h:54). Matches fork.
  array result = linalg::norm(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

}  // extern "C"
