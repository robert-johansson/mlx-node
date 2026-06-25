// mlx_genmlx_ext.cpp — GenMLX FFI extension surface (CUDA/Linux graft, phase P2).
//
// This translation unit owns two things for the GenMLX FFI graft:
//
//   (a) The SINGLE definition of the native-error trio
//       mlx_record_native_error / mlx_take_last_error + the two thread-locals
//       g_mlx_last_error / g_mlx_has_error. These are DECLARED extern in
//       mlx_common.h (so every MLX_GUARD_* expansion across the GenMLX shim
//       TUs links against them) but must be DEFINED in exactly one place — here.
//       Body lifted verbatim from the donor fork's mlx_array_ops.cpp ~45-61.
//
//   (b) The 14 "present in upstream MLX 0.32.0" array-op shims that our tree
//       lacked, LIFTED from the donor fork's mlx_array_ops.cpp and reconciled
//       against OUR frozen MLX 0.32.0 headers (ops.h / einsum.h). These are the
//       composite-free ops — every mlx::core symbol they call EXISTS in 0.32.0
//       (verified against ~/code/mlx/mlx-node/crates/mlx-sys/mlx/mlx/ops.h):
//         topk, logaddexp, logcumsumexp, nan_to_num, outer, inner, flatten,
//         trace, expm1, erfinv, all, any, diag, einsum.
//       Deliberately EXCLUDED: sigmoid/softmax/erf (already wrapped in our
//       tree) and the 5 fork-patched absent ops (lgamma, digamma, bessel_i0e,
//       bessel_i1e, searchsorted) which DO NOT exist in 0.32.0 and are
//       re-implemented as composites in a separate piece.
//
// einsum is NOT declared in ops.h — it lives in mlx/einsum.h, included below.
//
// Every shim body is wrapped in MLX_GUARD_PTR (from mlx_common.h): on an MLX
// exception it records the message into the thread-local slot above (read by
// genmlx-core via sys::mlx_take_last_error) and returns nullptr, which the Rust
// check_handle turns into a catchable napi error instead of aborting.

#include "mlx_common.h"

// einsum() is declared here, not in ops.h.
#include "mlx/einsum.h"

namespace {
using mlx::core::all;
using mlx::core::any;
using mlx::core::diag;
using mlx::core::einsum;
using mlx::core::erfinv;
using mlx::core::expm1;
using mlx::core::inner;
using mlx::core::logaddexp;
using mlx::core::logcumsumexp;
using mlx::core::nan_to_num;
using mlx::core::outer;
using mlx::core::topk;
using mlx::core::trace;
// NOTE: mlx::core::flatten is already brought in via `using` in mlx_common.h's
// anonymous namespace; re-declaring it here in this TU's own anonymous
// namespace would be a redefinition. It is referenced unqualified below and
// resolves through that header `using`.
}  // namespace

extern "C" {

// ============================================================================
// (a) Native error surface — SINGLE definition (declared in mlx_common.h).
//
// Thread-local: each MLX worker thread gets its own slot. The returned
// C-string is valid until the next record on the same thread, so the Rust side
// must copy it immediately (napi Error::from_reason does). Body lifted verbatim
// from the donor fork's mlx_array_ops.cpp ~45-61.
// ============================================================================

static thread_local std::string g_mlx_last_error;
static thread_local bool g_mlx_has_error = false;

void mlx_record_native_error(const char* context, const char* detail) {
  // Store only the MLX detail (e.what()); the Rust side adds its own call-site
  // context, so prefixing it here too would double it. `context` is still used
  // by the file tracer (mlx_report_error -> mlx_trace_native_error).
  (void)context;
  g_mlx_last_error.assign(detail ? detail : "");
  g_mlx_has_error = true;
}

const char* mlx_take_last_error() {
  if (!g_mlx_has_error) return nullptr;
  g_mlx_has_error = false;
  return g_mlx_last_error.c_str();
}

// ============================================================================
// (b) Present array-op shims lifted from the donor fork, reconciled against
// OUR MLX 0.32.0 headers. Each maps 1:1 to a mlx_sys lib.rs extern decl whose
// exact shape genmlx-core's sys::mlx_array_* calls expect.
// ============================================================================

// --- Unary special functions ---

mlx_array* mlx_array_expm1(mlx_array* handle) {
  MLX_GUARD_PTR("array_expm1",
  auto arr = reinterpret_cast<array*>(handle);
  array result = expm1(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_array_erfinv(mlx_array* handle) {
  MLX_GUARD_PTR("array_erfinv",
  auto arr = reinterpret_cast<array*>(handle);
  array result = erfinv(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// --- Binary ops ---

mlx_array* mlx_array_logaddexp(mlx_array* lhs, mlx_array* rhs) {
  MLX_GUARD_PTR("array_logaddexp",
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = logaddexp(*a, *b);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// --- nan_to_num (unary with optional posinf/neginf) ---
// ops.h:536  array nan_to_num(const array&, float nan, optional<float> posinf,
//                             optional<float> neginf, StreamOrDevice)

mlx_array* mlx_array_nan_to_num(mlx_array* handle,
                                float nan_val,
                                bool has_posinf, float posinf_val,
                                bool has_neginf, float neginf_val) {
  MLX_GUARD_PTR("array_nan_to_num",
  auto arr = reinterpret_cast<array*>(handle);
  auto posinf = has_posinf ? std::optional<float>(posinf_val) : std::nullopt;
  auto neginf = has_neginf ? std::optional<float>(neginf_val) : std::nullopt;
  array result = nan_to_num(*arr, nan_val, posinf, neginf);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// --- Shape / matrix ops ---

mlx_array* mlx_array_flatten(mlx_array* handle) {
  MLX_GUARD_PTR("array_flatten",
  auto arr = reinterpret_cast<array*>(handle);
  array result = flatten(*arr);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_array_inner(mlx_array* lhs, mlx_array* rhs) {
  MLX_GUARD_PTR("array_inner",
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = inner(*a, *b);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_array_outer(mlx_array* lhs, mlx_array* rhs) {
  MLX_GUARD_PTR("array_outer",
  auto a = reinterpret_cast<array*>(lhs);
  auto b = reinterpret_cast<array*>(rhs);
  array result = outer(*a, *b);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// ops.h:1621  array diag(const array&, int k = 0, StreamOrDevice)
mlx_array* mlx_array_diag(mlx_array* handle, int k) {
  MLX_GUARD_PTR("array_diag",
  auto arr = reinterpret_cast<array*>(handle);
  array result = diag(*arr, k);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// ops.h:1631  array trace(const array&, int offset, int axis1, int axis2,
//                         StreamOrDevice)
mlx_array* mlx_array_trace(mlx_array* handle, int offset, int axis1, int axis2) {
  MLX_GUARD_PTR("array_trace",
  auto arr = reinterpret_cast<array*>(handle);
  array result = trace(*arr, offset, axis1, axis2);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// --- Reduction ops ---
// ops.h:573  array all(const array&, const vector<int>& axes, bool keepdims, ...)
// ops.h:544  array all(const array&, bool keepdims, ...)

mlx_array* mlx_array_all(mlx_array* handle,
                         const int32_t* axes, size_t axes_len,
                         bool keepdims) {
  MLX_GUARD_PTR("array_all",
  auto arr = reinterpret_cast<array*>(handle);
  array result = (axes_len == 0)
                     ? all(*arr, keepdims)
                     : all(*arr, make_axes(axes, axes_len), keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

mlx_array* mlx_array_any(mlx_array* handle,
                         const int32_t* axes, size_t axes_len,
                         bool keepdims) {
  MLX_GUARD_PTR("array_any",
  auto arr = reinterpret_cast<array*>(handle);
  array result = (axes_len == 0)
                     ? any(*arr, keepdims)
                     : any(*arr, make_axes(axes, axes_len), keepdims);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// ops.h:846  array topk(const array&, int k, int axis, StreamOrDevice)
mlx_array* mlx_array_topk(mlx_array* handle, int k, int axis) {
  MLX_GUARD_PTR("array_topk",
  auto arr = reinterpret_cast<array*>(handle);
  array result = topk(*arr, k, axis);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// ops.h:856  array logcumsumexp(const array&, int axis, bool reverse = false,
//                               bool inclusive = true, StreamOrDevice)
// genmlx-core passes only (axis, reverse); `inclusive` keeps its true default.
mlx_array* mlx_array_logcumsumexp(mlx_array* handle, int axis, bool reverse) {
  MLX_GUARD_PTR("array_logcumsumexp",
  auto arr = reinterpret_cast<array*>(handle);
  array result = logcumsumexp(*arr, axis, reverse);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

// --- Einstein summation ---
// einsum.h  array einsum(const string& subscripts, const vector<array>&, ...)
mlx_array* mlx_array_einsum(const char* subscripts,
                            mlx_array* const* operand_handles,
                            size_t operand_count) {
  MLX_GUARD_PTR("array_einsum",
  std::vector<array> operands;
  operands.reserve(operand_count);
  for (size_t i = 0; i < operand_count; i++) {
    auto arr = reinterpret_cast<array*>(operand_handles[i]);
    operands.emplace_back(*arr);
  }
  array result = einsum(std::string(subscripts), operands);
  return reinterpret_cast<mlx_array*>(new array(std::move(result)));
  )
}

}  // extern "C"
