//! GenMLX module-level NAPI exports.
//!
//! Provides standalone functions that accept JS `number | MxArray` arguments
//! (via `Either<&MxArray, f64>`) and JS `number[]` for shapes (via `Vec<f64>`).
//! This eliminates ClojureScript-side `ensure-mx` and BigInt64Array conversion.

use mlx_core::array::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================================================
// Helpers
// ============================================================================

/// Convert JS number | MxArray to MxArray.
/// For references: clones the Arc handle (O(1)).
/// For numbers: creates a float32 scalar.
fn to_mx(value: Either<&MxArray, f64>) -> Result<MxArray> {
    match value {
        // `handle` is pub(crate) in mlx-core; MxArray: Clone is pub, so clone the
        // whole array (O(1) Arc clone) instead of reaching into the field.
        Either::A(arr) => Ok(arr.clone()),
        Either::B(num) => MxArray::scalar_float(num),
    }
}

/// Convert JS number[] to MLX shape (i64[]).
fn to_shape(shape: &[f64]) -> Vec<i64> {
    shape.iter().map(|&x| x as i64).collect()
}

/// Take (and clear) the last MLX exception message the C++ shim recorded on this
/// thread. Inlined here (W-B) over the pub `mlx_sys::mlx_take_last_error` FFI so
/// the mlx-core `take_last_native_error` re-export can revert to `pub(crate)`.
/// The shim records MLX throws (e.g. the Metal buffer limit) and returns a
/// null/false sentinel instead of aborting; this surfaces the detail so the
/// sentinel becomes a CATCHABLE napi error (bean genmlx-5ucd).
fn take_last_native_error() -> Option<String> {
    let p = unsafe { sys::mlx_take_last_error() };
    if p.is_null() {
        None
    } else {
        // Copy immediately — the pointer is only valid until the next shim call.
        Some(
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

/// Validate that all axes are within bounds for the array's dimensions.
/// Negative axes are normalized (axis + ndim) before checking bounds [0, ndim).
/// Copied from mlx-core reduction.rs so the reduction free fns below (which now
/// call the FFI directly) keep the same out-of-bounds guard the deleted methods
/// had — prevents an invalid axis reaching the C++ boundary as undefined behavior.
fn validate_axes(arr: &MxArray, axes: &[i32], context: &str) -> Result<()> {
    let ndim = arr.ndim()? as i32;
    for &axis in axes {
        let normalized = if axis < 0 { axis + ndim } else { axis };
        if normalized < 0 || normalized >= ndim {
            return Err(Error::from_reason(format!(
                "{}: axis {} is out of bounds for array with {} dimension{}",
                context,
                axis,
                ndim,
                if ndim == 1 { "" } else { "s" },
            )));
        }
    }
    Ok(())
}

// ============================================================================
// Unary ops: (MxArray | number) -> MxArray
// ============================================================================

// STOCK unary ops: delegate to stock mlx-core `MxArray::$name()` methods. These
// methods still live in mlx-core (they are upstream, not GenMLX additions), so
// these free fns keep delegating unchanged.
macro_rules! genmlx_unary {
    ($($name:ident),+ $(,)?) => {
        $(
            #[napi]
            pub fn $name(a: Either<&MxArray, f64>) -> Result<MxArray> {
                match a {
                    Either::A(arr) => arr.$name(),
                    Either::B(num) => MxArray::scalar_float(num)?.$name(),
                }
            }
        )+
    };
}

genmlx_unary!(
    exp, log, log2, log10,
    sin, cos, tan, arcsin, arccos, arctan,
    sinh, cosh, tanh,
    sqrt, square, abs, negative, sign, reciprocal,
    erf,
    floor, ceil, round,
    isnan, isinf, isfinite,
    logical_not,
);

// GenMLX-added unary ops: the matching mlx-core methods are being deleted (W-B),
// so these call the mlx-sys FFI directly. `$ffi` is the `sys::mlx_array_<op>`
// symbol; bodies are byte-identical to the former mlx-core method bodies.
macro_rules! genmlx_unary_ffi {
    ($($name:ident => $ffi:ident, $ctx:literal);+ $(;)?) => {
        $(
            #[napi]
            pub fn $name(a: Either<&MxArray, f64>) -> Result<MxArray> {
                let arr = to_mx(a)?;
                let handle = unsafe { sys::$ffi(arr.as_raw_ptr()) };
                MxArray::from_handle(handle, $ctx)
            }
        )+
    };
}

genmlx_unary_ffi!(
    expm1 => mlx_array_expm1, "expm1";
    sigmoid => mlx_array_sigmoid, "sigmoid";
    erfinv => mlx_array_erfinv, "erfinv";
    lgamma => mlx_array_lgamma, "lgamma";
    digamma => mlx_array_digamma, "digamma";
    flatten => mlx_array_flatten, "flatten";
);

// Ops with custom JS names

#[napi(js_name = "log1p")]
pub fn log1p(a: Either<&MxArray, f64>) -> Result<MxArray> {
    match a {
        Either::A(arr) => arr.log1p(),
        Either::B(num) => MxArray::scalar_float(num)?.log1p(),
    }
}

#[napi(js_name = "besselI0e")]
pub fn bessel_i0e(a: Either<&MxArray, f64>) -> Result<MxArray> {
    let arr = to_mx(a)?;
    let handle = unsafe { sys::mlx_array_bessel_i0e(arr.as_raw_ptr()) };
    MxArray::from_handle(handle, "bessel_i0e")
}

#[napi(js_name = "besselI1e")]
pub fn bessel_i1e(a: Either<&MxArray, f64>) -> Result<MxArray> {
    let arr = to_mx(a)?;
    let handle = unsafe { sys::mlx_array_bessel_i1e(arr.as_raw_ptr()) };
    MxArray::from_handle(handle, "bessel_i1e")
}

#[napi(js_name = "stopGradient")]
pub fn stop_gradient(a: &MxArray) -> Result<MxArray> {
    a.stop_gradient()
}

// ============================================================================
// Unary ops with extra parameters
// ============================================================================

#[napi]
pub fn softmax(a: Either<&MxArray, f64>, axis: Option<i32>) -> Result<MxArray> {
    let arr = to_mx(a)?;
    let handle = unsafe { sys::mlx_array_softmax(arr.as_raw_ptr(), axis.unwrap_or(-1)) };
    MxArray::from_handle(handle, "softmax")
}

#[napi(js_name = "logSoftmax")]
pub fn log_softmax(a: Either<&MxArray, f64>, axis: Option<i32>) -> Result<MxArray> {
    to_mx(a)?.log_softmax(axis.unwrap_or(-1))
}

#[napi]
pub fn clip(
    a: Either<&MxArray, f64>,
    minimum: Option<Either<&MxArray, f64>>,
    maximum: Option<Either<&MxArray, f64>>,
) -> Result<MxArray> {
    // Bounds accept MxArray or scalar (Either<&MxArray, f64>), matching the
    // coercion pattern used by every other genmlx.rs op. Array bounds are
    // applied via maximum/minimum (clip(a,lo,hi) = min(max(a,lo),hi)); MLX
    // broadcasting handles scalar bounds, so this also covers the f64 case.
    let mut arr = to_mx(a)?;
    if let Some(lo) = minimum {
        let lo_arr = to_mx(lo)?;
        arr = arr.maximum(&lo_arr)?;
    }
    if let Some(hi) = maximum {
        let hi_arr = to_mx(hi)?;
        arr = arr.minimum(&hi_arr)?;
    }
    Ok(arr)
}

#[napi(js_name = "nanToNum")]
pub fn nan_to_num(
    a: Either<&MxArray, f64>,
    nan_val: Option<f64>,
    posinf_val: Option<f64>,
    neginf_val: Option<f64>,
) -> Result<MxArray> {
    let arr = to_mx(a)?;
    let handle = unsafe {
        sys::mlx_array_nan_to_num(
            arr.as_raw_ptr(),
            nan_val.unwrap_or(0.0) as f32,
            posinf_val.is_some(),
            posinf_val.unwrap_or(0.0) as f32,
            neginf_val.is_some(),
            neginf_val.unwrap_or(0.0) as f32,
        )
    };
    MxArray::from_handle(handle, "nan_to_num")
}

#[napi]
pub fn diag(a: &MxArray, k: Option<i32>) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_array_diag(a.as_raw_ptr(), k.unwrap_or(0)) };
    MxArray::from_handle(handle, "diag")
}

#[napi(js_name = "trace")]
pub fn trace_(
    a: &MxArray,
    offset: Option<i32>,
    axis1: Option<i32>,
    axis2: Option<i32>,
) -> Result<MxArray> {
    let handle = unsafe {
        sys::mlx_array_trace(
            a.as_raw_ptr(),
            offset.unwrap_or(0),
            axis1.unwrap_or(0),
            axis2.unwrap_or(1),
        )
    };
    MxArray::from_handle(handle, "trace")
}

// ============================================================================
// Binary ops: (MxArray | number, MxArray | number) -> MxArray
// ============================================================================

// STOCK binary ops: delegate to stock mlx-core `MxArray::$name()` methods.
macro_rules! genmlx_binary {
    ($($name:ident),+ $(,)?) => {
        $(
            #[napi]
            pub fn $name(
                a: Either<&MxArray, f64>,
                b: Either<&MxArray, f64>,
            ) -> Result<MxArray> {
                let a_arr = to_mx(a)?;
                let b_arr = to_mx(b)?;
                a_arr.$name(&b_arr)
            }
        )+
    };
}

genmlx_binary!(
    add, sub, mul, div,
    power, maximum, minimum,
    floor_divide, remainder,
    matmul,
);

// GenMLX-added binary ops: matching mlx-core methods deleted (W-B) → direct FFI.
macro_rules! genmlx_binary_ffi {
    ($($name:ident => $ffi:ident, $ctx:literal);+ $(;)?) => {
        $(
            #[napi]
            pub fn $name(
                a: Either<&MxArray, f64>,
                b: Either<&MxArray, f64>,
            ) -> Result<MxArray> {
                let a_arr = to_mx(a)?;
                let b_arr = to_mx(b)?;
                let handle = unsafe { sys::$ffi(a_arr.as_raw_ptr(), b_arr.as_raw_ptr()) };
                MxArray::from_handle(handle, $ctx)
            }
        )+
    };
}

genmlx_binary_ffi!(
    logaddexp => mlx_array_logaddexp, "logaddexp";
    inner => mlx_array_inner, "inner";
    outer => mlx_array_outer, "outer";
);

// ============================================================================
// Comparison ops
// ============================================================================

genmlx_binary!(
    equal, not_equal,
    greater, greater_equal,
    less, less_equal,
    logical_and, logical_or,
);

/// Ternary selection: where_(condition, x, y).
#[napi]
pub fn where_(
    condition: &MxArray,
    x: Either<&MxArray, f64>,
    y: Either<&MxArray, f64>,
) -> Result<MxArray> {
    let x_arr = to_mx(x)?;
    let y_arr = to_mx(y)?;
    condition.where_(&x_arr, &y_arr)
}

// ============================================================================
// Reduction ops (accept number[] axes via Vec<i32>)
// ============================================================================

// STOCK reductions: delegate to stock mlx-core `MxArray::$name()` methods.
macro_rules! genmlx_reduce {
    ($($name:ident),+ $(,)?) => {
        $(
            #[napi]
            pub fn $name(
                a: Either<&MxArray, f64>,
                axes: Option<Vec<i32>>,
                keepdims: Option<bool>,
            ) -> Result<MxArray> {
                to_mx(a)?.$name(axes.as_deref(), keepdims)
            }
        )+
    };
}

genmlx_reduce!(sum, mean, prod, max, min, logsumexp);

// GenMLX-added reductions (`all`/`any`): matching mlx-core methods deleted
// (W-B) → direct FFI. Bodies are byte-identical to the former methods, including
// the validate_axes guard and the None=null-ptr path.
macro_rules! genmlx_reduce_ffi {
    ($($name:ident => $ffi:ident, $ctx:literal);+ $(;)?) => {
        $(
            #[napi]
            pub fn $name(
                a: Either<&MxArray, f64>,
                axes: Option<Vec<i32>>,
                keepdims: Option<bool>,
            ) -> Result<MxArray> {
                let arr = to_mx(a)?;
                let kd = keepdims.unwrap_or(false);
                let handle = match axes.as_deref() {
                    Some(ax) => {
                        validate_axes(&arr, ax, $ctx)?;
                        unsafe { sys::$ffi(arr.as_raw_ptr(), ax.as_ptr(), ax.len(), kd) }
                    }
                    None => unsafe { sys::$ffi(arr.as_raw_ptr(), std::ptr::null(), 0, kd) },
                };
                MxArray::from_handle(handle, $ctx)
            }
        )+
    };
}

genmlx_reduce_ffi!(
    all => mlx_array_all, "all";
    any => mlx_array_any, "any";
);

// var and std have an extra ddof parameter
#[napi]
pub fn var(
    a: Either<&MxArray, f64>,
    axes: Option<Vec<i32>>,
    keepdims: Option<bool>,
    ddof: Option<i32>,
) -> Result<MxArray> {
    to_mx(a)?.var(axes.as_deref(), keepdims, ddof)
}

#[napi]
pub fn std(
    a: Either<&MxArray, f64>,
    axes: Option<Vec<i32>>,
    keepdims: Option<bool>,
    ddof: Option<i32>,
) -> Result<MxArray> {
    to_mx(a)?.std(axes.as_deref(), keepdims, ddof)
}

#[napi]
pub fn cumsum(a: &MxArray, axis: Option<i32>) -> Result<MxArray> {
    a.cumsum(axis.unwrap_or(0))
}

#[napi]
pub fn cumprod(a: &MxArray, axis: Option<i32>) -> Result<MxArray> {
    a.cumprod(axis.unwrap_or(0))
}

#[napi]
pub fn logcumsumexp(a: &MxArray, axis: Option<i32>, reverse: Option<bool>) -> Result<MxArray> {
    let ax = axis.unwrap_or(0);
    validate_axes(a, &[ax], "logcumsumexp")?;
    let handle = unsafe { sys::mlx_array_logcumsumexp(a.as_raw_ptr(), ax, reverse.unwrap_or(false)) };
    MxArray::from_handle(handle, "logcumsumexp")
}

#[napi]
pub fn argmax(a: &MxArray, axis: Option<i32>, keepdims: Option<bool>) -> Result<MxArray> {
    a.argmax(axis.unwrap_or(0), keepdims)
}

#[napi]
pub fn argmin(a: &MxArray, axis: Option<i32>, keepdims: Option<bool>) -> Result<MxArray> {
    a.argmin(axis.unwrap_or(0), keepdims)
}

#[napi]
pub fn topk(a: &MxArray, k: i32, axis: Option<i32>) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_array_topk(a.as_raw_ptr(), k, axis.unwrap_or(-1)) };
    MxArray::from_handle(handle, "topk")
}

#[napi]
pub fn searchsorted(a: &MxArray, values: &MxArray, right: Option<bool>) -> Result<MxArray> {
    let handle = unsafe {
        sys::mlx_array_searchsorted(a.as_raw_ptr(), values.as_raw_ptr(), right.unwrap_or(false))
    };
    MxArray::from_handle(handle, "searchsorted")
}

// ============================================================================
// Shape ops (accept number[] shapes via Vec<f64>)
// ============================================================================

#[napi]
pub fn reshape(a: &MxArray, shape: Vec<f64>) -> Result<MxArray> {
    a.reshape(&to_shape(&shape))
}

#[napi]
pub fn transpose(a: &MxArray, axes: Option<Vec<i32>>) -> Result<MxArray> {
    a.transpose(axes.as_deref())
}

#[napi]
pub fn squeeze(a: &MxArray, axes: Option<Vec<i32>>) -> Result<MxArray> {
    a.squeeze(axes.as_deref())
}

#[napi(js_name = "expandDims")]
pub fn expand_dims(a: &MxArray, axis: i32) -> Result<MxArray> {
    a.expand_dims(axis)
}

#[napi(js_name = "broadcastTo")]
pub fn broadcast_to(a: &MxArray, shape: Vec<f64>) -> Result<MxArray> {
    a.broadcast_to(&to_shape(&shape))
}

#[napi]
pub fn take(a: &MxArray, indices: &MxArray, axis: i32) -> Result<MxArray> {
    a.take(indices, axis)
}

#[napi(js_name = "takeAlongAxis")]
pub fn take_along_axis(a: &MxArray, indices: &MxArray, axis: i32) -> Result<MxArray> {
    a.take_along_axis(indices, axis)
}

#[napi]
pub fn concatenate(arrays: Vec<&MxArray>, axis: Option<i32>) -> Result<MxArray> {
    MxArray::concatenate_many(arrays, axis)
}

#[napi]
pub fn stack(arrays: Vec<&MxArray>, axis: Option<i32>) -> Result<MxArray> {
    MxArray::stack(arrays, axis)
}

#[napi]
pub fn split(a: &MxArray, sections: i32, axis: Option<i32>) -> Result<Vec<MxArray>> {
    a.split(sections, axis)
}

#[napi]
pub fn tile(a: &MxArray, reps: Vec<i32>) -> Result<MxArray> {
    a.tile(&reps)
}

#[napi]
pub fn repeat(a: &MxArray, repeats: i32, axis: i32) -> Result<MxArray> {
    a.repeat(repeats, axis)
}

#[napi]
pub fn slice(a: &MxArray, starts: Vec<f64>, stops: Vec<f64>) -> Result<MxArray> {
    a.slice(&to_shape(&starts), &to_shape(&stops))
}

#[napi]
pub fn sort(a: &MxArray, axis: Option<i32>) -> Result<MxArray> {
    a.sort(axis)
}

#[napi]
pub fn argsort(a: &MxArray, axis: Option<i32>) -> Result<MxArray> {
    a.argsort(axis)
}

#[napi]
pub fn pad(a: &MxArray, pad_width: Vec<i32>, constant_value: f64) -> Result<MxArray> {
    a.pad(&pad_width, constant_value)
}

#[napi]
pub fn roll(a: &MxArray, shift: i32, axis: i32) -> Result<MxArray> {
    a.roll(shift, axis)
}

// ============================================================================
// Creation ops (accept number[] shapes via Vec<f64>)
// ============================================================================

#[napi]
pub fn zeros(shape: Vec<f64>, dtype: Option<DType>) -> Result<MxArray> {
    MxArray::zeros(&to_shape(&shape), dtype)
}

#[napi]
pub fn ones(shape: Vec<f64>, dtype: Option<DType>) -> Result<MxArray> {
    MxArray::ones(&to_shape(&shape), dtype)
}

#[napi]
pub fn full(
    shape: Vec<f64>,
    fill_value: Either<f64, &MxArray>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    MxArray::full(&to_shape(&shape), fill_value, dtype)
}

#[napi]
pub fn eye(n: i32, m: Option<i32>, k: Option<i32>, dtype: Option<DType>) -> Result<MxArray> {
    MxArray::eye(n, m, k, dtype)
}

#[napi]
pub fn arange(
    start: f64,
    stop: f64,
    step: Option<f64>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    MxArray::arange(start, stop, step, dtype)
}

#[napi]
pub fn linspace(
    start: f64,
    stop: f64,
    num: Option<i32>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    MxArray::linspace(start, stop, num, dtype)
}

/// Create a float32 scalar array.
#[napi]
pub fn scalar(value: f64) -> Result<MxArray> {
    MxArray::scalar_float(value)
}

/// Create an int32 scalar array.
#[napi(js_name = "scalarInt")]
pub fn scalar_int(value: i32) -> Result<MxArray> {
    MxArray::scalar_int(value)
}

/// Create array from Float32Array data with number[] shape.
#[napi(js_name = "fromFloat32")]
pub fn from_float32(data: &[f32], shape: Vec<f64>) -> Result<MxArray> {
    MxArray::from_float32(data, &to_shape(&shape))
}

/// Create array from Int32Array data with number[] shape.
#[napi(js_name = "fromInt32")]
pub fn from_int32(data: &[i32], shape: Vec<f64>) -> Result<MxArray> {
    MxArray::from_int32(data, &to_shape(&shape))
}

// ============================================================================
// Linear algebra
// ============================================================================

// All linalg ops below are GenMLX-added; matching mlx-core methods deleted (W-B).
// Bodies call mlx-sys `mlx_linalg_*` FFI directly, byte-identical to the former
// methods (with `self.handle.0` → `a.as_raw_ptr()`).

#[napi]
pub fn cholesky(a: &MxArray, upper: Option<bool>) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_linalg_cholesky(a.as_raw_ptr(), upper.unwrap_or(false)) };
    MxArray::from_handle(handle, "cholesky")
}

#[napi(js_name = "linalgSolve")]
pub fn linalg_solve(a: &MxArray, b: &MxArray) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_linalg_solve(a.as_raw_ptr(), b.as_raw_ptr()) };
    MxArray::from_handle(handle, "solve")
}

#[napi(js_name = "solveTriangular")]
pub fn solve_triangular(a: &MxArray, b: &MxArray, upper: Option<bool>) -> Result<MxArray> {
    let handle = unsafe {
        sys::mlx_linalg_solve_triangular(a.as_raw_ptr(), b.as_raw_ptr(), upper.unwrap_or(false))
    };
    MxArray::from_handle(handle, "solve_triangular")
}

#[napi(js_name = "linalgInv")]
pub fn linalg_inv(a: &MxArray) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_linalg_inv(a.as_raw_ptr()) };
    MxArray::from_handle(handle, "inv")
}

#[napi(js_name = "triInv")]
pub fn tri_inv(a: &MxArray, upper: Option<bool>) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_linalg_tri_inv(a.as_raw_ptr(), upper.unwrap_or(false)) };
    MxArray::from_handle(handle, "tri_inv")
}

#[napi(js_name = "choleskyInv")]
pub fn cholesky_inv(a: &MxArray, upper: Option<bool>) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_linalg_cholesky_inv(a.as_raw_ptr(), upper.unwrap_or(false)) };
    MxArray::from_handle(handle, "cholesky_inv")
}

/// QR decomposition. Returns [Q, R] as a two-element array.
#[napi]
pub fn qr(a: &MxArray) -> Result<Vec<MxArray>> {
    let mut q_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    let mut r_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    unsafe { sys::mlx_linalg_qr(a.as_raw_ptr(), &mut q_ptr, &mut r_ptr) };
    let q = MxArray::from_handle(q_ptr, "qr_q")?;
    let r = MxArray::from_handle(r_ptr, "qr_r")?;
    Ok(vec![q, r])
}

/// SVD decomposition. Returns [U, S, Vt] as a three-element array.
#[napi]
pub fn svd(a: &MxArray) -> Result<Vec<MxArray>> {
    let mut u_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    let mut s_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    let mut vt_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    unsafe { sys::mlx_linalg_svd(a.as_raw_ptr(), &mut u_ptr, &mut s_ptr, &mut vt_ptr) };
    let u = MxArray::from_handle(u_ptr, "svd_u")?;
    let s = MxArray::from_handle(s_ptr, "svd_s")?;
    let vt = MxArray::from_handle(vt_ptr, "svd_vt")?;
    Ok(vec![u, s, vt])
}

/// Eigendecomposition of symmetric matrix. Returns [eigenvalues, eigenvectors].
#[napi]
pub fn eigh(a: &MxArray, uplo: Option<String>) -> Result<Vec<MxArray>> {
    let uplo_str = uplo.unwrap_or_else(|| "L".to_string());
    let c_str = std::ffi::CString::new(uplo_str)
        .map_err(|e| napi::Error::from_reason(format!("Invalid UPLO: {e}")))?;
    let mut eigvals_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    let mut eigvecs_ptr: *mut sys::mlx_array = std::ptr::null_mut();
    unsafe {
        sys::mlx_linalg_eigh(
            a.as_raw_ptr(),
            c_str.as_ptr(),
            &mut eigvals_ptr,
            &mut eigvecs_ptr,
        )
    };
    let eigvals = MxArray::from_handle(eigvals_ptr, "eigh_eigvals")?;
    let eigvecs = MxArray::from_handle(eigvecs_ptr, "eigh_eigvecs")?;
    Ok(vec![eigvals, eigvecs])
}

#[napi]
pub fn eigvalsh(a: &MxArray, uplo: Option<String>) -> Result<MxArray> {
    let uplo_str = uplo.unwrap_or_else(|| "L".to_string());
    let c_str = std::ffi::CString::new(uplo_str)
        .map_err(|e| napi::Error::from_reason(format!("Invalid UPLO: {e}")))?;
    let handle = unsafe { sys::mlx_linalg_eigvalsh(a.as_raw_ptr(), c_str.as_ptr()) };
    MxArray::from_handle(handle, "eigvalsh")
}

#[napi(js_name = "linalgNorm")]
pub fn linalg_norm(a: &MxArray, ord: Option<f64>) -> Result<MxArray> {
    let handle = match ord {
        Some(o) => unsafe { sys::mlx_linalg_norm(a.as_raw_ptr(), o) },
        None => unsafe { sys::mlx_linalg_norm_default(a.as_raw_ptr()) },
    };
    MxArray::from_handle(handle, "norm")
}

#[napi]
pub fn einsum(subscripts: String, operands: Vec<&MxArray>) -> Result<MxArray> {
    let c_str = std::ffi::CString::new(subscripts)
        .map_err(|e| napi::Error::from_reason(format!("Invalid subscripts: {e}")))?;
    let handles: Vec<*mut sys::mlx_array> = operands.iter().map(|a| a.as_raw_ptr()).collect();
    let handle =
        unsafe { sys::mlx_array_einsum(c_str.as_ptr(), handles.as_ptr(), handles.len()) };
    MxArray::from_handle(handle, "einsum")
}

// ============================================================================
// Data accessors
// ============================================================================

/// Fused eval + scalar extraction. One NAPI call instead of three
/// (eval + toFloat32 + aget). Handles all dtypes via CPU-side cast.
#[napi]
pub fn item(a: &MxArray) -> Result<f64> {
    let mut value: f64 = 0.0;
    let ok = unsafe { sys::mlx_array_item_f64(a.as_raw_ptr(), &mut value) };
    if ok {
        Ok(value)
    } else {
        // false = either a non-size-1 array, or a guarded MLX throw (e.g. the
        // Metal buffer limit) — surface the real detail if the shim recorded one
        // so it is catchable rather than an abort (bean genmlx-5ucd).
        let msg = match take_last_native_error() {
            Some(detail) => format!("MLX error in item: {}", detail),
            None => "item: array must have size 1".to_string(),
        };
        Err(Error::from_reason(msg))
    }
}

#[napi]
pub fn astype(a: &MxArray, dtype: DType) -> Result<MxArray> {
    a.astype(dtype)
}

#[napi(js_name = "shapeOf")]
pub fn shape_of(a: &MxArray) -> Result<Vec<i32>> {
    // Inlined from the former mlx-core `shape_array` method (deleted in W-B).
    let ndim = unsafe { sys::mlx_array_ndim(a.as_raw_ptr()) };
    let mut shape = vec![0i64; ndim];
    unsafe { sys::mlx_array_shape(a.as_raw_ptr(), shape.as_mut_ptr()) };
    Ok(shape.into_iter().map(|x| x as i32).collect())
}

#[napi(js_name = "ndimOf")]
pub fn ndim_of(a: &MxArray) -> Result<u32> {
    a.ndim()
}

#[napi(js_name = "dtypeOf")]
pub fn dtype_of(a: &MxArray) -> Result<DType> {
    a.dtype()
}

#[napi(js_name = "sizeOf")]
pub fn size_of(a: &MxArray) -> Result<u64> {
    a.size()
}

// ============================================================================
// Random ops (key-based, module-level)
// ============================================================================

// Keyed-PRNG: the FFI (mlx_sys::mlx_random_*_key) is inlined here directly so the
// matching MxArray methods can be deleted from stock mlx-core. `dt as i32` equals
// the old `dt.code()` (DType is #[repr(i32)] with matching discriminants). Bodies
// are otherwise byte-identical to the former mlx-core methods.

#[napi(js_name = "randomKey")]
pub fn random_key(seed: f64) -> Result<MxArray> {
    let handle = unsafe { mlx_sys::mlx_random_key(seed as u64) };
    MxArray::from_handle(handle, "random_key")
}

/// Seed MLX's PROCESS-GLOBAL RNG (`mlx::core::random::seed`).
///
/// GenMLX's inference PRNG is keyed (`randomKey`/`randomSplit` + `key*`
/// samplers) and is NOT affected by this. The global stream is what the
/// native training engine's sampler consumes during GRPO generation, so
/// seeding it makes paired training runs share their sampling randomness
/// (common-random-numbers experiments — genmlx-at2q).
#[napi(js_name = "seedGlobalRng")]
pub fn seed_global_rng(seed: f64) -> Result<()> {
    unsafe { mlx_sys::mlx_seed(seed as u64) };
    Ok(())
}

#[napi(js_name = "randomSplit")]
pub fn random_split(key: &MxArray) -> Result<Vec<MxArray>> {
    let mut k1: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut k2: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    unsafe { mlx_sys::mlx_random_split(key.as_raw_ptr(), &mut k1, &mut k2) };
    let a = MxArray::from_handle(k1, "split_k1")?;
    let b = MxArray::from_handle(k2, "split_k2")?;
    Ok(vec![a, b])
}

#[napi(js_name = "randomSplitN")]
pub fn random_split_n(key: &MxArray, n: i32) -> Result<MxArray> {
    let handle = unsafe { mlx_sys::mlx_random_split_n(key.as_raw_ptr(), n) };
    MxArray::from_handle(handle, "split_n")
}

#[napi(js_name = "keyUniform")]
pub fn key_uniform(
    key: &MxArray,
    shape: Vec<f64>,
    low: Option<f64>,
    high: Option<f64>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    let dt = dtype.unwrap_or(DType::Float32);
    let shp = to_shape(&shape);
    let handle = unsafe {
        mlx_sys::mlx_random_uniform_key(
            key.as_raw_ptr(),
            shp.as_ptr(),
            shp.len(),
            low.unwrap_or(0.0) as f32,
            high.unwrap_or(1.0) as f32,
            dt as i32,
        )
    };
    MxArray::from_handle(handle, "key_uniform")
}

#[napi(js_name = "keyNormal")]
pub fn key_normal(key: &MxArray, shape: Vec<f64>, dtype: Option<DType>) -> Result<MxArray> {
    let dt = dtype.unwrap_or(DType::Float32);
    let shp = to_shape(&shape);
    let handle = unsafe {
        mlx_sys::mlx_random_normal_key(key.as_raw_ptr(), shp.as_ptr(), shp.len(), dt as i32)
    };
    MxArray::from_handle(handle, "key_normal")
}

#[napi(js_name = "keyBernoulli")]
pub fn key_bernoulli(key: &MxArray, prob: f64, shape: Vec<f64>) -> Result<MxArray> {
    let shp = to_shape(&shape);
    let handle = unsafe {
        mlx_sys::mlx_random_bernoulli_key(key.as_raw_ptr(), prob as f32, shp.as_ptr(), shp.len())
    };
    MxArray::from_handle(handle, "key_bernoulli")
}

#[napi(js_name = "keyCategorical")]
pub fn key_categorical(key: &MxArray, logits: &MxArray, axis: Option<i32>) -> Result<MxArray> {
    let handle = unsafe {
        mlx_sys::mlx_random_categorical_key(
            key.as_raw_ptr(),
            logits.as_raw_ptr(),
            axis.unwrap_or(-1),
        )
    };
    MxArray::from_handle(handle, "key_categorical")
}

#[napi(js_name = "keyRandint")]
pub fn key_randint(
    key: &MxArray,
    low: i32,
    high: i32,
    shape: Vec<f64>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    let dt = dtype.unwrap_or(DType::Int32);
    let shp = to_shape(&shape);
    let handle = unsafe {
        mlx_sys::mlx_random_randint_key(
            key.as_raw_ptr(),
            low,
            high,
            shp.as_ptr(),
            shp.len(),
            dt as i32,
        )
    };
    MxArray::from_handle(handle, "key_randint")
}

#[napi(js_name = "keyGumbel")]
pub fn key_gumbel(key: &MxArray, shape: Vec<f64>, dtype: Option<DType>) -> Result<MxArray> {
    let dt = dtype.unwrap_or(DType::Float32);
    let shp = to_shape(&shape);
    let handle = unsafe {
        mlx_sys::mlx_random_gumbel_key(key.as_raw_ptr(), shp.as_ptr(), shp.len(), dt as i32)
    };
    MxArray::from_handle(handle, "key_gumbel")
}

#[napi(js_name = "keyLaplace")]
pub fn key_laplace(key: &MxArray, shape: Vec<f64>, dtype: Option<DType>) -> Result<MxArray> {
    let dt = dtype.unwrap_or(DType::Float32);
    let shp = to_shape(&shape);
    let handle = unsafe {
        mlx_sys::mlx_random_laplace_key(key.as_raw_ptr(), shp.as_ptr(), shp.len(), dt as i32)
    };
    MxArray::from_handle(handle, "key_laplace")
}

#[napi(js_name = "keyTruncatedNormal")]
pub fn key_truncated_normal(
    key: &MxArray,
    lower: &MxArray,
    upper: &MxArray,
    shape: Vec<f64>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    let dt = dtype.unwrap_or(DType::Float32);
    let shp = to_shape(&shape);
    let handle = unsafe {
        mlx_sys::mlx_random_truncated_normal_key(
            key.as_raw_ptr(),
            lower.as_raw_ptr(),
            upper.as_raw_ptr(),
            shp.as_ptr(),
            shp.len(),
            dt as i32,
        )
    };
    MxArray::from_handle(handle, "key_truncated_normal")
}

#[napi(js_name = "keyMultivariateNormal")]
pub fn key_multivariate_normal(
    key: &MxArray,
    mean: &MxArray,
    cov: &MxArray,
    shape: Vec<f64>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    let dt = dtype.unwrap_or(DType::Float32);
    let shp = to_shape(&shape);
    let handle = unsafe {
        mlx_sys::mlx_random_multivariate_normal_key(
            key.as_raw_ptr(),
            mean.as_raw_ptr(),
            cov.as_raw_ptr(),
            shp.as_ptr(),
            shp.len(),
            dt as i32,
        )
    };
    MxArray::from_handle(handle, "key_multivariate_normal")
}

// ============================================================================
// Fused NN primitives (f6ov: GenMLX-owned LLM forward pass)
//
// These delegate to the STABLE MLX fast:: / nn ops (mlx-sys mlx_fast_*) that
// upstream model refactors never touch, so a GenMLX-owned CLJS forward composed
// over them is decoupled from upstream's per-model forward structs. SDPA takes
// an EXPLICIT mask array (not the "causal" string mode, which null-ptrs across
// MLX builds — genmlx-7siy); pass null/None for no mask.
// ============================================================================

/// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight (variance in f32 for stability).
#[napi(js_name = "rmsNorm")]
pub fn rms_norm(x: &MxArray, weight: &MxArray, eps: f64) -> Result<MxArray> {
    mlx_core::utils::functional::rms_norm_functional(x, weight, eps)
}

/// Rotary position embedding (fast::rope). offset = KV-cache position.
#[napi]
pub fn rope(
    x: &MxArray,
    dims: i32,
    traditional: bool,
    base: f64,
    scale: f64,
    offset: i32,
) -> Result<MxArray> {
    mlx_core::nn::RoPE::new(dims, Some(traditional), Some(base), Some(scale)).forward(x, Some(offset))
}

/// Scaled dot-product attention with an EXPLICIT optional mask array.
/// q/k/v: [batch, n_heads, seq, head_dim]; scale = 1/sqrt(head_dim).
#[napi(js_name = "scaledDotProductAttention")]
pub fn scaled_dot_product_attention(
    queries: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    scale: f64,
    mask: Option<&MxArray>,
) -> Result<MxArray> {
    mlx_core::array::scaled_dot_product_attention(queries, keys, values, scale, mask)
}

/// SiLU / swish activation: x * sigmoid(x) (dtype-preserving).
#[napi]
pub fn silu(x: &MxArray) -> Result<MxArray> {
    mlx_core::nn::Activations::silu(x)
}

/// Load all tensors from a .safetensors file as a {name -> MxArray} map (f6ov
/// P1: GenMLX-owned weight loading, decoupled from upstream's model structs).
/// Lazy — tensors are graph leaves, materialized on first eval.
#[napi(js_name = "loadSafetensors")]
pub fn load_safetensors(path: String) -> Result<std::collections::HashMap<String, MxArray>> {
    mlx_core::utils::safetensors::load_safetensors_lazy(&path)
}

// ============================================================================
// Autograd (relocated from mlx-core into genmlx-core, W-B)
//
// These were `MxArray.valueAndGrad` / `MxArray.computeGradients` static methods
// in mlx-core; the orphan rule forbids re-`impl MxArray` outside mlx-core, so
// here they are module-level free fns wrapping the stock-pub generic
// `mlx_core::autograd::{value_and_grad, compute_gradients}`. The JS-closure →
// MxArray recipe is the one already proven in `transforms.rs` (the same
// into_instance / napi_unwrap dance). The JS surface moves from the MxArray
// class to top-level addon exports — bodies and FFI are otherwise unchanged.
// ============================================================================

/// Build a loss closure that calls a JS function synchronously via raw NAPI.
///
/// The JS function receives MxArray arguments and must return a scalar MxArray.
/// All calls are synchronous on the JS thread — safe because MLX's
/// value_and_grad / compute_gradients are synchronous.
fn make_js_loss_closure(
    raw_env: napi::sys::napi_env,
    raw_func: napi::sys::napi_value,
) -> impl FnMut(&[MxArray]) -> Result<MxArray> {
    move |params: &[MxArray]| -> Result<MxArray> {
        unsafe {
            let env_wrapper = Env::from_raw(raw_env);

            // Convert each MxArray param to a JS MxArray instance. Clone the Arc
            // (O(1)) — both Rust and JS hold references; C++ owns the underlying
            // handle, the Arc prevents premature Rust-side cleanup mid-callback.
            let mut js_args: Vec<napi::sys::napi_value> = Vec::with_capacity(params.len());
            for param in params {
                let cloned = param.clone();
                let instance = cloned.into_instance(&env_wrapper)?;
                js_args.push(instance.raw());
            }

            // Call the JS function synchronously (this = global).
            let mut result: napi::sys::napi_value = std::ptr::null_mut();
            let mut global: napi::sys::napi_value = std::ptr::null_mut();
            napi::sys::napi_get_global(raw_env, &mut global);
            let status = napi::sys::napi_call_function(
                raw_env,
                global,
                raw_func,
                js_args.len(),
                if js_args.is_empty() {
                    std::ptr::null()
                } else {
                    js_args.as_ptr()
                },
                &mut result,
            );

            if status != napi::sys::Status::napi_ok || result.is_null() {
                return Err(Error::from_reason("JS loss function call failed"));
            }

            // Extract MxArray from the returned napi_value (NAPI-RS #[napi]
            // classes are unwrapped via napi_unwrap). Clone the Arc so we own it.
            let mut wrapped: *mut std::ffi::c_void = std::ptr::null_mut();
            let unwrap_status = napi::sys::napi_unwrap(raw_env, result, &mut wrapped);
            if unwrap_status != napi::sys::Status::napi_ok || wrapped.is_null() {
                return Err(Error::from_reason(
                    "JS loss function must return an MxArray",
                ));
            }
            let loss_ref = &*(wrapped as *const MxArray);
            Ok(loss_ref.clone())
        }
    }
}

/// Compute value and gradients of a JS loss function.
///
/// The loss function receives MxArray arguments and must return a scalar MxArray.
/// Returns `[lossValue, grad0, grad1, ...]`.
///
/// ```js
/// const [loss, dx, dy] = valueAndGrad((x, y) => x.mul(y).sum(), [x, y]);
/// ```
#[napi(js_name = "valueAndGrad")]
pub fn value_and_grad(
    env: Env,
    #[napi(ts_arg_type = "(...args: MxArray[]) => MxArray")]
    loss_fn: napi::bindgen_prelude::Function<'static>,
    inputs: Vec<&MxArray>,
) -> Result<Vec<MxArray>> {
    if inputs.is_empty() {
        return Err(Error::from_reason("valueAndGrad: inputs cannot be empty"));
    }
    let raw_env = env.raw();
    let raw_func =
        unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(raw_env, loss_fn)? };
    let loss_closure = make_js_loss_closure(raw_env, raw_func);
    let input_refs: Vec<&MxArray> = inputs.iter().copied().collect();
    let (loss, grads) = mlx_core::autograd::value_and_grad(input_refs, loss_closure)?;
    let mut result = Vec::with_capacity(1 + grads.len());
    result.push(loss);
    result.extend(grads);
    Ok(result)
}

/// Compute only gradients (not loss value) of a JS function.
/// Returns `[grad0, grad1, ...]`.
#[napi(js_name = "computeGradients")]
pub fn compute_gradients(
    env: Env,
    #[napi(ts_arg_type = "(...args: MxArray[]) => MxArray")]
    loss_fn: napi::bindgen_prelude::Function<'static>,
    inputs: Vec<&MxArray>,
) -> Result<Vec<MxArray>> {
    if inputs.is_empty() {
        return Err(Error::from_reason("computeGradients: inputs cannot be empty"));
    }
    let raw_env = env.raw();
    let raw_func =
        unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(raw_env, loss_fn)? };
    let loss_closure = make_js_loss_closure(raw_env, raw_func);
    let input_refs: Vec<&MxArray> = inputs.iter().copied().collect();
    mlx_core::autograd::compute_gradients(input_refs, loss_closure)
}
