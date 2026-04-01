//! GenMLX module-level NAPI exports.
//!
//! Provides standalone functions that accept JS `number | MxArray` arguments
//! (via `Either<&MxArray, f64>`) and JS `number[]` for shapes (via `Vec<f64>`).
//! This eliminates ClojureScript-side `ensure-mx` and BigInt64Array conversion.

use crate::array::{DType, MxArray};
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
        Either::A(arr) => Ok(MxArray {
            handle: arr.handle.clone(),
        }),
        Either::B(num) => MxArray::scalar_float(num),
    }
}

/// Convert JS number[] to MLX shape (i64[]).
fn to_shape(shape: &[f64]) -> Vec<i64> {
    shape.iter().map(|&x| x as i64).collect()
}

// ============================================================================
// Unary ops: (MxArray | number) -> MxArray
// ============================================================================

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
    exp, log, log2, log10, expm1,
    sin, cos, tan, arcsin, arccos, arctan,
    sinh, cosh, tanh,
    sqrt, square, abs, negative, sign, reciprocal,
    sigmoid, erf, erfinv, lgamma, digamma,
    floor, ceil, round,
    flatten,
    isnan, isinf, isfinite,
    logical_not,
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
    match a {
        Either::A(arr) => arr.bessel_i0e(),
        Either::B(num) => MxArray::scalar_float(num)?.bessel_i0e(),
    }
}

#[napi(js_name = "besselI1e")]
pub fn bessel_i1e(a: Either<&MxArray, f64>) -> Result<MxArray> {
    match a {
        Either::A(arr) => arr.bessel_i1e(),
        Either::B(num) => MxArray::scalar_float(num)?.bessel_i1e(),
    }
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
    to_mx(a)?.softmax(axis.unwrap_or(-1))
}

#[napi(js_name = "logSoftmax")]
pub fn log_softmax(a: Either<&MxArray, f64>, axis: Option<i32>) -> Result<MxArray> {
    to_mx(a)?.log_softmax(axis.unwrap_or(-1))
}

#[napi]
pub fn clip(
    a: Either<&MxArray, f64>,
    minimum: Option<f64>,
    maximum: Option<f64>,
) -> Result<MxArray> {
    to_mx(a)?.clip(minimum, maximum)
}

#[napi(js_name = "nanToNum")]
pub fn nan_to_num(
    a: Either<&MxArray, f64>,
    nan_val: Option<f64>,
    posinf_val: Option<f64>,
    neginf_val: Option<f64>,
) -> Result<MxArray> {
    to_mx(a)?.nan_to_num(nan_val, posinf_val, neginf_val)
}

#[napi]
pub fn diag(a: &MxArray, k: Option<i32>) -> Result<MxArray> {
    a.diag(k)
}

#[napi(js_name = "trace")]
pub fn trace_(
    a: &MxArray,
    offset: Option<i32>,
    axis1: Option<i32>,
    axis2: Option<i32>,
) -> Result<MxArray> {
    a.trace_(offset, axis1, axis2)
}

// ============================================================================
// Binary ops: (MxArray | number, MxArray | number) -> MxArray
// ============================================================================

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
    floor_divide, remainder, logaddexp,
    matmul, inner, outer,
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

genmlx_reduce!(sum, mean, prod, max, min, all, any, logsumexp);

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
    a.logcumsumexp(axis.unwrap_or(0), reverse)
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
    a.topk(k, axis)
}

#[napi]
pub fn searchsorted(a: &MxArray, values: &MxArray, right: Option<bool>) -> Result<MxArray> {
    a.searchsorted(values, right)
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

#[napi]
pub fn cholesky(a: &MxArray, upper: Option<bool>) -> Result<MxArray> {
    a.cholesky(upper)
}

#[napi(js_name = "linalgSolve")]
pub fn linalg_solve(a: &MxArray, b: &MxArray) -> Result<MxArray> {
    a.linalg_solve(b)
}

#[napi(js_name = "solveTriangular")]
pub fn solve_triangular(a: &MxArray, b: &MxArray, upper: Option<bool>) -> Result<MxArray> {
    a.solve_triangular(b, upper)
}

#[napi(js_name = "linalgInv")]
pub fn linalg_inv(a: &MxArray) -> Result<MxArray> {
    a.linalg_inv()
}

#[napi(js_name = "triInv")]
pub fn tri_inv(a: &MxArray, upper: Option<bool>) -> Result<MxArray> {
    a.tri_inv(upper)
}

#[napi(js_name = "choleskyInv")]
pub fn cholesky_inv(a: &MxArray, upper: Option<bool>) -> Result<MxArray> {
    a.cholesky_inv(upper)
}

#[napi]
pub fn qr(a: &MxArray) -> Result<Vec<MxArray>> {
    a.qr()
}

#[napi]
pub fn svd(a: &MxArray) -> Result<Vec<MxArray>> {
    a.svd()
}

#[napi]
pub fn eigh(a: &MxArray, uplo: Option<String>) -> Result<Vec<MxArray>> {
    a.eigh(uplo)
}

#[napi]
pub fn eigvalsh(a: &MxArray, uplo: Option<String>) -> Result<MxArray> {
    a.eigvalsh(uplo)
}

#[napi(js_name = "linalgNorm")]
pub fn linalg_norm(a: &MxArray, ord: Option<f64>) -> Result<MxArray> {
    a.linalg_norm(ord)
}

#[napi]
pub fn einsum(subscripts: String, operands: Vec<&MxArray>) -> Result<MxArray> {
    MxArray::einsum(subscripts, operands)
}

// ============================================================================
// Data accessors
// ============================================================================

/// Fused eval + scalar extraction. One NAPI call instead of three
/// (eval + toFloat32 + aget). Handles all dtypes via CPU-side cast.
#[napi]
pub fn item(a: &MxArray) -> Result<f64> {
    let mut value: f64 = 0.0;
    let ok = unsafe { mlx_sys::mlx_array_item_f64(a.handle.0, &mut value) };
    if ok {
        Ok(value)
    } else {
        Err(Error::from_reason("item: array must have size 1"))
    }
}

#[napi]
pub fn astype(a: &MxArray, dtype: DType) -> Result<MxArray> {
    a.astype(dtype)
}

#[napi(js_name = "shapeOf")]
pub fn shape_of(a: &MxArray) -> Result<Vec<i32>> {
    a.shape_array()
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

#[napi(js_name = "randomKey")]
pub fn random_key(seed: f64) -> Result<MxArray> {
    MxArray::random_key(seed as i64)
}

#[napi(js_name = "randomSplit")]
pub fn random_split(key: &MxArray) -> Result<Vec<MxArray>> {
    key.random_split()
}

#[napi(js_name = "randomSplitN")]
pub fn random_split_n(key: &MxArray, n: i32) -> Result<MxArray> {
    key.random_split_n(n)
}

#[napi(js_name = "keyUniform")]
pub fn key_uniform(
    key: &MxArray,
    shape: Vec<f64>,
    low: Option<f64>,
    high: Option<f64>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    key.key_uniform(&to_shape(&shape), low, high, dtype)
}

#[napi(js_name = "keyNormal")]
pub fn key_normal(key: &MxArray, shape: Vec<f64>, dtype: Option<DType>) -> Result<MxArray> {
    key.key_normal(&to_shape(&shape), dtype)
}

#[napi(js_name = "keyBernoulli")]
pub fn key_bernoulli(key: &MxArray, prob: f64, shape: Vec<f64>) -> Result<MxArray> {
    key.key_bernoulli(prob, &to_shape(&shape))
}

#[napi(js_name = "keyCategorical")]
pub fn key_categorical(key: &MxArray, logits: &MxArray, axis: Option<i32>) -> Result<MxArray> {
    key.key_categorical(logits, axis)
}

#[napi(js_name = "keyRandint")]
pub fn key_randint(
    key: &MxArray,
    low: i32,
    high: i32,
    shape: Vec<f64>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    key.key_randint(low, high, &to_shape(&shape), dtype)
}

#[napi(js_name = "keyGumbel")]
pub fn key_gumbel(key: &MxArray, shape: Vec<f64>, dtype: Option<DType>) -> Result<MxArray> {
    key.key_gumbel(&to_shape(&shape), dtype)
}

#[napi(js_name = "keyLaplace")]
pub fn key_laplace(key: &MxArray, shape: Vec<f64>, dtype: Option<DType>) -> Result<MxArray> {
    key.key_laplace(&to_shape(&shape), dtype)
}

#[napi(js_name = "keyTruncatedNormal")]
pub fn key_truncated_normal(
    key: &MxArray,
    lower: &MxArray,
    upper: &MxArray,
    shape: Vec<f64>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    key.key_truncated_normal(lower, upper, &to_shape(&shape), dtype)
}

#[napi(js_name = "keyMultivariateNormal")]
pub fn key_multivariate_normal(
    key: &MxArray,
    mean: &MxArray,
    cov: &MxArray,
    shape: Vec<f64>,
    dtype: Option<DType>,
) -> Result<MxArray> {
    key.key_multivariate_normal(mean, cov, &to_shape(&shape), dtype)
}
