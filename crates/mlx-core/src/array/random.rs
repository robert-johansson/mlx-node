use super::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
impl MxArray {
    #[napi]
    pub fn random_uniform(
        shape: &[i64],
        low: f64,
        high: f64,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_array_random_uniform(
                shape.as_ptr(),
                shape.len(),
                low as f32,
                high as f32,
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "random_uniform")
    }

    #[napi]
    pub fn random_normal(shape: &[i64], mean: f64, std: f64, dtype: Option<DType>) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_array_random_normal(
                shape.as_ptr(),
                shape.len(),
                mean as f32,
                std as f32,
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "random_normal")
    }

    #[napi]
    pub fn random_bernoulli(shape: &[i64], prob: f64) -> Result<Self> {
        let handle =
            unsafe { sys::mlx_array_random_bernoulli(shape.as_ptr(), shape.len(), prob as f32) };
        MxArray::from_handle(handle, "random_bernoulli")
    }

    #[napi]
    pub fn randint(shape: &[i64], low: i32, high: i32) -> Result<Self> {
        let handle = unsafe { sys::mlx_array_randint(shape.as_ptr(), shape.len(), low, high) };
        MxArray::from_handle(handle, "randint")
    }

    /// Sample from categorical distribution
    /// Takes logits and returns sampled indices
    #[napi]
    pub fn categorical(&self, axis: Option<i32>) -> Result<MxArray> {
        let axis_val = axis.unwrap_or(-1);
        let handle = unsafe { sys::mlx_array_categorical(self.handle.0, axis_val) };
        MxArray::from_handle(handle, "categorical")
    }

    // ================================================================
    // GenMLX consolidation: key-based functional PRNG
    // ================================================================

    /// Create a PRNG key from a seed.
    #[napi(js_name = "randomKey")]
    pub fn random_key(seed: i64) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_random_key(seed as u64) };
        MxArray::from_handle(handle, "random_key")
    }

    /// Split a key into two independent sub-keys. Returns [k1, k2].
    #[napi(js_name = "randomSplit")]
    pub fn random_split(&self) -> Result<Vec<MxArray>> {
        let mut k1: *mut sys::mlx_array = std::ptr::null_mut();
        let mut k2: *mut sys::mlx_array = std::ptr::null_mut();
        unsafe { sys::mlx_random_split(self.handle.0, &mut k1, &mut k2) };
        let a = MxArray::from_handle(k1, "split_k1")?;
        let b = MxArray::from_handle(k2, "split_k2")?;
        Ok(vec![a, b])
    }

    /// Split a key into n independent sub-keys.
    #[napi(js_name = "randomSplitN")]
    pub fn random_split_n(&self, n: i32) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_random_split_n(self.handle.0, n) };
        MxArray::from_handle(handle, "split_n")
    }

    /// Key-based uniform sampling.
    #[napi(js_name = "keyUniform")]
    pub fn key_uniform(
        &self,
        shape: &[i64],
        low: Option<f64>,
        high: Option<f64>,
        dtype: Option<DType>,
    ) -> Result<MxArray> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_random_uniform_key(
                self.handle.0,
                shape.as_ptr(),
                shape.len(),
                low.unwrap_or(0.0) as f32,
                high.unwrap_or(1.0) as f32,
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "key_uniform")
    }

    /// Key-based normal sampling.
    #[napi(js_name = "keyNormal")]
    pub fn key_normal(
        &self,
        shape: &[i64],
        dtype: Option<DType>,
    ) -> Result<MxArray> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_random_normal_key(
                self.handle.0,
                shape.as_ptr(),
                shape.len(),
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "key_normal")
    }

    /// Key-based bernoulli sampling.
    #[napi(js_name = "keyBernoulli")]
    pub fn key_bernoulli(
        &self,
        prob: f64,
        shape: &[i64],
    ) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_random_bernoulli_key(
                self.handle.0,
                prob as f32,
                shape.as_ptr(),
                shape.len(),
            )
        };
        MxArray::from_handle(handle, "key_bernoulli")
    }

    /// Key-based categorical sampling from logits.
    #[napi(js_name = "keyCategorical")]
    pub fn key_categorical(
        &self,
        logits: &MxArray,
        axis: Option<i32>,
    ) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_random_categorical_key(
                self.handle.0,
                logits.handle.0,
                axis.unwrap_or(-1),
            )
        };
        MxArray::from_handle(handle, "key_categorical")
    }

    /// Key-based random integer sampling.
    #[napi(js_name = "keyRandint")]
    pub fn key_randint(
        &self,
        low: i32,
        high: i32,
        shape: &[i64],
        dtype: Option<DType>,
    ) -> Result<MxArray> {
        let dt = dtype.unwrap_or(DType::Int32);
        let handle = unsafe {
            sys::mlx_random_randint_key(
                self.handle.0,
                low,
                high,
                shape.as_ptr(),
                shape.len(),
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "key_randint")
    }

    /// Key-based Gumbel sampling.
    #[napi(js_name = "keyGumbel")]
    pub fn key_gumbel(
        &self,
        shape: &[i64],
        dtype: Option<DType>,
    ) -> Result<MxArray> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_random_gumbel_key(
                self.handle.0,
                shape.as_ptr(),
                shape.len(),
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "key_gumbel")
    }

    /// Key-based Laplace sampling.
    #[napi(js_name = "keyLaplace")]
    pub fn key_laplace(
        &self,
        shape: &[i64],
        dtype: Option<DType>,
    ) -> Result<MxArray> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_random_laplace_key(
                self.handle.0,
                shape.as_ptr(),
                shape.len(),
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "key_laplace")
    }

    /// Key-based truncated normal sampling.
    #[napi(js_name = "keyTruncatedNormal")]
    pub fn key_truncated_normal(
        &self,
        lower: &MxArray,
        upper: &MxArray,
        shape: &[i64],
        dtype: Option<DType>,
    ) -> Result<MxArray> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_random_truncated_normal_key(
                self.handle.0,
                lower.handle.0,
                upper.handle.0,
                shape.as_ptr(),
                shape.len(),
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "key_truncated_normal")
    }

    /// Key-based multivariate normal sampling.
    #[napi(js_name = "keyMultivariateNormal")]
    pub fn key_multivariate_normal(
        &self,
        mean: &MxArray,
        cov: &MxArray,
        shape: &[i64],
        dtype: Option<DType>,
    ) -> Result<MxArray> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_random_multivariate_normal_key(
                self.handle.0,
                mean.handle.0,
                cov.handle.0,
                shape.as_ptr(),
                shape.len(),
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "key_multivariate_normal")
    }
}
