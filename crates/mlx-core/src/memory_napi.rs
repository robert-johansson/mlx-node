//! GenMLX consolidation: Memory management, device, and utility NAPI exports.
//!
//! These wrap existing Rust functions with #[napi] attributes to expose
//! them to JavaScript. No new C++ FFI needed — all ops exist already.

use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================================================
// Memory management (Phase 9)
// ============================================================================

#[napi(js_name = "clearCache")]
pub fn clear_cache() {
    crate::array::memory::clear_cache();
}

#[napi]
pub fn synchronize() {
    crate::array::memory::synchronize();
}

#[napi(js_name = "getActiveMemory")]
pub fn get_active_memory() -> f64 {
    crate::array::memory::get_active_memory()
}

#[napi(js_name = "getCacheMemory")]
pub fn get_cache_memory() -> f64 {
    crate::array::memory::get_cache_memory()
}

#[napi(js_name = "getPeakMemory")]
pub fn get_peak_memory() -> f64 {
    crate::array::memory::get_peak_memory()
}

#[napi(js_name = "resetPeakMemory")]
pub fn reset_peak_memory() {
    crate::array::memory::reset_peak_memory();
}

#[napi(js_name = "setMemoryLimit")]
pub fn set_memory_limit(limit: f64) -> f64 {
    crate::array::memory::set_memory_limit(limit)
}

#[napi(js_name = "getMemoryLimit")]
pub fn get_memory_limit() -> f64 {
    crate::array::memory::get_memory_limit()
}

#[napi(js_name = "setCacheLimit")]
pub fn set_cache_limit(limit: f64) -> f64 {
    crate::array::memory::set_cache_limit(limit)
}

#[napi(js_name = "setWiredLimit")]
pub fn set_wired_limit(limit: f64) -> f64 {
    unsafe { sys::mlx_set_wired_limit(limit as usize) as f64 }
}

#[napi(js_name = "getWiredLimit")]
pub fn get_wired_limit() -> f64 {
    unsafe { sys::mlx_get_wired_limit() as f64 }
}

#[napi(js_name = "compileClearCache")]
pub fn compile_clear_cache() -> bool {
    crate::array::memory::compile_clear_cache()
}

// ============================================================================
// Metal / Device (Phase 10)
// ============================================================================

#[napi(js_name = "metalIsAvailable")]
pub fn metal_is_available() -> bool {
    unsafe { sys::mlx_metal_is_available() }
}

#[napi(js_name = "metalDeviceInfo")]
pub fn metal_device_info() -> String {
    let ptr = unsafe { sys::mlx_metal_device_info() };
    if ptr.is_null() {
        return "{}".to_string();
    }
    unsafe {
        std::ffi::CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned()
    }
}

#[napi(js_name = "gpuArchitectureGen")]
pub fn gpu_architecture_gen() -> i32 {
    unsafe { sys::mlx_gpu_architecture_gen() }
}

// ============================================================================
// Eval helpers
// ============================================================================

/// Evaluate multiple arrays synchronously.
#[napi(js_name = "evalArrays")]
pub fn eval_arrays(arrays: Vec<&MxArray>) {
    if arrays.is_empty() {
        return;
    }
    let mut handles: Vec<*mut sys::mlx_array> =
        arrays.iter().map(|a| a.handle.0).collect();
    unsafe {
        sys::mlx_eval(handles.as_mut_ptr(), handles.len());
    }
}
