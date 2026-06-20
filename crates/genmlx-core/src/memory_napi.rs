//! Memory management, device, and utility NAPI exports (relocated from mlx-core
//! into genmlx-core).
//!
//! These wrap existing mlx-core / mlx-sys functions with #[napi] attributes to
//! expose them to JavaScript. No new C++ FFI needed — all ops exist already.
//! Pure relocation: every function body and FFI call is byte-identical to the
//! mlx-core original (only `crate::` -> `mlx_core::` and field access via the
//! public `as_raw_ptr()`).

use mlx_core::array::MxArray;
use mlx_sys as sys;
use napi_derive::napi;

// ============================================================================
// Memory management (Phase 9)
// ============================================================================

#[napi(js_name = "clearCache")]
pub fn clear_cache() {
    mlx_core::array::memory::clear_cache();
}

#[napi]
pub fn synchronize() {
    mlx_core::array::memory::synchronize();
}

#[napi(js_name = "getActiveMemory")]
pub fn get_active_memory() -> f64 {
    mlx_core::array::memory::get_active_memory()
}

#[napi(js_name = "getCacheMemory")]
pub fn get_cache_memory() -> f64 {
    mlx_core::array::memory::get_cache_memory()
}

#[napi(js_name = "getPeakMemory")]
pub fn get_peak_memory() -> f64 {
    mlx_core::array::memory::get_peak_memory()
}

/// Live Metal buffer allocation count (active + cached). Counts toward the
/// macOS resource limit (~499000); the membrane's Layer-2 proactive sweep
/// reads this to reclaim dead buffers before the limit is hit.
#[napi(js_name = "getNumResources")]
pub fn get_num_resources() -> f64 {
    mlx_core::array::memory::get_num_resources()
}

/// The Metal buffer resource limit (count at which allocations fail).
#[napi(js_name = "getResourceLimit")]
pub fn get_resource_limit() -> f64 {
    mlx_core::array::memory::get_resource_limit()
}

#[napi(js_name = "resetPeakMemory")]
pub fn reset_peak_memory() {
    mlx_core::array::memory::reset_peak_memory();
}

#[napi(js_name = "setMemoryLimit")]
pub fn set_memory_limit(limit: f64) -> f64 {
    mlx_core::array::memory::set_memory_limit(limit)
}

#[napi(js_name = "getMemoryLimit")]
pub fn get_memory_limit() -> f64 {
    mlx_core::array::memory::get_memory_limit()
}

#[napi(js_name = "setCacheLimit")]
pub fn set_cache_limit(limit: f64) -> f64 {
    mlx_core::array::memory::set_cache_limit(limit).unwrap_or(0.0)
}

#[napi(js_name = "setWiredLimit")]
pub fn set_wired_limit(limit: f64) -> f64 {
    let mut prev: u64 = 0;
    let rc = unsafe { sys::mlx_set_wired_limit(limit as u64, &mut prev) };
    if rc != 0 { 0.0 } else { prev as f64 }
}

#[napi(js_name = "getWiredLimit")]
pub fn get_wired_limit() -> f64 {
    let mut v: u64 = 0;
    let rc = unsafe { sys::mlx_get_wired_limit(&mut v) };
    if rc != 0 { 0.0 } else { v as f64 }
}

#[napi(js_name = "compileClearCache")]
pub fn compile_clear_cache() -> bool {
    mlx_core::array::memory::compile_clear_cache()
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
        arrays.iter().map(|a| a.as_raw_ptr()).collect();
    unsafe {
        sys::mlx_eval(handles.as_mut_ptr(), handles.len());
    }
}
