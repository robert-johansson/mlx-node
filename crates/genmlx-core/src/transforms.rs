//! vmap and compile NAPI exports (relocated from mlx-core into genmlx-core).
//!
//! Both take JS functions and apply MLX transforms via synchronous callbacks.
//! These were `MxArray` static methods in mlx-core; the orphan rule forbids an
//! inherent `impl MxArray` outside mlx-core, so here they are module-level free
//! functions. The JS surface changes from `MxArray.vmap` / `MxArray.compileFn`
//! to top-level `vmap` / `compileFn` addon exports — the bodies and the
//! `sys::mlx_vmap_apply` / `sys::mlx_compile_apply` FFI calls are byte-identical,
//! so there is no compute-path or GPU change.

use mlx_core::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::ffi::c_void;

// ============================================================================
// Vmap
// ============================================================================

struct VmapContext {
    env: napi::sys::napi_env,
    func: napi::sys::napi_value,
    error: Option<String>,
}

extern "C-unwind" fn vmap_callback(
    inputs: *const *mut sys::mlx_array,
    input_count: usize,
    context: *mut c_void,
) -> *mut sys::mlx_array {
    let ctx = unsafe { &mut *(context as *mut VmapContext) };
    if ctx.error.is_some() {
        return std::ptr::null_mut();
    }

    unsafe {
        let env = Env::from_raw(ctx.env);
        let input_slice = std::slice::from_raw_parts(inputs, input_count);

        // Convert handles to JS MxArray instances
        let mut js_args: Vec<napi::sys::napi_value> = Vec::with_capacity(input_count);
        for &handle in input_slice {
            match MxArray::from_handle(handle, "vmap_input") {
                Ok(arr) => {
                    let cloned = arr.clone();
                    std::mem::forget(arr); // C++ owns the handle
                    match cloned.into_instance(&env) {
                        Ok(instance) => js_args.push(instance.raw()),
                        Err(e) => {
                            ctx.error = Some(format!("{:?}", e));
                            return std::ptr::null_mut();
                        }
                    }
                }
                Err(e) => {
                    ctx.error = Some(format!("{:?}", e));
                    return std::ptr::null_mut();
                }
            }
        }

        // Call JS function
        let mut result: napi::sys::napi_value = std::ptr::null_mut();
        let mut global: napi::sys::napi_value = std::ptr::null_mut();
        napi::sys::napi_get_global(ctx.env, &mut global);
        let status = napi::sys::napi_call_function(
            ctx.env,
            global,
            ctx.func,
            js_args.len(),
            if js_args.is_empty() {
                std::ptr::null()
            } else {
                js_args.as_ptr()
            },
            &mut result,
        );
        if status != napi::sys::Status::napi_ok || result.is_null() {
            ctx.error = Some("JS vmap function call failed".to_string());
            return std::ptr::null_mut();
        }

        // Extract MxArray from result
        let mut wrapped: *mut c_void = std::ptr::null_mut();
        if napi::sys::napi_unwrap(ctx.env, result, &mut wrapped)
            != napi::sys::Status::napi_ok
        {
            ctx.error = Some("vmap function must return MxArray".to_string());
            return std::ptr::null_mut();
        }
        let result_ref = &*(wrapped as *const MxArray);
        // Clone the Arc so we don't steal ownership from JS
        let cloned = result_ref.clone();
        let handle = cloned.as_raw_ptr();
        std::mem::forget(cloned); // C++ will manage this handle
        handle
    }
}

/// Apply vmap to a JS function.
///
/// ```js
/// const result = vmap(
///   (x) => x.square(),  // function to vectorize
///   [x_batched],         // batched inputs
///   [0],                 // in_axes
///   [0]                  // out_axes
/// );
/// ```
#[napi]
pub fn vmap(
    env: Env,
    #[napi(ts_arg_type = "(...args: MxArray[]) => MxArray")]
    func: napi::bindgen_prelude::Function<'static>,
    inputs: Vec<&MxArray>,
    in_axes: Option<Vec<i32>>,
    out_axes: Option<Vec<i32>>,
) -> Result<Vec<MxArray>> {
    if inputs.is_empty() {
        return Err(Error::from_reason("vmap: inputs cannot be empty"));
    }

    let raw_env = env.raw();
    let raw_func = unsafe {
        napi::bindgen_prelude::ToNapiValue::to_napi_value(raw_env, func)?
    };

    let in_ax = in_axes.unwrap_or_default();
    let out_ax = out_axes.unwrap_or_default();

    let input_handles: Vec<*mut sys::mlx_array> =
        inputs.iter().map(|a| a.as_raw_ptr()).collect();

    let mut ctx = Box::new(VmapContext {
        env: raw_env,
        func: raw_func,
        error: None,
    });
    let ctx_ptr = &mut *ctx as *mut VmapContext as *mut c_void;

    const MAX_OUTPUTS: usize = 16;
    let mut output_handles: Vec<*mut sys::mlx_array> =
        vec![std::ptr::null_mut(); MAX_OUTPUTS];
    let mut num_outputs: usize = 0;

    unsafe {
        sys::mlx_vmap_apply(
            vmap_callback,
            ctx_ptr,
            input_handles.as_ptr(),
            input_handles.len(),
            in_ax.as_ptr(),
            in_ax.len(),
            out_ax.as_ptr(),
            out_ax.len(),
            output_handles.as_mut_ptr(),
            MAX_OUTPUTS,
            &mut num_outputs,
        );
    }

    if let Some(error) = &ctx.error {
        return Err(Error::from_reason(format!("vmap: {}", error)));
    }

    let results: Result<Vec<MxArray>> = output_handles[..num_outputs]
        .iter()
        .map(|&h| MxArray::from_handle(h, "vmap_output"))
        .collect();
    results
}

// ============================================================================
// Compile
// ============================================================================

struct CompileContext {
    env: napi::sys::napi_env,
    func: napi::sys::napi_value,
    error: Option<String>,
}

extern "C-unwind" fn compile_callback(
    inputs: *const *mut sys::mlx_array,
    input_count: usize,
    outputs: *mut *mut sys::mlx_array,
    max_outputs: usize,
    context: *mut c_void,
) -> usize {
    let ctx = unsafe { &mut *(context as *mut CompileContext) };
    if ctx.error.is_some() {
        return 0;
    }

    unsafe {
        let env = Env::from_raw(ctx.env);
        let input_slice = std::slice::from_raw_parts(inputs, input_count);

        let mut js_args: Vec<napi::sys::napi_value> = Vec::with_capacity(input_count);
        for &handle in input_slice {
            match MxArray::from_handle(handle, "compile_input") {
                Ok(arr) => {
                    let cloned = arr.clone();
                    std::mem::forget(arr);
                    match cloned.into_instance(&env) {
                        Ok(instance) => js_args.push(instance.raw()),
                        Err(e) => {
                            ctx.error = Some(format!("{:?}", e));
                            return 0;
                        }
                    }
                }
                Err(e) => {
                    ctx.error = Some(format!("{:?}", e));
                    return 0;
                }
            }
        }

        let mut result: napi::sys::napi_value = std::ptr::null_mut();
        let mut global: napi::sys::napi_value = std::ptr::null_mut();
        napi::sys::napi_get_global(ctx.env, &mut global);
        let status = napi::sys::napi_call_function(
            ctx.env,
            global,
            ctx.func,
            js_args.len(),
            if js_args.is_empty() {
                std::ptr::null()
            } else {
                js_args.as_ptr()
            },
            &mut result,
        );
        if status != napi::sys::Status::napi_ok || result.is_null() {
            ctx.error = Some("JS compile function call failed".to_string());
            return 0;
        }

        // Extract MxArray(s) from result — handles both single and multi-output.
        let output_slice = std::slice::from_raw_parts_mut(outputs, max_outputs);

        // Check if result is a JS array (multi-output)
        let mut is_array = false;
        napi::sys::napi_is_array(ctx.env, result, &mut is_array);

        if is_array {
            // Multi-output: extract each element
            let mut length: u32 = 0;
            napi::sys::napi_get_array_length(ctx.env, result, &mut length);
            let count = (length as usize).min(max_outputs);
            for i in 0..count {
                let mut element: napi::sys::napi_value = std::ptr::null_mut();
                napi::sys::napi_get_element(ctx.env, result, i as u32, &mut element);
                let mut wrapped: *mut c_void = std::ptr::null_mut();
                if napi::sys::napi_unwrap(ctx.env, element, &mut wrapped)
                    != napi::sys::Status::napi_ok
                {
                    ctx.error = Some(format!(
                        "compile: output[{}] is not MxArray", i
                    ));
                    return 0;
                }
                let elem_ref = &*(wrapped as *const MxArray);
                let cloned = elem_ref.clone();
                output_slice[i] = cloned.as_raw_ptr();
                std::mem::forget(cloned);
            }
            count
        } else {
            // Single output: unwrap directly
            let mut wrapped: *mut c_void = std::ptr::null_mut();
            if napi::sys::napi_unwrap(ctx.env, result, &mut wrapped)
                != napi::sys::Status::napi_ok
            {
                ctx.error = Some("compile function must return MxArray".to_string());
                return 0;
            }
            let result_ref = &*(wrapped as *const MxArray);
            let cloned = result_ref.clone();
            output_slice[0] = cloned.as_raw_ptr();
            std::mem::forget(cloned);
            1
        }
    }
}

/// Compile and apply a JS function.
///
/// ```js
/// const result = compileFn(
///   (x) => x.square().sum(),
///   [x],
///   false  // shapeless
/// );
/// ```
#[napi(js_name = "compileFn")]
pub fn compile_fn(
    env: Env,
    #[napi(ts_arg_type = "(...args: MxArray[]) => MxArray")]
    func: napi::bindgen_prelude::Function<'static>,
    inputs: Vec<&MxArray>,
    shapeless: Option<bool>,
) -> Result<Vec<MxArray>> {
    if inputs.is_empty() {
        return Err(Error::from_reason("compileFn: inputs cannot be empty"));
    }

    let raw_env = env.raw();
    let raw_func = unsafe {
        napi::bindgen_prelude::ToNapiValue::to_napi_value(raw_env, func)?
    };

    let input_handles: Vec<*mut sys::mlx_array> =
        inputs.iter().map(|a| a.as_raw_ptr()).collect();

    let mut ctx = Box::new(CompileContext {
        env: raw_env,
        func: raw_func,
        error: None,
    });
    let ctx_ptr = &mut *ctx as *mut CompileContext as *mut c_void;

    const MAX_OUTPUTS: usize = 16;
    let mut output_handles: Vec<*mut sys::mlx_array> =
        vec![std::ptr::null_mut(); MAX_OUTPUTS];

    let num_outputs = unsafe {
        sys::mlx_compile_apply(
            compile_callback,
            ctx_ptr,
            input_handles.as_ptr(),
            input_handles.len(),
            shapeless.unwrap_or(false),
            output_handles.as_mut_ptr(),
            MAX_OUTPUTS,
        )
    };

    if let Some(error) = &ctx.error {
        return Err(Error::from_reason(format!("compileFn: {}", error)));
    }

    if num_outputs == 0 {
        return Err(Error::from_reason("compileFn: returned 0 outputs"));
    }

    let results: Result<Vec<MxArray>> = output_handles[..num_outputs]
        .iter()
        .map(|&h| MxArray::from_handle(h, "compile_output"))
        .collect();
    results
}

// ============================================================================
// Persistent compile (genmlx-z2gt Phase 1): trace once, replay in C++.
//
// `compileFn` above creates a fresh closure identity per call, so
// mlx::core::compile re-traces every invocation. The trio below holds the
// compiled closure in a native handle: `compileCreate` builds it (no trace
// yet), the first `compiledCall` per input-shape traces by calling the JS
// builder synchronously, and every later call replays the cached graph
// entirely in C++. The JS builder is kept alive via a persistent napi
// reference because shape-change retraces call back into it; it is released
// by `compiledFree`. Main-thread synchronous use only.
// ============================================================================

pub struct CompiledFnHandle {
    handle: *mut c_void, // mlx_compiled_fn* (owned; freed on dispose/Drop)
    func_ref: napi::sys::napi_ref,
    ctx: Box<CompileContext>, // stable address — the C++ closure captures it
    disposed: bool,
}

impl Drop for CompiledFnHandle {
    fn drop(&mut self) {
        // GC-finalizer path: free the native closure. The napi_ref cannot be
        // deleted here (no env in Drop) — explicit compiledFree releases it;
        // a leaked handle leaks one JS function reference, nothing native.
        if !self.disposed && !self.handle.is_null() {
            unsafe { sys::mlx_compiled_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Create a persistent compiled function handle.
///
/// ```js
/// const cf = compileCreate((x) => x.square().sum(), false);
/// const [r1] = compiledCall(cf, [a]);   // traces + compiles + runs
/// const [r2] = compiledCall(cf, [b]);   // replays the cached graph
/// compiledFree(cf);
/// ```
#[napi(js_name = "compileCreate")]
pub fn compile_create(
    env: Env,
    #[napi(ts_arg_type = "(...args: MxArray[]) => MxArray | MxArray[]")]
    func: napi::bindgen_prelude::Function<'static>,
    shapeless: Option<bool>,
) -> Result<External<CompiledFnHandle>> {
    let raw_env = env.raw();
    let raw_func = unsafe {
        napi::bindgen_prelude::ToNapiValue::to_napi_value(raw_env, func)?
    };

    let mut func_ref: napi::sys::napi_ref = std::ptr::null_mut();
    let status =
        unsafe { napi::sys::napi_create_reference(raw_env, raw_func, 1, &mut func_ref) };
    if status != napi::sys::Status::napi_ok {
        return Err(Error::from_reason("compileCreate: could not reference function"));
    }

    // env/func are refreshed at every compiledCall before the C++ side may
    // trace; the Box gives the C++ closure a stable context address.
    let mut ctx = Box::new(CompileContext {
        env: raw_env,
        func: std::ptr::null_mut(),
        error: None,
    });
    let ctx_ptr = &mut *ctx as *mut CompileContext as *mut c_void;

    let handle = unsafe {
        sys::mlx_compile_create(compile_callback, ctx_ptr, shapeless.unwrap_or(false))
    };
    if handle.is_null() {
        unsafe { napi::sys::napi_delete_reference(raw_env, func_ref) };
        return Err(Error::from_reason("compileCreate: native create failed"));
    }

    Ok(External::new(CompiledFnHandle {
        handle,
        func_ref,
        ctx,
        disposed: false,
    }))
}

/// Call a persistent compiled function. First call per input-shape traces
/// (invoking the JS builder); later calls replay the cached graph in C++.
#[napi(js_name = "compiledCall")]
pub fn compiled_call(
    env: Env,
    handle: &mut External<CompiledFnHandle>,
    inputs: Vec<&MxArray>,
) -> Result<Vec<MxArray>> {
    let h = handle.as_mut();
    if h.disposed {
        return Err(Error::from_reason("compiledCall: handle already freed"));
    }
    if inputs.is_empty() {
        return Err(Error::from_reason("compiledCall: inputs cannot be empty"));
    }

    let raw_env = env.raw();
    let mut func_val: napi::sys::napi_value = std::ptr::null_mut();
    let status =
        unsafe { napi::sys::napi_get_reference_value(raw_env, h.func_ref, &mut func_val) };
    if status != napi::sys::Status::napi_ok || func_val.is_null() {
        return Err(Error::from_reason("compiledCall: builder function is gone"));
    }
    h.ctx.env = raw_env;
    h.ctx.func = func_val;
    h.ctx.error = None;

    let input_handles: Vec<*mut sys::mlx_array> =
        inputs.iter().map(|a| a.as_raw_ptr()).collect();

    const MAX_OUTPUTS: usize = 16;
    let mut output_handles: Vec<*mut sys::mlx_array> =
        vec![std::ptr::null_mut(); MAX_OUTPUTS];

    let num_outputs = unsafe {
        sys::mlx_compiled_call(
            h.handle,
            input_handles.as_ptr(),
            input_handles.len(),
            output_handles.as_mut_ptr(),
            MAX_OUTPUTS,
        )
    };

    if let Some(error) = &h.ctx.error {
        return Err(Error::from_reason(format!("compiledCall: {}", error)));
    }
    if num_outputs == 0 {
        return Err(Error::from_reason("compiledCall: returned 0 outputs"));
    }

    output_handles[..num_outputs]
        .iter()
        .map(|&hd| MxArray::from_handle(hd, "compiled_call_output"))
        .collect()
}

/// Captured-replay call (genmlx-7prh): like `compiledCall`, but the outputs
/// come back EVALUATED, and after the first successful capture every later
/// call is launch-only in the CUDA backend (retained graph execs + retained
/// buffers — no tape clone, no per-op scheduler walk). On Metal, on CPU-only
/// builds, or when the tape cannot be captured (a CPU-stream op, a graph
/// fallback), the handle permanently falls back to trace-cache replay +
/// eval — behaviorally identical, just slower. `compiledIsCaptured` reports
/// which path a handle is on.
#[napi(js_name = "compiledCallCaptured")]
pub fn compiled_call_captured(
    env: Env,
    handle: &mut External<CompiledFnHandle>,
    inputs: Vec<&MxArray>,
) -> Result<Vec<MxArray>> {
    let h = handle.as_mut();
    if h.disposed {
        return Err(Error::from_reason("compiledCallCaptured: handle already freed"));
    }
    if inputs.is_empty() {
        return Err(Error::from_reason("compiledCallCaptured: inputs cannot be empty"));
    }

    let raw_env = env.raw();
    let mut func_val: napi::sys::napi_value = std::ptr::null_mut();
    let status =
        unsafe { napi::sys::napi_get_reference_value(raw_env, h.func_ref, &mut func_val) };
    if status != napi::sys::Status::napi_ok || func_val.is_null() {
        return Err(Error::from_reason("compiledCallCaptured: builder function is gone"));
    }
    h.ctx.env = raw_env;
    h.ctx.func = func_val;
    h.ctx.error = None;

    let input_handles: Vec<*mut sys::mlx_array> =
        inputs.iter().map(|a| a.as_raw_ptr()).collect();

    const MAX_OUTPUTS: usize = 16;
    let mut output_handles: Vec<*mut sys::mlx_array> =
        vec![std::ptr::null_mut(); MAX_OUTPUTS];

    let num_outputs = unsafe {
        sys::mlx_compiled_call_captured(
            h.handle,
            input_handles.as_ptr(),
            input_handles.len(),
            output_handles.as_mut_ptr(),
            MAX_OUTPUTS,
        )
    };

    if let Some(error) = &h.ctx.error {
        return Err(Error::from_reason(format!("compiledCallCaptured: {}", error)));
    }
    if num_outputs == 0 {
        return Err(Error::from_reason("compiledCallCaptured: returned 0 outputs"));
    }

    output_handles[..num_outputs]
        .iter()
        .map(|&hd| MxArray::from_handle(hd, "compiled_call_captured_output"))
        .collect()
}

/// True when the handle completed a replay capture and its
/// `compiledCallCaptured` calls are launch-only (the honesty probe — no
/// silent path ambiguity in tests or benches).
#[napi(js_name = "compiledIsCaptured")]
pub fn compiled_is_captured(handle: &External<CompiledFnHandle>) -> bool {
    let h = handle.as_ref();
    if h.disposed || h.handle.is_null() {
        return false;
    }
    unsafe { sys::mlx_compiled_is_captured(h.handle) }
}

/// Free a persistent compiled function: destroys the cached graph and
/// releases the JS builder reference. Idempotent; returns false if already
/// freed.
#[napi(js_name = "compiledFree")]
pub fn compiled_free(env: Env, handle: &mut External<CompiledFnHandle>) -> Result<bool> {
    let h = handle.as_mut();
    if h.disposed {
        return Ok(false);
    }
    unsafe { sys::mlx_compiled_free(h.handle) };
    h.handle = std::ptr::null_mut();
    h.disposed = true;
    unsafe { napi::sys::napi_delete_reference(env.raw(), h.func_ref) };
    h.func_ref = std::ptr::null_mut();
    Ok(true)
}
