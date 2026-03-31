//! GenMLX consolidation: vmap and compile NAPI exports.
//!
//! Both take JS functions and apply MLX transforms via synchronous callbacks.

use crate::array::MxArray;
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
                    let cloned = MxArray {
                        handle: arr.handle.clone(),
                    };
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
        let out_handle = result_ref.handle.0;
        // Clone the Arc so we don't steal ownership from JS
        let cloned = MxArray {
            handle: result_ref.handle.clone(),
        };
        let handle = cloned.handle.0;
        std::mem::forget(cloned); // C++ will manage this handle
        handle
    }
}

#[napi]
impl MxArray {
    /// Apply vmap to a JS function.
    ///
    /// ```js
    /// const result = MxArray.vmap(
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
            inputs.iter().map(|a| a.handle.0).collect();

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
                    let cloned = MxArray {
                        handle: arr.handle.clone(),
                    };
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

        // Extract MxArray from result
        let mut wrapped: *mut c_void = std::ptr::null_mut();
        if napi::sys::napi_unwrap(ctx.env, result, &mut wrapped)
            != napi::sys::Status::napi_ok
        {
            ctx.error = Some("compile function must return MxArray".to_string());
            return 0;
        }
        let result_ref = &*(wrapped as *const MxArray);
        let cloned = MxArray {
            handle: result_ref.handle.clone(),
        };
        let output_slice = std::slice::from_raw_parts_mut(outputs, max_outputs);
        output_slice[0] = cloned.handle.0;
        std::mem::forget(cloned);
        1 // one output
    }
}

#[napi]
impl MxArray {
    /// Compile and apply a JS function.
    ///
    /// ```js
    /// const result = MxArray.compileFn(
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
            inputs.iter().map(|a| a.handle.0).collect();

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
}
