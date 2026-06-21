use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Take (and clear) the last MLX exception message the C++ shim recorded on this
/// thread. The shim catches MLX throws (e.g. "[metal::malloc] Resource limit
/// (499000) exceeded") and returns a null/false sentinel instead of letting the
/// exception abort the process; this surfaces the detail so the sentinel can
/// become a CATCHABLE napi error (bean genmlx-5ucd).
pub(crate) fn take_last_native_error() -> Option<String> {
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

pub(crate) fn check_handle(
    handle: *mut sys::mlx_array,
    context: &str,
) -> Result<*mut sys::mlx_array> {
    if handle.is_null() {
        let msg = match take_last_native_error() {
            Some(detail) => format!("MLX error in {}: {}", context, detail),
            None => format!("null handle returned: {}", context),
        };
        Err(Error::from_reason(msg))
    } else {
        Ok(handle)
    }
}

/// Internal handle wrapper that owns the MLX C++ array pointer
/// and ensures proper cleanup via Drop
pub(crate) struct MxHandle(pub(crate) *mut sys::mlx_array);

unsafe impl Send for MxHandle {}
unsafe impl Sync for MxHandle {}

impl MxHandle {
    /// Overwrite this handle's pointer with a new one, cleaning up the old pointer.
    /// This matches mlx-lm's `overwrite_descriptor` pattern for zero-allocation cache updates.
    ///
    /// # Safety
    /// The new_handle must be a valid MLX array pointer that this MxHandle will now own.
    pub(crate) unsafe fn overwrite(&mut self, new_handle: *mut sys::mlx_array) {
        // Delete old array if not null
        // SAFETY: caller guarantees self.0 is a valid array pointer (or null)
        // and new_handle is a valid array pointer
        unsafe {
            if !self.0.is_null() {
                sys::mlx_array_delete(self.0);
            }
            // Take ownership of new array
            self.0 = new_handle;
        }
    }
}

impl Drop for MxHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { sys::mlx_array_delete(self.0) };
        }
    }
}
