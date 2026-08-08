//! Double-instance diagnostics for the native addon (genmlx-vqn0).
//!
//! Requiring the addon through two different resolved paths (symlink vs
//! realpath, two package copies) dlopens two independent copies of this
//! library into one process: separate Rust statics, separate allocator
//! bookkeeping — `memoryStats()` on the second copy reads 0 while the
//! first is mid-training, and a `clearCache()` drains an empty duplicate
//! context. The same confusion arises across PROCESSES (a Bun worker
//! pool next to a Node trainer): stats are per-process by nature.
//!
//! Two aids:
//!  * a `#[module_init]` sentinel that detects a second in-process copy
//!    at load time and warns loudly with both file paths;
//!  * `nativeInstanceInfo()` exposing pid + addon path + load ordinal so
//!    host apps can log which instance they are actually talking to.
//!
//! The sentinel uses the process environment because it is the only
//! plain channel shared by every dlopen'd copy of this library. The
//! write happens during module load (dlopen), before the host can be
//! deep in multithreaded work, which keeps the classic setenv/getenv
//! thread-safety caveat theoretical.

use napi_derive::{module_init, napi};

/// Env key shared by all copies of the addon inside one process.
const INSTANCE_SENTINEL_ENV: &str = "MLX_NODE_NATIVE_INSTANCE";

/// 1-based ordinal of THIS copy of the library within the process,
/// captured at module load. 0 means the init hook has not run (never
/// expected in practice).
static LOAD_ORDINAL: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

/// Resolve the on-disk path of this loaded copy of the addon.
#[cfg(unix)]
fn addon_path() -> Option<String> {
    unsafe {
        let mut info: libc::Dl_info = std::mem::zeroed();
        if libc::dladdr(addon_path as *const libc::c_void, &mut info) != 0
            && !info.dli_fname.is_null()
        {
            Some(
                std::ffi::CStr::from_ptr(info.dli_fname)
                    .to_string_lossy()
                    .into_owned(),
            )
        } else {
            None
        }
    }
}

#[cfg(not(unix))]
fn addon_path() -> Option<String> {
    None
}

#[module_init]
pub fn init_instance_guard() {
    let own = addon_path().unwrap_or_else(|| "<unknown>".to_string());
    let prior = std::env::var(INSTANCE_SENTINEL_ENV).ok();
    let ordinal = match &prior {
        None => 1,
        Some(v) => {
            let (count, first_path) = v
                .split_once('#')
                .and_then(|(c, p)| c.parse::<u32>().ok().map(|c| (c, p)))
                .unwrap_or((1, v.as_str()));
            eprintln!(
                "[mlx-node] WARNING: a second copy of the native addon was loaded into this \
                 process.\n  first:  {first_path}\n  now:    {own}\n  Each copy has independent \
                 allocator/session state — memoryStats() and clearCache() on one copy cannot see \
                 the other. Require the addon through ONE resolved path (check symlinks and \
                 duplicate node_modules copies). nativeInstanceInfo() reports which copy you are \
                 talking to."
            );
            count + 1
        }
    };
    let _ = LOAD_ORDINAL.set(ordinal);
    let first_path = prior
        .as_deref()
        .and_then(|v| v.split_once('#').map(|(_, p)| p.to_string()))
        .unwrap_or(own);
    // SAFETY (edition 2024): set_var is process-global mutation. This runs
    // during dlopen of the addon — effectively startup — and the value is
    // only ever read back by later copies' inits and diagnostics.
    unsafe {
        std::env::set_var(INSTANCE_SENTINEL_ENV, format!("{ordinal}#{first_path}"));
    }
}

/// Identity of the native-addon instance answering this call.
#[napi(object)]
pub struct NativeInstanceInfo {
    /// OS process id. Cross-process confusion (a worker-pool process next
    /// to a trainer process) is the most common cause of "stats read 0".
    pub pid: u32,
    /// On-disk path of THIS loaded copy of the addon library.
    pub addon_path: Option<String>,
    /// 1-based load order of this copy within the process. Anything above
    /// 1 means the process holds duplicate copies of the addon.
    pub load_ordinal: u32,
    /// Total addon copies the process had loaded when this call ran.
    pub process_instances: u32,
}

/// Report pid, addon path, and duplicate-copy status for this native
/// instance. Cheap; safe to call from telemetry paths.
#[napi]
pub fn native_instance_info() -> NativeInstanceInfo {
    let process_instances = std::env::var(INSTANCE_SENTINEL_ENV)
        .ok()
        .and_then(|v| v.split_once('#').and_then(|(c, _)| c.parse::<u32>().ok()))
        .unwrap_or(0);
    NativeInstanceInfo {
        pid: std::process::id(),
        addon_path: addon_path(),
        load_ordinal: LOAD_ORDINAL.get().copied().unwrap_or(0),
        process_instances,
    }
}
