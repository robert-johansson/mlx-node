//! genmlx-core: the GenMLX-owned superset NAPI addon.
//!
//! Depends on stock `mlx-core` (rlib) + `mlx-sys`. Their `#[napi]` ctors survive
//! the rlib->cdylib link (DCE-survival, spike-proven — no -force_load), so this
//! single `.node` re-exports the full mlx-core surface PLUS the ported GenMLX
//! NAPI functions, sharing ONE NAPI type registry and ONE MLX runtime.
//!
//! NO `#[global_allocator]` here — mlx-core already installs mimalloc; a second
//! global-allocator attribute in the final artifact is a hard compile error.
//!
//! `transforms` + `memory_napi` are relocated here (free functions; the orphan
//! rule forbids re-`impl MxArray` outside mlx-core), and the keyed-PRNG FFI is
//! inlined into `genmlx`'s free fns — so mlx-core is stock above the substrate.

pub mod genmlx;
pub mod memory_napi;
pub mod transforms;
