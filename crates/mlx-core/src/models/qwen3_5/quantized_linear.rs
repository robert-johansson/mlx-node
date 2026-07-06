use std::collections::HashMap;
use std::ffi::CString;

use crate::array::MxArray;
use crate::nn::{Activations, Linear};
use crate::transformer::MLP;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::int8_gemm;

/// Default quantization parameters for 4-bit models.
pub const DEFAULT_QUANT_BITS: i32 = 4;
pub const DEFAULT_QUANT_GROUP_SIZE: i32 = 64;
/// Router gates use higher precision (8-bit affine, group_size=64).
pub const GATE_QUANT_BITS: i32 = 8;
pub const GATE_QUANT_GROUP_SIZE: i32 = 64;
pub const DEFAULT_QUANT_MODE: &str = "affine";

/// MXFP8 quantization parameters (for FP8 source checkpoints).
pub const MXFP8_BITS: i32 = 8;
pub const MXFP8_GROUP_SIZE: i32 = 32;
pub const MXFP8_MODE: &str = "mxfp8";

/// MXFP4 quantization parameters (E2M1 format, fixed bits/group_size).
pub const MXFP4_BITS: i32 = 4;
pub const MXFP4_GROUP_SIZE: i32 = 32;
pub const MXFP4_MODE: &str = "mxfp4";

/// NVFP4 quantization parameters (E2M1 4-bit weights with E4M3 uint8 scales,
/// group_size 16).
pub const NVFP4_BITS: i32 = 4;
pub const NVFP4_GROUP_SIZE: i32 = 16;
pub const NVFP4_MODE: &str = "nvfp4";

// `PerLayerMode` and `PerLayerQuant` are family-neutral types shared with
// `qwen3_5_moe` and `gemma4`; they live in `crate::models::quant_dispatch`
// so the three families don't cross-import from each other.
pub use crate::models::quant_dispatch::{PerLayerMode, PerLayerQuant};

/// A linear projection that can be either standard or quantized.
///
/// Shared between attention, GatedDeltaNet, and SparseMoeBlock.
pub enum LinearProj {
    Standard(Linear),
    Quantized(QuantizedLinear),
}

impl LinearProj {
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            LinearProj::Standard(l) => l.forward(x),
            LinearProj::Quantized(l) => l.forward(x),
        }
    }

    pub fn set_weight(&mut self, w: &MxArray, name: &str) -> Result<()> {
        match self {
            LinearProj::Standard(l) => l.set_weight(w),
            LinearProj::Quantized(_) => Err(Error::from_reason(format!(
                "Cannot set weight on quantized {}",
                name
            ))),
        }
    }

    pub fn set_bias(&mut self, b: Option<&MxArray>, name: &str) -> Result<()> {
        match self {
            LinearProj::Standard(l) => l.set_bias(b),
            LinearProj::Quantized(_) => Err(Error::from_reason(format!(
                "Cannot set bias on quantized {}",
                name
            ))),
        }
    }

    pub fn set_quantized(&mut self, ql: QuantizedLinear) {
        *self = LinearProj::Quantized(ql);
    }

    pub fn get_weight(&self) -> MxArray {
        match self {
            LinearProj::Standard(l) => l.get_weight(),
            LinearProj::Quantized(ql) => ql.get_weight().clone(),
        }
    }

    /// Replace a `Quantized` backend with a `Standard(Linear)` holding the
    /// dequantized dense weight (and any additive bias). Returns whether a
    /// conversion happened; a `Standard` projection is a no-op.
    ///
    /// This is the training-side counterpart of `nn::Linear::set_weight` /
    /// `nn::Embedding::load_weight` clearing their quantized backends: the
    /// GRPO/SFT functional forward and `apply_gradients` both require dense
    /// weights (a packed uint32 tensor fed to a plain matmul is a shape error
    /// — genmlx-x76x), so `init_training_sync` converts the whole model once
    /// up front rather than dequantizing per step inside the autograd graph.
    pub fn dequantize_to_standard(&mut self) -> Result<bool> {
        match self {
            LinearProj::Standard(_) => Ok(false),
            LinearProj::Quantized(ql) => {
                let dense = ql.dequantize_dense()?;
                let linear = Linear::from_weights(&dense, ql.get_linear_bias())?;
                *self = LinearProj::Standard(linear);
                Ok(true)
            }
        }
    }

    /// Whether this projection holds a quantized backend.
    ///
    /// Used by the dense/bf16-only `save_model_sync` MTP path to refuse
    /// emitting a quantized projection's stale dense `weight` as if it were
    /// a valid bf16 tensor (see `Qwen3_5MTPModule::has_quantized_weights`).
    pub fn is_quantized(&self) -> bool {
        matches!(self, LinearProj::Quantized(_))
    }
}

/// An MLP that can be either standard or quantized.
///
/// Shared between decoder_layer and sparse_moe.
pub enum MLPVariant {
    Standard(MLP),
    Quantized {
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    },
}

impl MLPVariant {
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            MLPVariant::Standard(mlp) => mlp.forward(x),
            MLPVariant::Quantized {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                let gate = gate_proj.forward(x)?;
                let up = up_proj.forward(x)?;
                let activated = Activations::swiglu(&gate, &up)?;
                down_proj.forward(&activated)
            }
        }
    }

    pub fn get_gate_proj_weight(&self) -> MxArray {
        match self {
            MLPVariant::Standard(mlp) => mlp.get_gate_proj_weight(),
            MLPVariant::Quantized { gate_proj, .. } => gate_proj.get_weight().clone(),
        }
    }

    pub fn get_up_proj_weight(&self) -> MxArray {
        match self {
            MLPVariant::Standard(mlp) => mlp.get_up_proj_weight(),
            MLPVariant::Quantized { up_proj, .. } => up_proj.get_weight().clone(),
        }
    }

    pub fn get_down_proj_weight(&self) -> MxArray {
        match self {
            MLPVariant::Standard(mlp) => mlp.get_down_proj_weight(),
            MLPVariant::Quantized { down_proj, .. } => down_proj.get_weight().clone(),
        }
    }

    /// Whether this MLP holds a quantized backend.
    ///
    /// Used by the dense/bf16-only `save_model_sync` MTP path to refuse
    /// emitting a quantized MLP's stale dense weights (see
    /// `Qwen3_5MTPModule::has_quantized_weights`).
    pub fn is_quantized(&self) -> bool {
        matches!(self, MLPVariant::Quantized { .. })
    }

    pub fn set_gate_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            MLPVariant::Standard(mlp) => mlp.set_gate_proj_weight(w),
            MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }

    pub fn set_up_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            MLPVariant::Standard(mlp) => mlp.set_up_proj_weight(w),
            MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }

    /// E39: finalize stacked gate+up weight. No-op for quantized variant.
    pub fn finalize_gate_up(&mut self) -> Result<()> {
        match self {
            MLPVariant::Standard(mlp) => mlp.finalize_gate_up(),
            MLPVariant::Quantized { .. } => Ok(()),
        }
    }

    pub fn set_down_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            MLPVariant::Standard(mlp) => mlp.set_down_proj_weight(w),
            MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }

    /// Replace a `Quantized` MLP with a `Standard(MLP)` built from the three
    /// dequantized dense weights. Returns whether a conversion happened.
    /// See [`LinearProj::dequantize_to_standard`] for why (genmlx-x76x).
    ///
    /// Errors on an additive linear bias: `MLP` has no bias slots (qwen3.5
    /// MLPs are bias-free), and silently dropping one would corrupt the model.
    pub fn dequantize_to_standard(&mut self) -> Result<bool> {
        match self {
            MLPVariant::Standard(_) => Ok(false),
            MLPVariant::Quantized {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                if gate_proj.get_linear_bias().is_some()
                    || up_proj.get_linear_bias().is_some()
                    || down_proj.get_linear_bias().is_some()
                {
                    return Err(Error::from_reason(
                        "dequantize_to_standard: quantized MLP carries an additive bias, \
                         which the dense MLP cannot hold",
                    ));
                }
                let mlp = MLP::from_weights(
                    &gate_proj.dequantize_dense()?,
                    &up_proj.dequantize_dense()?,
                    &down_proj.dequantize_dense()?,
                )?;
                *self = MLPVariant::Standard(mlp);
                Ok(true)
            }
        }
    }
}

/// Check if a model checkpoint is quantized by looking for `.scales` keys.
pub fn is_quantized_checkpoint(params: &HashMap<String, MxArray>) -> bool {
    params.keys().any(|k| k.ends_with(".scales"))
}

/// Check if a checkpoint uses MXFP8 quantization (Uint8 scales = E8M0 format).
pub fn is_mxfp8_checkpoint(params: &HashMap<String, MxArray>) -> bool {
    params
        .iter()
        .any(|(k, v)| k.ends_with(".scales") && matches!(v.dtype(), Ok(crate::array::DType::Uint8)))
}

/// Try to build an MXFP8 QuantizedLinear from weight/scales keys in a params map.
/// MXFP8 has no biases (only weight + scales).
pub fn try_build_mxfp8_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Option<QuantizedLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    Some(QuantizedLinear::new(
        weight.clone(),
        scales.clone(),
        None,
        None,
        MXFP8_GROUP_SIZE,
        MXFP8_BITS,
        MXFP8_MODE.to_string(),
    ))
}

/// Try to build an MXFP4 QuantizedLinear from weight/scales keys in a params map.
/// MXFP4 has no biases (only weight + uint8 E2M1 scales), fixed at 4 bits / group_size 32.
pub fn try_build_mxfp4_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Option<QuantizedLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    Some(QuantizedLinear::new(
        weight.clone(),
        scales.clone(),
        None,
        None,
        MXFP4_GROUP_SIZE,
        MXFP4_BITS,
        MXFP4_MODE.to_string(),
    ))
}

/// Try to build an NVFP4 QuantizedLinear from weight/scales keys in a params map.
/// NVFP4 has no biases (only weight + uint8 E4M3 scales), fixed at 4 bits / group_size 16.
pub fn try_build_nvfp4_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Option<QuantizedLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    Some(QuantizedLinear::new(
        weight.clone(),
        scales.clone(),
        None,
        None,
        NVFP4_GROUP_SIZE,
        NVFP4_BITS,
        NVFP4_MODE.to_string(),
    ))
}

/// Try to build a QuantizedLinear from weight/scales/biases keys in a params map.
pub fn try_build_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
    group_size: i32,
    bits: i32,
) -> Option<QuantizedLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    let biases = params.get(&format!("{}.biases", key_prefix)).cloned();
    Some(QuantizedLinear::new(
        weight.clone(),
        scales.clone(),
        biases,
        None,
        group_size,
        bits,
        DEFAULT_QUANT_MODE.to_string(),
    ))
}

/// sym8 quantization parameters (per-output-channel symmetric int8 weights
/// with f32 `[N]` scales; `group_size` is null in the checkpoint and
/// meaningless at runtime — `SYM8_GROUP_SIZE` is a placeholder for the
/// struct field only).
pub const SYM8_BITS: i32 = 8;
pub const SYM8_GROUP_SIZE: i32 = -1;
pub const SYM8_MODE: &str = "sym8";

/// Try to build a sym8 `QuantizedLinear` from `{prefix}.weight` (int8 `[N,K]`)
/// + `{prefix}.scales` (f32 `[N]`) in a params map.
///
/// Returns `Ok(None)` ONLY when `{prefix}.scales` is absent — that is the
/// "this layer is not quantized" signal shared with the other `try_build_*`
/// helpers (a sym8-default checkpoint legitimately carries bf16 layers with
/// no sidecar, e.g. a forced-affine tensor that also failed the K%64 gate).
///
/// Everything else is FAIL-LOUD `Err` (convert should have prevented all of
/// these — assert anyway, a silent fallback would emit garbage):
///   * `.scales` present but `.weight` missing (corrupt checkpoint),
///   * a `.biases` sidecar (sym8 has none by construction),
///   * weight not 2-D int8, scales not 1-D f32, or `scales.len() != N`,
///   * `K % 16 != 0` (kernel contract),
///   * GPU gen < 17 (the int8 kernels need M5+; the convert-side
///     `sym8_eligible` deliberately omits this runtime-only gate).
///
/// The `[K,N]` kernel operand is built ONCE here at load time
/// ([`int8_gemm::sym8_kernel_operand`] — the exact transpose+contiguous tail
/// of `quantize_weight_int8`, requant-free), so forward does zero weight
/// reshaping.
pub fn try_build_sym8_quantized_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
) -> Result<Option<QuantizedLinear>> {
    let Some(scales) = params.get(&format!("{}.scales", key_prefix)) else {
        return Ok(None);
    };
    let Some(weight) = params.get(&format!("{}.weight", key_prefix)) else {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': .scales present but .weight missing (corrupt checkpoint)",
            key_prefix
        )));
    };
    if params.contains_key(&format!("{}.biases", key_prefix)) {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': unexpected .biases sidecar (sym8 is symmetric — convert never emits one)",
            key_prefix
        )));
    }

    let gpu_gen = unsafe { sys::mlx_gpu_architecture_gen() };
    if gpu_gen < 17 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': sym8 checkpoints require an M5+ GPU (gen >= 17), got gen {}. \
             Re-convert the model with an affine quant mode for this host.",
            key_prefix, gpu_gen
        )));
    }

    let w_dtype = weight.dtype()?;
    if w_dtype != crate::array::DType::Int8 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': expected int8 .weight, got {:?}",
            key_prefix, w_dtype
        )));
    }
    let w_shape = weight.shape()?;
    if w_shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': expected 2-D [N,K] .weight, got {:?}",
            key_prefix,
            &w_shape[..]
        )));
    }
    let (n, k) = (w_shape[0], w_shape[1]);
    if k % 16 != 0 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': K={} violates the kernel's K % 16 == 0 contract \
             (convert's sym8_eligible gate should have forced this layer to affine)",
            key_prefix, k
        )));
    }
    let s_dtype = scales.dtype()?;
    if s_dtype != crate::array::DType::Float32 {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': expected f32 .scales, got {:?}",
            key_prefix, s_dtype
        )));
    }
    let s_shape = scales.shape()?;
    if s_shape.len() != 1 || s_shape[0] != n {
        return Err(Error::from_reason(format!(
            "sym8 layer '{}': expected 1-D [N={}] .scales, got {:?}",
            key_prefix,
            n,
            &s_shape[..]
        )));
    }

    // Build the [K,N] kernel operand ONCE at load (fail-loud on FFI error).
    let w_kn = int8_gemm::sym8_kernel_operand(weight)?;
    Ok(Some(QuantizedLinear::new_sym8(
        weight.clone(),
        w_kn,
        scales.clone(),
        None,
    )))
}

/// QuantizedLinear: Linear layer using quantized_matmul for efficient inference.
///
/// Stores weights in packed uint32 format with separate scales and optional biases.
/// Uses MLX's fused dequantize+matmul Metal kernel for ~4x memory reduction.
pub struct QuantizedLinear {
    weight: MxArray,         // Packed uint32 quantized weights [out, in_packed]
    scales: MxArray,         // Quantization scales
    biases: Option<MxArray>, // Quantization biases (for affine mode)
    bias: Option<MxArray>,   // Linear bias (additive)
    group_size: i32,
    bits: i32,
    mode: String, // "affine", "mxfp8", "mxfp4", "nvfp4", or "sym8"
    // sym8 kernel operands (`Some` iff `mode == "sym8"`): `w_i8` is the opaque
    // contiguous [K,N] int8 weight (pre-transposed at load), `s_w` is the f32
    // [N] per-output-channel scale. Consumed by `int8_w8a16_qmv` (M <= 2,
    // decode — bf16 activations, no act quant; ALSO takes `self.weight`, the
    // [N,K] checkpoint tensor, which its default simd_sum kernel streams
    // row-major) / `int8_w8a8_matmul` (M >= 3, prefill) — NEVER by
    // `mlx_quantized_matmul` (there is no affine pack for sym8).
    w_i8: Option<MxArray>,
    s_w: Option<MxArray>,
}

/// Routing observability for the sym8 forward (unit-test scope only):
/// counts how many sym8 forwards took the QMV (decode) vs GEMM (prefill)
/// kernel, so tests can assert the M-dispatch without relying on the two
/// kernels producing different bits.
#[cfg(test)]
pub(crate) static SYM8_QMV_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(crate) static SYM8_GEMM_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `MLX_SYM8_DEBUG=1` prints one line per sym8 forward with the chosen kernel
/// and the (M, K, N) shape — e2e dispatch evidence. Read once per process.
fn sym8_debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var("MLX_SYM8_DEBUG") {
        Ok(v) => !v.is_empty() && v != "0" && v != "false",
        Err(_) => false,
    })
}

impl QuantizedLinear {
    pub fn new(
        weight: MxArray,
        scales: MxArray,
        biases: Option<MxArray>,
        bias: Option<MxArray>,
        group_size: i32,
        bits: i32,
        mode: String,
    ) -> Self {
        Self {
            weight,
            scales,
            biases,
            bias,
            group_size,
            bits,
            mode,
            w_i8: None,
            s_w: None,
        }
    }

    /// Construct a sym8 linear from pre-validated operands (see
    /// [`try_build_sym8_quantized_linear`] for the load-time validation).
    ///
    /// `weight` is the STORED int8 `[N,K]` checkpoint tensor (kept so
    /// `get_weight()` returns the source-layout tensor like every other
    /// mode — it shares the underlying buffer with the params map entry);
    /// `w_kn` is the contiguous `[K,N]` kernel operand; `s_w` is the f32
    /// `[N]` scale (doubling as the `scales` field).
    pub fn new_sym8(weight: MxArray, w_kn: MxArray, s_w: MxArray, bias: Option<MxArray>) -> Self {
        Self {
            weight,
            scales: s_w.clone(),
            biases: None,
            bias,
            group_size: SYM8_GROUP_SIZE,
            bits: SYM8_BITS,
            mode: SYM8_MODE.to_string(),
            w_i8: Some(w_kn),
            s_w: Some(s_w),
        }
    }

    /// sym8 forward: int8-weight GEMM/QMV + rescale.
    ///
    /// Dispatch rule: `M <= 2` → [`int8_gemm::int8_w8a16_qmv`] (the dedicated
    /// W8A16 decode matvec — bf16 activations read directly, NO act quant,
    /// activation-exact; the W8A8 act-quant passes were pure in-stream
    /// overhead at decode M and the prefill tile wastes 127/128 rows at M=1),
    /// `M >= 3` → [`int8_gemm::int8_w8a8_matmul`] (the W8A8 prefill GEMM —
    /// act quant amortizes at prefill M). The only env gate is the
    /// same-binary A/B escape hatch `INT8_QMV_W8A16=0` (read inside the
    /// shared C++ builder, so eager and compiled stay byte-identical) which
    /// reroutes decode back to the W8A8 qmv. Fail-loud on kernel error
    /// (there is no affine pack to fall back to).
    fn forward_sym8(&self, x: &MxArray) -> Result<MxArray> {
        let (Some(w_i8), Some(s_w)) = (self.w_i8.as_ref(), self.s_w.as_ref()) else {
            return Err(Error::from_reason(
                "sym8 QuantizedLinear missing kernel operands (w_i8/s_w) — \
                 constructed without new_sym8?",
            ));
        };
        let shape = x.shape()?;
        if shape.is_empty() {
            return Err(Error::from_reason("sym8 forward: scalar input"));
        }
        let k = shape[shape.len() - 1];
        let m: i64 = shape[..shape.len() - 1].iter().product();
        let x2d = x.reshape(&[m, k])?;
        let y2d = if m <= 2 {
            #[cfg(test)]
            SYM8_QMV_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // `self.weight` is the stored [N,K] int8 checkpoint tensor — the
            // decode matvec's simd_sum kernel streams that orientation
            // directly (the [K,N] operand stays for fallback/A-B kernels).
            int8_gemm::int8_w8a16_qmv(&x2d, w_i8, &self.weight, s_w)?
        } else {
            #[cfg(test)]
            SYM8_GEMM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            int8_gemm::int8_w8a8_matmul(&x2d, w_i8, s_w)?
        };
        let n = y2d.shape_at(1)?;
        if sym8_debug_enabled() {
            eprintln!(
                "[sym8] {} M={m} K={k} N={n}",
                if m <= 2 { "qmv" } else { "gemm" }
            );
        }
        let mut out_shape: Vec<i64> = shape[..shape.len() - 1].to_vec();
        out_shape.push(n);
        let mut result = y2d.reshape(&out_shape)?;
        if let Some(ref b) = self.bias {
            result = result.add(b)?;
        }
        Ok(result)
    }

    /// Forward pass using quantized_matmul (sym8 routes to the int8 W8A8
    /// kernels instead — `mlx_quantized_matmul` has no sym8 pack).
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        if self.mode == SYM8_MODE {
            return self.forward_sym8(x);
        }

        let mode_c = CString::new(self.mode.as_str())
            .map_err(|e| Error::from_reason(format!("Invalid mode string: {}", e)))?;

        let biases_ptr = self
            .biases
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.handle.0);

        let handle = unsafe {
            sys::mlx_quantized_matmul(
                x.handle.0,
                self.weight.handle.0,
                self.scales.handle.0,
                biases_ptr,
                true, // transpose
                self.group_size,
                self.bits,
                mode_c.as_ptr(),
            )
        };
        let mut result = MxArray::from_handle(handle, "quantized_matmul")?;

        // Add linear bias if present
        if let Some(ref b) = self.bias {
            result = result.add(b)?;
        }

        Ok(result)
    }

    pub fn set_weight(&mut self, weight: MxArray) {
        self.weight = weight;
    }

    pub fn set_scales(&mut self, scales: MxArray) {
        self.scales = scales;
    }

    pub fn set_biases(&mut self, biases: Option<MxArray>) {
        self.biases = biases;
    }

    pub fn set_bias(&mut self, bias: Option<MxArray>) {
        self.bias = bias;
    }

    pub fn get_weight(&self) -> &MxArray {
        &self.weight
    }

    pub fn get_scales(&self) -> &MxArray {
        &self.scales
    }

    pub fn get_biases(&self) -> Option<&MxArray> {
        self.biases.as_ref()
    }

    /// The additive linear bias (NOT the quantization biases), if present.
    pub fn get_linear_bias(&self) -> Option<&MxArray> {
        self.bias.as_ref()
    }

    /// Dequantize the packed weight to the dense `[out, in]` tensor forward
    /// semantics are defined over (`quantized_matmul(x, w, ...) ==
    /// matmul(x, dequantize(w, ...).T)`). Row order is preserved, so merged
    /// projections (qkvz/ba) keep their contiguous-or-interleaved layout.
    ///
    /// affine/mxfp8/mxfp4/nvfp4 route through `mlx_dequantize` with the stored
    /// mode (out dtype = the scales dtype for affine, i.e. the model dtype);
    /// sym8 is `w_i8[N,K] * s_w[N,1]` cast to bf16 (there is no affine pack).
    pub fn dequantize_dense(&self) -> Result<MxArray> {
        if self.mode == SYM8_MODE {
            let s_w = self.s_w.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "dequantize_dense: sym8 QuantizedLinear missing s_w scale operand",
                )
            })?;
            let n = s_w.shape_at(0)?;
            let w_f = self.weight.astype(crate::array::DType::Float32)?;
            let dense = w_f.mul(&s_w.reshape(&[n, 1])?)?;
            return dense.astype(crate::array::DType::BFloat16);
        }
        let mode_c = CString::new(self.mode.as_str())
            .map_err(|e| Error::from_reason(format!("Invalid mode string: {}", e)))?;
        let biases_ptr = self
            .biases
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.handle.0);
        let handle = unsafe {
            sys::mlx_dequantize(
                self.weight.handle.0,
                self.scales.handle.0,
                biases_ptr,
                self.group_size,
                self.bits,
                -1, // input (scales) dtype — bf16 for affine checkpoints
                mode_c.as_ptr(),
            )
        };
        MxArray::from_handle(handle, "quantized_linear_dequantize")
    }

    /// Quantization mode discriminator string ("affine", "mxfp8", "mxfp4",
    /// "nvfp4", or "sym8").
    pub fn mode(&self) -> &str {
        &self.mode
    }

    /// Test-scope accessor for the sym8 operands
    /// `(w_nk [N,K] checkpoint, w_kn [K,N] kernel operand, s_w [N])`.
    /// Used by the routing/parity unit tests to call the reference kernels with
    /// the exact operands forward consumes.
    #[cfg(test)]
    pub(crate) fn sym8_operands(&self) -> Option<(&MxArray, &MxArray, &MxArray)> {
        match (self.w_i8.as_ref(), self.s_w.as_ref()) {
            (Some(w), Some(s)) => Some((&self.weight, w, s)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod sym8_tests {
    use super::*;
    use crate::array::DType;
    use std::sync::atomic::Ordering;

    fn gpu_gen() -> i32 {
        unsafe { sys::mlx_gpu_architecture_gen() }
    }

    /// Deterministic pseudo-random integer in `[lo, hi]` (LCG — failures
    /// reproduce exactly). Mirrors the helper in `int8_gemm::tests`.
    fn next_int(state: &mut u64, lo: i32, hi: i32) -> i32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let span = (hi - lo + 1) as u64;
        lo + ((*state >> 33) % span) as i32
    }

    /// Fabricate a synthetic sym8 checkpoint layer: int8 `[N,K]` weight with
    /// integer values in [-127,127] plus positive f32 `[N]` scales, inserted
    /// under `{prefix}.weight` / `{prefix}.scales`.
    fn synth_sym8_params(prefix: &str, n: i64, k: i64, seed: u64) -> HashMap<String, MxArray> {
        let mut state = seed;
        let q: Vec<f32> = (0..n * k)
            .map(|_| next_int(&mut state, -127, 127) as f32)
            .collect();
        let w_i8 = MxArray::from_float32(&q, &[n, k])
            .unwrap()
            .astype(DType::Int8)
            .unwrap();
        let scales: Vec<f32> = (0..n)
            .map(|_| 0.001 + (next_int(&mut state, 1, 1000) as f32) * 1e-5)
            .collect();
        let s_w = MxArray::from_float32(&scales, &[n]).unwrap();
        let mut params = HashMap::new();
        params.insert(format!("{prefix}.weight"), w_i8);
        params.insert(format!("{prefix}.scales"), s_w);
        params
    }

    /// Random bf16 activations `[shape]` in roughly [-2, 2].
    fn synth_x_bf16(shape: &[i64], seed: u64) -> MxArray {
        let mut state = seed;
        let len: i64 = shape.iter().product();
        let v: Vec<f32> = (0..len)
            .map(|_| next_int(&mut state, -2000, 2000) as f32 / 1000.0)
            .collect();
        MxArray::from_float32(&v, shape)
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    /// bf16 outputs compared bit-for-bit via the native u16 payload
    /// (no f32 round-trip — see project memory).
    fn assert_bf16_bit_identical(a: &MxArray, b: &MxArray, ctx: &str) {
        a.eval();
        b.eval();
        let av = a.to_uint16_native().unwrap();
        let bv = b.to_uint16_native().unwrap();
        assert_eq!(av.len(), bv.len(), "{ctx}: length mismatch");
        let bad = av.iter().zip(bv.iter()).filter(|(x, y)| x != y).count();
        assert_eq!(bad, 0, "{ctx}: {bad}/{} bf16 words differ", av.len());
    }

    /// GATE (a): M=1 routes the QMV kernel, M=512 routes the GEMM kernel, and
    /// each output is bit-for-bit identical to calling the matching
    /// `int8_gemm` reference op directly with the layer's own operands.
    #[test]
    fn sym8_forward_routes_qmv_at_m1_gemm_at_m512_bit_exact() {
        if gpu_gen() < 17 {
            eprintln!(
                "[sym8] SKIP: gpu gen {} < 17 (int8 kernels need M5+)",
                gpu_gen()
            );
            return;
        }
        let (n, k) = (48i64, 64i64); // K % 16 == 0
        let params = synth_sym8_params("test_layer", n, k, 0x5197_0001);
        let ql = try_build_sym8_quantized_linear(&params, "test_layer")
            .expect("builder must succeed on a well-formed sym8 layer")
            .expect("scales present => Some");
        assert_eq!(ql.mode(), SYM8_MODE);
        let (w_nk, w_kn, s_w) = ql.sym8_operands().expect("sym8 operands present");
        // The load-time kernel operand is the contiguous [K,N] transpose of
        // the stored [N,K] weight — verify element-exactly via int8 bytes.
        let stored = params.get("test_layer.weight").unwrap().to_int8().unwrap();
        let kn = w_kn.to_int8().unwrap();
        assert_eq!(w_kn.shape().unwrap().to_vec(), vec![k, n]);
        for ni in 0..n as usize {
            for ki in 0..k as usize {
                assert_eq!(
                    stored[ni * k as usize + ki],
                    kn[ki * n as usize + ni],
                    "kernel operand transpose mismatch at (n={ni}, k={ki})"
                );
            }
        }

        // --- M=1 → QMV ---
        let x1 = synth_x_bf16(&[1, k], 0xaaaa_0001);
        let qmv_before = SYM8_QMV_CALLS.load(Ordering::Relaxed);
        let gemm_before = SYM8_GEMM_CALLS.load(Ordering::Relaxed);
        let y1 = ql.forward(&x1).unwrap();
        assert_eq!(
            SYM8_QMV_CALLS.load(Ordering::Relaxed),
            qmv_before + 1,
            "M=1 must route the QMV kernel"
        );
        assert_eq!(
            SYM8_GEMM_CALLS.load(Ordering::Relaxed),
            gemm_before,
            "M=1 must NOT route the GEMM kernel"
        );
        let y1_ref = int8_gemm::int8_w8a16_qmv(&x1, w_kn, w_nk, s_w).unwrap();
        assert_bf16_bit_identical(&y1, &y1_ref, "M=1 qmv parity");

        // --- M=2 still QMV (decode-dispatch upper bound) ---
        let x2 = synth_x_bf16(&[2, k], 0xaaaa_0002);
        let qmv_before = SYM8_QMV_CALLS.load(Ordering::Relaxed);
        ql.forward(&x2).unwrap().eval();
        assert_eq!(SYM8_QMV_CALLS.load(Ordering::Relaxed), qmv_before + 1);

        // --- M=3 first GEMM M ---
        let x3 = synth_x_bf16(&[3, k], 0xaaaa_0003);
        let gemm_before = SYM8_GEMM_CALLS.load(Ordering::Relaxed);
        ql.forward(&x3).unwrap().eval();
        assert_eq!(SYM8_GEMM_CALLS.load(Ordering::Relaxed), gemm_before + 1);

        // --- M=512 (prefill, 3-D input [B, S, K]) → GEMM ---
        let x512 = synth_x_bf16(&[4, 128, k], 0xaaaa_0512);
        let qmv_before = SYM8_QMV_CALLS.load(Ordering::Relaxed);
        let gemm_before = SYM8_GEMM_CALLS.load(Ordering::Relaxed);
        let y512 = ql.forward(&x512).unwrap();
        assert_eq!(
            SYM8_GEMM_CALLS.load(Ordering::Relaxed),
            gemm_before + 1,
            "M=512 must route the GEMM kernel"
        );
        assert_eq!(
            SYM8_QMV_CALLS.load(Ordering::Relaxed),
            qmv_before,
            "M=512 must NOT route the QMV kernel"
        );
        assert_eq!(y512.shape().unwrap().to_vec(), vec![4, 128, n]);
        let x512_2d = x512.reshape(&[512, k]).unwrap();
        let y512_ref = int8_gemm::int8_w8a8_matmul(&x512_2d, w_kn, s_w)
            .unwrap()
            .reshape(&[4, 128, n])
            .unwrap();
        assert_bf16_bit_identical(&y512, &y512_ref, "M=512 gemm parity");
    }

    /// Additive linear bias is applied after the int8 kernel.
    #[test]
    fn sym8_forward_applies_linear_bias() {
        if gpu_gen() < 17 {
            eprintln!("[sym8] SKIP: gpu gen {} < 17", gpu_gen());
            return;
        }
        let (n, k) = (32i64, 64i64);
        let params = synth_sym8_params("biased", n, k, 0x5197_0002);
        let weight = params.get("biased.weight").unwrap().clone();
        let scales = params.get("biased.scales").unwrap().clone();
        let bias = synth_x_bf16(&[n], 0xbbbb_0001);
        let w_kn = int8_gemm::sym8_kernel_operand(&weight).unwrap();
        let ql = QuantizedLinear::new_sym8(
            weight.clone(),
            w_kn.clone(),
            scales.clone(),
            Some(bias.clone()),
        );
        let x = synth_x_bf16(&[1, k], 0xbbbb_0002);
        let y = ql.forward(&x).unwrap();
        let y_ref = int8_gemm::int8_w8a16_qmv(&x, &w_kn, &weight, &scales)
            .unwrap()
            .add(&bias)
            .unwrap();
        assert_bf16_bit_identical(&y, &y_ref, "bias add parity");
    }

    /// Load-time fail-loud contract: every malformed sym8 layer is an `Err`
    /// (never a silent `None` fallback), while a genuinely-absent sidecar is
    /// `Ok(None)`.
    #[test]
    fn sym8_builder_fail_loud_contract() {
        if gpu_gen() < 17 {
            eprintln!(
                "[sym8] SKIP: gpu gen {} < 17 (builder gen-gate untestable)",
                gpu_gen()
            );
            return;
        }
        let (n, k) = (16i64, 32i64);

        // Missing .scales → Ok(None) (bf16-fallback layer in a sym8 checkpoint).
        let mut p = synth_sym8_params("l", n, k, 1);
        p.remove("l.scales");
        assert!(matches!(try_build_sym8_quantized_linear(&p, "l"), Ok(None)));

        // .scales present but .weight missing → Err.
        let mut p = synth_sym8_params("l", n, k, 2);
        p.remove("l.weight");
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // Unexpected .biases sidecar → Err.
        let mut p = synth_sym8_params("l", n, k, 3);
        let zeros = vec![0.0f32; n as usize];
        p.insert(
            "l.biases".into(),
            MxArray::from_float32(&zeros, &[n]).unwrap(),
        );
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // Non-int8 weight dtype → Err.
        let mut p = synth_sym8_params("l", n, k, 4);
        let w_f = p.get("l.weight").unwrap().astype(DType::Float32).unwrap();
        p.insert("l.weight".into(), w_f);
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // K % 16 != 0 → Err.
        let p = synth_sym8_params("l", n, 24, 5);
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // Non-f32 scales dtype → Err.
        let mut p = synth_sym8_params("l", n, k, 6);
        let s_b = p.get("l.scales").unwrap().astype(DType::BFloat16).unwrap();
        p.insert("l.scales".into(), s_b);
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());

        // Scales length != N → Err.
        let mut p = synth_sym8_params("l", n, k, 7);
        let short_scales = vec![0.001f32; (n - 1) as usize];
        p.insert(
            "l.scales".into(),
            MxArray::from_float32(&short_scales, &[n - 1]).unwrap(),
        );
        assert!(try_build_sym8_quantized_linear(&p, "l").is_err());
    }
}

#[cfg(test)]
mod dequantize_to_standard_tests {
    use super::*;

    /// Affine-quantize a dense f32 weight via `mlx_quantize`, returning
    /// `(packed_weight, scales, biases)`. Mirrors the helper in
    /// `nn::embedding` tests.
    fn quantize_affine(weight: &MxArray, group_size: i32, bits: i32) -> (MxArray, MxArray, MxArray) {
        let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_quantize(
                weight.as_raw_ptr(),
                group_size,
                bits,
                c"affine".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize affine failed");
        assert!(!out_b.is_null(), "affine quantize must return biases");
        (
            MxArray::from_handle(out_q, "q").expect("q"),
            MxArray::from_handle(out_s, "s").expect("s"),
            MxArray::from_handle(out_b, "b").expect("b"),
        )
    }

    fn to_vec(arr: &MxArray) -> Vec<f32> {
        arr.astype(crate::array::DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec()
    }

    /// genmlx-x76x: converting a Quantized projection to Standard must
    /// preserve forward semantics — `quantized_matmul(x, w, s, b, T)` and
    /// `x @ dequantize(w, s, b).T` are the same math.
    #[test]
    fn linear_proj_conversion_preserves_forward() {
        let (out_f, in_f) = (8i64, 64i64);
        let w_data: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i as f32 * 0.317).sin()) * 0.1)
            .collect();
        let weight = MxArray::from_float32(&w_data, &[out_f, in_f]).unwrap();
        let (q, s, b) = quantize_affine(&weight, 32, 4);

        let mut proj = LinearProj::Quantized(QuantizedLinear::new(
            q,
            s,
            Some(b),
            None,
            32,
            4,
            "affine".to_string(),
        ));

        let x_data: Vec<f32> = (0..2 * in_f).map(|i| ((i as f32 * 0.113).cos()) * 0.5).collect();
        let x = MxArray::from_float32(&x_data, &[2, in_f]).unwrap();

        let y_quant = to_vec(&proj.forward(&x).unwrap());

        assert!(proj.dequantize_to_standard().unwrap());
        assert!(!proj.is_quantized());
        // Idempotent on a Standard projection.
        assert!(!proj.dequantize_to_standard().unwrap());

        let y_dense = to_vec(&proj.forward(&x).unwrap());
        assert_eq!(y_quant.len(), y_dense.len());
        for (i, (a, c)) in y_quant.iter().zip(y_dense.iter()).enumerate() {
            assert!(
                (a - c).abs() <= 1e-3 + 1e-2 * c.abs(),
                "forward diverged at {i}: quantized={a} dense={c}"
            );
        }
    }

    /// The dense weight round-trips shape and stays within the 4-bit
    /// quantization error of the original.
    #[test]
    fn dequantize_dense_shape_and_error_bound() {
        let (out_f, in_f) = (6i64, 32i64);
        let w_data: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32 * 0.071).sin()).collect();
        let weight = MxArray::from_float32(&w_data, &[out_f, in_f]).unwrap();
        let (q, s, b) = quantize_affine(&weight, 32, 4);
        let ql = QuantizedLinear::new(q, s, Some(b), None, 32, 4, "affine".to_string());

        let dense = ql.dequantize_dense().unwrap();
        assert_eq!(dense.shape_at(0).unwrap(), out_f);
        assert_eq!(dense.shape_at(1).unwrap(), in_f);
        let dv = to_vec(&dense);
        // 4-bit affine over group 32: max |err| <= half a quantization step;
        // the weights span ~[-1, 1] so a generous 0.2 bound catches layout
        // bugs (transposed/packed reads) without flaking on rounding.
        for (i, (orig, deq)) in w_data.iter().zip(dv.iter()).enumerate() {
            assert!(
                (orig - deq).abs() < 0.2,
                "dequantized weight diverged at {i}: orig={orig} deq={deq}"
            );
        }
    }

    /// A quantized MLP converts to a dense Standard MLP with matching forward.
    #[test]
    fn mlp_variant_conversion_preserves_forward() {
        let (hidden, inter) = (32i64, 48i64);
        let mk = |seed: f32, rows: i64, cols: i64| {
            let data: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32 * seed).sin()) * 0.1)
                .collect();
            MxArray::from_float32(&data, &[rows, cols]).unwrap()
        };
        let build = |w: &MxArray| {
            let (q, s, b) = quantize_affine(w, 32, 4);
            QuantizedLinear::new(q, s, Some(b), None, 32, 4, "affine".to_string())
        };
        let mut mlp = MLPVariant::Quantized {
            gate_proj: build(&mk(0.211, inter, hidden)),
            up_proj: build(&mk(0.157, inter, hidden)),
            down_proj: build(&mk(0.359, hidden, inter)),
        };

        let x_data: Vec<f32> = (0..2 * hidden).map(|i| ((i as f32 * 0.093).cos()) * 0.5).collect();
        let x = MxArray::from_float32(&x_data, &[2, hidden]).unwrap();

        let y_quant = to_vec(&mlp.forward(&x).unwrap());
        assert!(mlp.dequantize_to_standard().unwrap());
        assert!(!mlp.is_quantized());
        let y_dense = to_vec(&mlp.forward(&x).unwrap());

        for (i, (a, c)) in y_quant.iter().zip(y_dense.iter()).enumerate() {
            assert!(
                (a - c).abs() <= 1e-3 + 1e-2 * c.abs(),
                "MLP forward diverged at {i}: quantized={a} dense={c}"
            );
        }
    }
}
