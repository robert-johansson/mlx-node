use crate::array::MxArray;
use crate::nn::Activations;
use napi::bindgen_prelude::*;

use super::quantized_linear::QuantizedSwitchLinear;
use super::switch_linear::SwitchLinear;

/// A projection layer that can be either standard or quantized.
pub enum SwitchProj {
    Standard(SwitchLinear),
    Quantized(QuantizedSwitchLinear),
}

impl SwitchProj {
    fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        match self {
            SwitchProj::Standard(l) => l.forward(x, indices, sorted),
            SwitchProj::Quantized(l) => l.forward(x, indices, sorted),
        }
    }
}

/// SwitchGLU: Expert-indexed SwiGLU MLP using SwitchLinear (gather_mm) or
/// QuantizedSwitchLinear (gather_qmm).
///
/// Each of the three projections (gate, up, down) has per-expert weights.
/// Uses expand_dims to broadcast input across expert slots, matching
/// the mlx-lm Python implementation.
///
/// When `indices.size >= 64`, tokens are sorted by expert index for better
/// `gather_mm`/`gather_qmm` memory locality (1.5-3x speedup).
pub struct SwitchGLU {
    pub(crate) gate_proj: SwitchProj,
    pub(crate) up_proj: SwitchProj,
    pub(crate) down_proj: SwitchProj,
}

/// Result of sorting tokens by expert index for optimized gather_mm.
struct GatherSortResult {
    x_sorted: MxArray,   // Sorted input tokens
    idx_sorted: MxArray,  // Sorted expert indices (flat)
    inv_order: MxArray,   // Inverse permutation to unsort
}

/// Sort tokens by expert index for better gather_mm memory locality.
///
/// Ports Python mlx-lm `_gather_sort`:
/// ```python
/// *_, M = indices.shape
/// indices = indices.flatten()
/// order = mx.argsort(indices)
/// inv_order = mx.argsort(order)
/// return x.flatten(0, -3)[order // M], indices[order], inv_order
/// ```
fn gather_sort(x: &MxArray, indices: &MxArray) -> Result<GatherSortResult> {
    let idx_shape = indices.shape()?;
    let m = *idx_shape.last().ok_or_else(|| Error::from_reason("empty indices"))?;

    // Flatten indices: [ne, k] -> [ne*k]
    let flat_indices = indices.reshape(&[-1])?;

    // Sort order: argsort gives permutation that sorts by expert index
    let order = flat_indices.argsort(Some(-1))?;

    // Inverse order: argsort(argsort(x)) gives the inverse permutation
    let inv_order = order.argsort(Some(-1))?;

    // Sorted indices
    let idx_sorted = flat_indices.take(&order, 0)?;

    // Python: x.flatten(0, -3)[order // M]
    // x is [ne, 1, 1, D]. flatten(0, -3) collapses dims 0 through ndim-3:
    //   [ne, 1, 1, D] -> [ne*1, 1, D] = [ne, 1, D]
    // Then [order // M] indexes dim 0 -> [ne*k, 1, D]
    let x_shape = x.shape()?;
    let d = *x_shape.last().unwrap();

    // Flatten all dims except the last 2: [..., 1, D] -> [ne, 1, D]
    let x_flat = x.reshape(&[-1, 1, d])?;

    // order // M: maps from sorted position -> original token index
    let m_scalar = MxArray::scalar_int(m as i32)?;
    let token_indices = order.floor_divide(&m_scalar)?;

    // x_sorted: [ne*k, 1, D] indexed by token_indices
    let x_sorted = x_flat.take(&token_indices, 0)?;

    Ok(GatherSortResult {
        x_sorted,
        idx_sorted,
        inv_order,
    })
}

/// Unsort the output back to original token order.
fn scatter_unsort(x: &MxArray, inv_order: &MxArray, orig_shape: &[i64]) -> Result<MxArray> {
    // x is [ne*k, ...], inv_order is [ne*k]
    let unsorted = x.take(inv_order, 0)?;

    // Unflatten back to [ne, k, ...]
    let x_shape = unsorted.shape()?;
    let mut new_shape = orig_shape.to_vec();
    for &dim in &x_shape[1..] {
        new_shape.push(dim);
    }
    unsorted.reshape(&new_shape)
}

impl SwitchGLU {
    /// Create with standard (non-quantized) SwitchLinear layers.
    pub fn new(input_dims: u32, hidden_dims: u32, num_experts: u32) -> Result<Self> {
        let gate_proj = SwitchLinear::new(input_dims, hidden_dims, num_experts)?;
        let up_proj = SwitchLinear::new(input_dims, hidden_dims, num_experts)?;
        let down_proj = SwitchLinear::new(hidden_dims, input_dims, num_experts)?;

        Ok(Self {
            gate_proj: SwitchProj::Standard(gate_proj),
            up_proj: SwitchProj::Standard(up_proj),
            down_proj: SwitchProj::Standard(down_proj),
        })
    }

    /// Create with quantized SwitchLinear layers.
    pub fn new_quantized(
        gate_proj: QuantizedSwitchLinear,
        up_proj: QuantizedSwitchLinear,
        down_proj: QuantizedSwitchLinear,
    ) -> Self {
        Self {
            gate_proj: SwitchProj::Quantized(gate_proj),
            up_proj: SwitchProj::Quantized(up_proj),
            down_proj: SwitchProj::Quantized(down_proj),
        }
    }

    /// Forward pass with optional sorted expert indices for memory locality.
    ///
    /// Matches Python mlx-lm SwitchGLU:
    ///   x = expand_dims(x, (-2, -3))   # (ne, D) -> (ne, 1, 1, D)
    ///   ... gather_mm ops (sorted or unsorted) ...
    ///   if do_sort: x = _scatter_unsort(x, inv_order, indices.shape)
    ///   return x.squeeze(-2)            # final squeeze: (ne, k, 1, D) -> (ne, k, D)
    ///
    /// # Arguments
    /// * `x` - Input tensor [B*T, D]
    /// * `indices` - Expert indices [B*T, k] (int32)
    ///
    /// # Returns
    /// Output tensor [B*T, k, D]
    pub fn forward(&self, x: &MxArray, indices: &MxArray) -> Result<MxArray> {
        let x_shape = x.shape()?;
        let ne = x_shape[0];
        let d = x_shape[1];
        let idx_shape = indices.shape()?;

        // Expand x: (ne, D) -> (ne, 1, 1, D) for gather_mm broadcasting
        let x_expanded = x.reshape(&[ne, 1, 1, d])?;

        // Sort tokens by expert index when we have enough tokens (>= 64)
        let do_sort = indices.size()? >= 64;

        let out = if do_sort {
            let sorted = gather_sort(&x_expanded, indices)?;

            // Keep sorted indices 1D [ne*k] — matches Python where idx is flattened
            let idx = &sorted.idx_sorted;

            let gate_out = self.gate_proj.forward(&sorted.x_sorted, idx, true)?;
            let up_out = self.up_proj.forward(&sorted.x_sorted, idx, true)?;

            let activated = Activations::swiglu(&gate_out, &up_out)?;

            let result = self.down_proj.forward(&activated, idx, true)?;

            // Unsort back to original order
            scatter_unsort(&result, &sorted.inv_order, &idx_shape)?
        } else {
            let gate_out = self.gate_proj.forward(&x_expanded, indices, false)?;
            let up_out = self.up_proj.forward(&x_expanded, indices, false)?;

            let activated = Activations::swiglu(&gate_out, &up_out)?;

            self.down_proj.forward(&activated, indices, false)?
        };

        // Final squeeze: (ne, k, 1, D) -> (ne, k, D) — matches Python's x.squeeze(-2)
        out.squeeze(Some(&[-2]))
    }

    // Weight accessors — only work for standard (non-quantized) layers
    pub fn set_gate_proj_weight(&mut self, w: &MxArray) {
        if let SwitchProj::Standard(ref mut l) = self.gate_proj {
            l.set_weight(w);
        }
    }
    pub fn set_up_proj_weight(&mut self, w: &MxArray) {
        if let SwitchProj::Standard(ref mut l) = self.up_proj {
            l.set_weight(w);
        }
    }
    pub fn set_down_proj_weight(&mut self, w: &MxArray) {
        if let SwitchProj::Standard(ref mut l) = self.down_proj {
            l.set_weight(w);
        }
    }
    pub fn get_gate_proj_weight(&self) -> MxArray {
        match &self.gate_proj {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }
    pub fn get_up_proj_weight(&self) -> MxArray {
        match &self.up_proj {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }
    pub fn get_down_proj_weight(&self) -> MxArray {
        match &self.down_proj {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }

    /// Check if this SwitchGLU uses quantized layers.
    pub fn is_quantized(&self) -> bool {
        matches!(&self.gate_proj, SwitchProj::Quantized(_))
    }
}
