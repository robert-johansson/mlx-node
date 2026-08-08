//! Standalone two-model draft support for Qwen3.5 dense speculative
//! decoding (genmlx-orsr).
//!
//! Unlike the in-checkpoint MTP head (`Qwen3_5MTPModule`, chained off the
//! TARGET's hidden states) and unlike gemma4's assistant/DSpark drafts
//! (target-coupled heads), the draft here is an ordinary, separately
//! trained smaller checkpoint of the same family — e.g. a SCI-tuned
//! Qwen3.5-0.8B proposing for a SCI-tuned 4B. It runs its own full
//! forward over its own flat caches; only token ids and logits connect
//! it to the target, which is exactly the [`DsparkStepper`]
//! (`crate::engine::backend`) trait boundary.
//!
//! The loader is deliberately RESTRICTED: dense, text-only, non-quantized
//! (bf16/fp16/fp32) checkpoints. Vision or MoE drafts are refused with a
//! clear error rather than half-supported — a draft exists to be small
//! and fast, and SCI's use case (the only driver today) is a small dense
//! fine-tune. Quantized drafts are a plausible follow-up; refusing them
//! today keeps the apply-weights path on the one configuration this
//! loader is validated on.

use std::path::Path;

use napi::Result;
use napi::bindgen_prelude::Error;
use serde_json::Value;
use tracing::info;

use super::model::Qwen35Inner;
use super::persistence;
use crate::models::quant_dispatch::select_quantization_block;

/// A loaded standalone draft model plus its speculative bookkeeping.
pub(crate) struct Qwen35Draft {
    /// The draft's full model state (embedding, layers, caches, …).
    /// A second `Qwen35Inner` is heavier than a bespoke struct but reuses
    /// the family's forward/prefill/cache code verbatim — the same
    /// trade gemma4 declined only because its drafts are architecturally
    /// coupled heads, which this one is not.
    pub model: Box<Qwen35Inner>,
    /// Checkpoint tensor bytes, for cache-limit accounting
    /// (mirrors `Gemma4Draft::weight_bytes`).
    pub weight_bytes: u64,
    /// Number of tokens of committed history currently represented in the
    /// draft's caches. The stepper keeps this equal to the target's
    /// committed position at every cycle boundary.
    pub committed_len: usize,
}

/// Load a standalone draft checkpoint for two-model speculation.
///
/// `target_vocab_size` guards the one compatibility requirement exact
/// verification has: draft and target must share the vocabulary (token
/// ids are the only thing that crosses the verify boundary).
pub(crate) fn load_draft(
    draft_path: &Path,
    target_vocab_size: i32,
) -> Result<Qwen35Draft> {
    if !draft_path.is_dir() {
        return Err(Error::from_reason(format!(
            "draftModelPath is not a directory: {}",
            draft_path.display()
        )));
    }
    let config_path = draft_path.join("config.json");
    let config_data = std::fs::read_to_string(&config_path).map_err(|e| {
        Error::from_reason(format!(
            "Failed to read draft config {}: {e}",
            config_path.display()
        ))
    })?;
    let raw: Value = serde_json::from_str(&config_data)
        .map_err(|e| Error::from_reason(format!("Failed to parse draft config: {e}")))?;

    let model_type = raw.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    if model_type.contains("moe") {
        return Err(Error::from_reason(format!(
            "draft model_type {model_type:?} is MoE — only dense Qwen3.5 drafts are supported"
        )));
    }
    if select_quantization_block(&raw)?.is_some() {
        return Err(Error::from_reason(
            "quantized draft checkpoints are not supported yet — use a bf16 export \
             (the draft loader is validated on dense bf16 only)",
        ));
    }

    let mut config = persistence::parse_config(&raw)?;
    // The draft proposes on the flat path only; paged attention, the MTP
    // head, and vision have no role in drafting.
    config.use_block_paged_cache = Some(false);
    config.n_mtp_layers = 0;

    persistence::prewarm_draft_checkpoint(draft_path);
    let raw_params = persistence::load_draft_tensors(draft_path)?;
    if raw_params.keys().any(|n| n.starts_with("visual.")) {
        return Err(Error::from_reason(
            "draft checkpoint carries visual.* tensors — vision drafts are not supported",
        ));
    }
    if config.vocab_size != target_vocab_size {
        return Err(Error::from_reason(format!(
            "draft vocab_size {} != target vocab_size {} — exact speculative \
             verification requires a shared vocabulary",
            config.vocab_size, target_vocab_size
        )));
    }

    let (inner, weight_bytes) =
        persistence::build_draft_inner(draft_path, config, raw_params)?;

    info!(
        "Loaded standalone draft model from {} ({:.2} GB, {} layers)",
        draft_path.display(),
        weight_bytes as f64 / (1u64 << 30) as f64,
        inner.config.num_layers,
    );
    Ok(Qwen35Draft {
        model: Box::new(inner),
        weight_bytes,
        committed_len: 0,
    })
}
