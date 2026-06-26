//! Tier-2 branchable-cache round-trip on the REAL qwen3_next MoE model
//! (bean mlx-19wy / P1 — the end-to-end prove-or-kill).
//!
//! Forks the model-internal cache after a prefix, drives two INDEPENDENT
//! branches, and asserts each branch's per-step continuation is logit-identical
//! to the LINEAR (Tier-1) continuation on the SAME token — and that the branches
//! stay isolated from the parent advancing and from each other.
//!
//! IMPORTANT: the equality bar is a SAME-PER-STEP linear continuation
//! (`forwardWithCache` single-token == `forwardBranch` single-token, both ->
//! `forward_inner`), NOT a from-scratch chunked recompute (which diverges by
//! GDN bf16 ULP on long prefixes — see the feasibility synthesis, bean mlx-6k9c).
//!
//! Gated on `MLX_TEST_MOE_MODEL_PATH` and `#[ignore]` so plain `cargo test`
//! stays green. Run:
//!
//! ```shell
//! MLX_TEST_MOE_MODEL_PATH=/path/to/Qwen3-Coder-Next-4bit/snapshots/<hash> \
//!   cargo test -p mlx-core --test qwen3_5_moe_tier2_branch_roundtrip \
//!   -- --ignored --nocapture
//! ```

use std::path::Path;

use mlx_core::array::{DType, MxArray};
use mlx_core::models::qwen3_5_moe::model::Qwen3_5MoeModel;

/// Load the model (the `load` is async) on a throwaway runtime, then return it
/// to the caller's plain thread. CRITICAL: the per-step forward surface is SYNC
/// (`send_and_block` -> `blocking_recv`), which panics if called from within a
/// Tokio async context — so the test body must NOT be `#[tokio::test]`; it calls
/// the sync methods after `block_on` has returned. Returns `None` to skip.
fn load_model_or_skip() -> Option<Qwen3_5MoeModel> {
    let Ok(model_path) = std::env::var("MLX_TEST_MOE_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_MOE_MODEL_PATH unset");
        return None;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_MOE_MODEL_PATH does not exist: {model_path}"
    );
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");
    let model = rt
        .block_on(Qwen3_5MoeModel::load(model_path))
        .expect("failed to load qwen3_next MoE model");
    // `rt` is dropped here, but the model owns its own dedicated std::thread, so
    // the sync command path keeps working without an active async runtime.
    Some(model)
}

/// Build a `[1, N]` int32 id tensor.
fn ids(tokens: &[i32]) -> MxArray {
    MxArray::from_int32(tokens, &[1, tokens.len() as i64]).expect("from_int32")
}

/// Read `[1, 1, vocab]` logits as a flat f32 vec.
fn logits_vec(logits: &MxArray) -> Vec<f32> {
    let f = logits.astype(DType::Float32).expect("astype f32");
    f.eval();
    f.to_float32().expect("to_float32").to_vec()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .expect("non-empty logits")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The Tier-2 correctness criterion: the greedy (argmax) token must match —
/// that is what regenerate / token-MCMC actually consume. Logit VALUES may
/// differ slightly because the 4-bit MoE CUDA path is not bit-deterministic
/// run-to-run (atomic expert reductions); that floor is measured separately
/// from two independent forks. `band` bounds how far the value diff may stray,
/// catching a SYSTEMATIC fork-copy bias while tolerating that nondeterminism.
fn assert_continuation_eq(got: &[f32], reference: &[f32], band: f32, label: &str) {
    let d = max_abs_diff(got, reference);
    let (gm, rm) = (argmax(got), argmax(reference));
    eprintln!("[{label}] argmax {gm} vs {rm}, max_abs_diff {d} (band {band})");
    assert_eq!(gm, rm, "[{label}] greedy token differs (max_abs_diff={d})");
    assert!(
        d <= band,
        "[{label}] logits differ by {d} > band {band} — systematic fork bias?"
    );
}

// Plain `#[test]` (NOT `#[tokio::test]`): the SYNC per-step forward surface uses
// `blocking_recv`, which panics inside a Tokio async context. The async `load`
// is driven on a throwaway runtime inside `load_model_or_skip`.
#[test]
#[ignore = "needs MLX_TEST_MOE_MODEL_PATH pointing to a real qwen3_next MoE checkpoint"]
fn tier2_branch_roundtrip_isolation_on_real_moe() {
    let Some(model) = load_model_or_skip() else {
        return;
    };

    // A >1-token prefix (drives chunked_prefill). Arbitrary valid ids; semantic
    // coherence is irrelevant — we only compare branch vs linear continuations.
    let prefix: Vec<i32> = (100..164).collect(); // 64-token prefix
    let x: i32 = 12345; // continuation token compared across paths
    let y: i32 = 54321; // second token, used to advance the linear path

    // Build the shared prefix in the Tier-1 model-internal cache.
    model.init_caches().expect("init_caches");
    model
        .forward_with_cache(&ids(&prefix), true)
        .expect("prefill prefix");

    // Fork TWO independent branches at the prefix (O(prefix) deep copy each).
    let id_a = model.branch_cache().expect("branch A");
    let id_b = model.branch_cache().expect("branch B");
    assert_ne!(id_a, id_b, "branch ids must be distinct");

    // --- Fork correctness + the nondeterminism floor --------------------
    // Branch A and branch B are INDEPENDENT forks of the same prefix; given the
    // same token X they must reach the same greedy token. Their value diff is
    // the pure run-to-run nondeterminism floor of the MoE CUDA path (no
    // fork-vs-direct asymmetry — both are forks).
    let l_a = logits_vec(&model.forward_branch(id_a, &ids(&[x])).expect("forward A,X"));
    let l_b = logits_vec(&model.forward_branch(id_b, &ids(&[x])).expect("forward B,X"));
    // Linear (Tier-1) ground truth: prefix + X via the SAME per-step path.
    let l_lin = logits_vec(&model.forward_with_cache(&ids(&[x]), true).expect("linear X"));

    let floor = max_abs_diff(&l_a, &l_b);
    eprintln!("[nondeterminism floor] two independent forks, same token: max_abs_diff {floor}");
    // Fork-vs-direct must not exceed the fork-vs-fork floor by more than a small
    // margin. A systematic fork-copy error would show up here as a large,
    // consistent bias (both forks biased the SAME way vs direct, yet close to
    // each other) — which this bound catches.
    let band = 3.0 * floor + 0.5;
    assert_continuation_eq(&l_a, &l_lin, band, "branch A vs linear (prefix+X)");
    assert_continuation_eq(&l_b, &l_lin, band, "branch B vs linear (prefix+X)");

    // --- Isolation across a divergent parent advance --------------------
    // Advance the LINEAR path to prefix+X+Y, then continue branch A with Y to
    // prefix+X+Y. Branch A must match the linear continuation even though the
    // parent advanced AND branch B was forwarded in between — i.e. branches are
    // isolated from the parent and from each other, across multiple steps.
    let l_lin_xy = logits_vec(&model.forward_with_cache(&ids(&[y]), true).expect("linear Y"));
    let l_a_xy = logits_vec(&model.forward_branch(id_a, &ids(&[y])).expect("forward A,Y"));
    assert_continuation_eq(
        &l_a_xy,
        &l_lin_xy,
        band,
        "branch A vs linear (prefix+X+Y) — isolation",
    );

    // Divergence sanity: prefix+X and prefix+X+Y must differ by more than noise.
    assert!(
        max_abs_diff(&l_a, &l_lin_xy) > floor,
        "prefix+X and prefix+X+Y produced near-identical logits — cache not advancing"
    );

    // --- Lifecycle ------------------------------------------------------
    model.dispose_branch(id_a).expect("dispose A");
    model.dispose_branch(id_b).expect("dispose B");
    assert!(
        model.forward_branch(id_a, &ids(&[x])).is_err(),
        "forwardBranch on a disposed id must error"
    );

    eprintln!(
        "Tier-2 branch round-trip: PASS on the real qwen3_next MoE \
         (argmax-exact fork + isolation; MoE logit nondeterminism floor {floor})."
    );
}
