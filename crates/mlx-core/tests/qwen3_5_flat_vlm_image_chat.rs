//! Correctness gate for the dense Qwen3.5 **flat (non-paged) vision** path —
//! the CUDA/Linux and paging-off image-turn route (bean genmlx-9v44).
//!
//! Dense image turns historically ran ONLY on the block-paged backend, which is
//! Metal-gated (`mlx_metal_is_available()`), so on a non-Metal build the image
//! turn errored at dispatch. `vision_flat_whole_turn_core` adds the flat arm:
//! same vision merge (`vlm_prepare_vision_features`) and plain-AR decode, but it
//! writes the flat `self.caches` and rotates decode queries at the
//! image-compressed M-RoPE position (`physical_position + rope_deltas`) instead
//! of using the paged pool.
//!
//! This test FORCES the flat path on ANY host by cloning the checkpoint with a
//! config-only patch (`use_block_paged_cache: false`), so it is a genuine
//! flat-path gate on Metal and CUDA alike (not just where Metal is absent). It
//! proves three independent properties (mirroring `qwen3_5_vl_image_chat.rs`,
//! since there is no flat-vs-paged byte ground truth on a non-Metal host):
//!   * COHERENCE      — flat(image) produces real (non-empty) output.
//!   * DETERMINISM    — flat(image) at T=0 is byte-identical run-to-run.
//!   * IMAGE-DEPENDENCE — flat(image) differs from flat(no-image), so the vision
//!     features actually reach generation.
//!
//! Single fresh turn only: the flat image core is NON-continuable (it resets
//! caches on the way out), matching the sync-only scope of genmlx-9v44. Multi-
//! turn image continuation / streaming are follow-ups (genmlx-djno).
//!
//! Gated on `MLX_TEST_QWEN35_VL_MODEL_PATH` (a dense Qwen3.5-VL checkpoint) and a
//! test image (`MLX_TEST_VLM_IMAGE_PATH` else `examples/ocr.png`). A plain
//! `cargo test --ignored` without the env vars early-returns before any model
//! load. The 0.8b checkpoint runs but is weakly vision-conditioning; the 4B is
//! the smallest dense VL checkpoint that genuinely conditions on the image.
//!
//! Run:
//!   MLX_TEST_QWEN35_VL_MODEL_PATH=$HOME/code/mlx/models/Qwen3.5-4B-MLX-4bit/snapshots/<h> \
//!   MLX_TEST_VLM_IMAGE_PATH=genmlx.png \
//!     cargo test -p mlx-core --test qwen3_5_flat_vlm_image_chat -- --ignored --nocapture

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::ChatConfig;
use mlx_core::models::qwen3_5::model::Qwen3_5Model;
use mlx_core::tokenizer::ChatMessage;
use napi::bindgen_prelude::Uint8Array;

/// Clone a model dir into the workspace target, symlinking every weight file and
/// patching ONLY `config.json` to force `use_block_paged_cache: false`. The flat
/// copy therefore loads with `paged_adapter = None` on every host, so the image
/// turn takes `vision_flat_whole_turn_core`.
fn clone_model_dir_flat(src: &Path) -> Result<PathBuf, String> {
    let pid = std::process::id();
    let workspace_target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = std::env::var("CARGO_MANIFEST_DIR")
                .expect("CARGO_MANIFEST_DIR must be set when running cargo test");
            let mut p = PathBuf::from(manifest);
            p.pop();
            p.pop();
            p.join("target")
        });

    let dst = workspace_target.join(format!("flat-vlm-image-{pid}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

    let read_dir = fs::read_dir(src).map_err(|e| format!("read_dir({}): {e}", src.display()))?;
    for entry in read_dir {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_file() {
            if entry.file_name() == "config.json" {
                fs::copy(&from, &to)
                    .map_err(|e| format!("copy({} -> {}): {e}", from.display(), to.display()))?;
            } else {
                std::os::unix::fs::symlink(&from, &to)
                    .map_err(|e| format!("symlink({} -> {}): {e}", from.display(), to.display()))?;
            }
        }
    }

    let cfg_path = dst.join("config.json");
    let raw = fs::read_to_string(&cfg_path).map_err(|e| format!("read config.json: {e}"))?;
    let mut cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| format!("parse config.json: {e}"))?;
    cfg["use_block_paged_cache"] = serde_json::Value::Bool(false);
    let pretty =
        serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize config.json: {e}"))?;
    fs::write(&cfg_path, pretty).map_err(|e| format!("write config.json: {e}"))?;

    Ok(dst)
}

fn cfg(max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(max_new_tokens),
        temperature: Some(0.0),
        top_k: None,
        top_p: None,
        min_p: None,
        repetition_penalty: None,
        repetition_context_size: None,
        presence_penalty: None,
        presence_context_size: None,
        frequency_penalty: None,
        frequency_context_size: None,
        max_consecutive_tokens: None,
        max_ngram_repeats: None,
        ngram_size: None,
        tools: None,
        reasoning_effort: None,
        thinking_token_budget: Some(32),
        include_reasoning: Some(true),
        report_performance: Some(false),
        reuse_cache: Some(true),
        enable_mtp: None,
        mtp_depth: None,
        mtp_adaptive_depth: None,
    }
}

fn user_msg(content: &str, image: Option<&[u8]>) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        images: image.map(|b| vec![Uint8Array::new(b.to_vec())]),
        audio: None,
    }
}

fn resolve_image_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MLX_TEST_VLM_IMAGE_PATH") {
        let pb = PathBuf::from(p);
        return pb.exists().then_some(pb);
    }
    let pb = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/ocr.png");
    pb.exists().then_some(pb)
}

fn hash8(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn oneline(s: &str) -> String {
    s.replace('\n', "\\n").chars().take(96).collect()
}

const PROMPT: &str = "Describe this image briefly.";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_QWEN35_VL_MODEL_PATH + MLX_TEST_VLM_IMAGE_PATH for a Qwen3.5-VL dense checkpoint + test image"]
async fn qwen3_5_flat_vlm_image_t0() {
    let Ok(model_path) = std::env::var("MLX_TEST_QWEN35_VL_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_QWEN35_VL_MODEL_PATH unset");
        return;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_QWEN35_VL_MODEL_PATH does not exist: {model_path}"
    );
    let Some(image_path) = resolve_image_path() else {
        eprintln!("skipping: no test image (set MLX_TEST_VLM_IMAGE_PATH or add examples/ocr.png)");
        return;
    };
    let image = std::fs::read(&image_path).expect("failed to read test image");

    // Force the flat (non-paged) path on ANY host via a config-only clone.
    let flat_dir = clone_model_dir_flat(Path::new(&model_path)).expect("clone flat model dir");
    let model = Qwen3_5Model::load(flat_dir.to_string_lossy().to_string())
        .await
        .expect("failed to load flat (paging-off) Qwen3.5-VL model");

    let reset = |m: &Qwen3_5Model| {
        tokio::task::block_in_place(|| m.reset_caches()).expect("reset_caches failed");
    };

    // --- Pass 1: one image turn through the flat vision core. ---
    reset(&model);
    let t1 = model
        .chat_session_start(vec![user_msg(PROMPT, Some(&image))], Some(cfg(64)))
        .await
        .expect("flat image turn chat_session_start failed");
    let hash1 = hash8(&t1.raw_text);
    println!(
        "FLAT-IMAGE ntok={} finish={} sha={:016x} :: {}",
        t1.num_tokens,
        t1.finish_reason,
        hash1,
        oneline(&t1.raw_text)
    );

    // COHERENCE: real, non-empty output with a clean finish.
    assert!(t1.num_tokens > 0, "flat image turn produced zero tokens");
    assert!(
        !t1.raw_text.trim().is_empty(),
        "flat image turn produced empty text"
    );
    assert!(
        t1.finish_reason == "stop" || t1.finish_reason == "length",
        "flat image turn unexpected finish_reason: {}",
        t1.finish_reason
    );

    // --- IMAGE-DEPENDENCE: same prompt, NO image, fresh session. ---
    reset(&model);
    let t_noimg = model
        .chat_session_start(vec![user_msg(PROMPT, None)], Some(cfg(64)))
        .await
        .expect("no-image control chat_session_start failed");
    println!(
        "FLAT-NOIMG ntok={} sha={:016x} :: {}",
        t_noimg.num_tokens,
        hash8(&t_noimg.raw_text),
        oneline(&t_noimg.raw_text)
    );
    assert_ne!(
        hash1,
        hash8(&t_noimg.raw_text),
        "flat image output is not image-dependent: identical with and without the \
         image (vision features are not reaching generation through the flat path)"
    );

    // --- DETERMINISM: reset + replay the image turn; must be byte-identical. ---
    reset(&model);
    let t1b = model
        .chat_session_start(vec![user_msg(PROMPT, Some(&image))], Some(cfg(64)))
        .await
        .expect("flat image turn replay failed");
    assert_eq!(
        hash1,
        hash8(&t1b.raw_text),
        "flat image turn is not deterministic at T=0 (run-to-run digest differs)"
    );

    let _ = fs::remove_dir_all(&flat_dir);
}
