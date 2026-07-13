# CLI (`@mlx-node/cli`)

The `mlx` binary is built from `packages/cli/` and exposes three top-level commands: `download`, `convert`, and `launch`.

## `mlx download`

### Models

```bash
mlx download model --model Qwen/Qwen3-0.6B
```

| Flag             | Default           | Purpose                                                |
| ---------------- | ----------------- | ------------------------------------------------------ |
| `-m`, `--model`  | `Qwen/Qwen3-0.6B` | HuggingFace model id                                   |
| `-g`, `--glob`   | —                 | Filename pattern filter (download only matching files) |
| `--set-token`    | —                 | Store HuggingFace credentials                          |
| `-o`, `--output` | —                 | Output directory                                       |

### Datasets

```bash
mlx download dataset
```

Default dataset: `openai/gsm8k`. Parquet inputs are automatically converted to JSONL via `convertParquetToJsonl()`.

| Flag               | Default        | Purpose                |
| ------------------ | -------------- | ---------------------- |
| `-d`, `--dataset`  | `openai/gsm8k` | HuggingFace dataset id |
| `-r`, `--revision` | —              | Dataset revision       |
| `-o`, `--output`   | —              | Output directory       |

## `mlx convert`

The convert command uses `--input` / `--output` (not `--model`).

### Dtype conversion

```bash
mlx convert --input ./model --output ./model-bf16 --dtype bf16
```

### Quantization (affine, default)

```bash
mlx convert --input ./model --output ./model-q --quantize --q-recipe mixed_4_6
```

### Qwen MTP quantization conversion

```bash
mlx convert \
  --input .cache/models/qwen3.6-27b \
  --output .cache/models/qwen3.6-27b-unsloth-nvfp4-mtplx-sidecar \
  --model-type qwen3_5 \
  --quantize --q-mode nvfp4 --q-recipe unsloth \
  --imatrix-path ./imatrix.gguf \
  --q-mtp cyankiwi
```

`--q-mtp cyankiwi` keeps `mtp.fc` and MTP norms BF16 and packs the MTP layer
linears as 4-bit affine group-size 32 tensors with MTPLX-compatible metadata.
Where those quantized tensors land depends on the model family:

- Dense `qwen3_5` — emitted into a separate `mtp.safetensors` sidecar.
- MoE `qwen3_5_moe` — there is **no sidecar**; the MTP tensors are quantized in
  place and stored inline in the main safetensors shards.

`--q-mtp all` additionally quantizes `mtp.fc` (same dense-sidecar / MoE-inline
split). `--q-mtp split` (alias `drafter`) emits a body checkpoint with **no
`mtp.*` tensors** plus a separate `mtp-drafter/` directory in mlx-vlm's
`qwen3_5_mtp` format (bare-keyed, BF16 MTP head); it does not require
`--quantize`/`--q-recipe` and the body may be BF16 or already-quantized.

| Flag               | Purpose                                                                         |
| ------------------ | ------------------------------------------------------------------------------- |
| `-i`, `--input`    | Source model directory (required)                                               |
| `-o`, `--output`   | Output directory (required)                                                     |
| `-d`, `--dtype`    | Target dtype: `float32` / `float16` / `bfloat16`                                |
| `-q`, `--quantize` | Enable quantization                                                             |
| `--q-recipe`       | One of `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`, `qwen3_5`, `unsloth` |
| `--q-mode`         | `affine` (default) or `mxfp8`                                                   |
| `--q-mtp`          | Qwen MTP-quant policy: `off`, `cyankiwi`, `all`, or `split` (alias `drafter`)   |
| `--imatrix-path`   | Path to imatrix file for AWQ pre-scaling                                        |
| `--mmproj`         | Vision-encoder conversion path                                                  |
| `-v`, `--verbose`  | Verbose logging                                                                 |

### GGUF → SafeTensors

```bash
mlx convert --input ./model.gguf --output ./model-mlx
```

Auto-detected by the `.gguf` extension. Supports BF16, F16, F32, Q4_0, Q4_1, Q8_0 source quantization types.

### Model-type auto-detection

The converter auto-detects model families and applies family-specific sanitization passes:

- `qwen3_5`, `qwen3_5_moe`
- `gemma4`, `gemma4_unified`
- `paddleocr-vl`, `qianfan-ocr`
- `pp-lcnet-ori`, `uvdoc`

Sharded models are also supported (parses `model.safetensors.index.json`).

Foreign weight formats: Paddle `.pdiparams`, PyTorch `.pkl`.

## `mlx launch claude`

Launches the local `@mlx-node/server` and spawns Claude Code against it — the entry point for using MLX-Node as a Claude Code backend. The "serve" terminology in commit messages refers to internal server components only; there is no `mlx serve` command.

## `mlx agent`

A fully-local coding agent — MLX-Node's first all-in-one local agent. It embeds the [pi coding agent](https://www.npmjs.com/org/earendil-works) (`@earendil-works/*`) and serves every model turn through in-process `@mlx-node/lm` inference. There is no HTTP server, no external process, and no API keys: prompts, tools, and weights all stay on the machine. Requires Node.js ≥ 22.19.

```bash
mlx agent                       # interactive session (first run: setup wizard)
mlx agent -c                    # resume the most recent session
mlx agent -p 'summarize this repo' --no-session   # headless / print mode
mlx agent --mode json           # newline-delimited JSON events (for scripting)
mlx agent --models-dir ./models # use a specific local models directory
```

Almost every flag belongs to pi and is forwarded verbatim; `mlx agent` only handles the options below before handing off.

### Model selection and first-run wizard

`mlx agent` discovers local models under the resolved models directory (`--models-dir <dir>`, else `MLX_MODELS_DIR`, else `modelsDir` in `~/.mlx-node/config.json`, else `~/.mlx-node/models`). A dash-leading path must use the `--models-dir=<dir>` form so it is not mistaken for another flag.

On a fresh run (no explicit `--model`/`--provider`/session flag), it injects the first discovered local model — honoring a persisted `/model` pick when that model is still present — so ambient cloud credentials (e.g. a stray `GROQ_API_KEY`) never win over the local model this command promises.

When no local model exists, an interactive terminal shows a first-run wizard over a curated catalog and downloads the choice via `mlx download model`. In a non-interactive shell it prints the equivalent `mlx download model` commands instead. The catalog:

| Model                 | HuggingFace repo                                 | Size   | Notes                        |
| --------------------- | ------------------------------------------------ | ------ | ---------------------------- |
| Qwen3.6-27B (default) | `Brooooooklyn/Qwen3.6-27B-NVFP4-mlx`             | ~22 GB | Best tool use — recommended  |
| Qwen-AgentWorld-35B   | `Brooooooklyn/Qwen-AgentWorld-35B-A3B-nvfp4-mlx` | ~23 GB | Agent-tuned MoE, fast decode |
| Gemma-4-26B-A4B       | `Brooooooklyn/Gemma-4-26B-A4B-NVFP4-mlx`         | ~19 GB | MoE, fast decode             |

A more compact Gemma-4-12B entry (mxfp4 MLP + mxfp8 attention, ~9 GB, for smaller machines) is coming and will appear in the wizard once it is published.

### Config home

pi's config home is `~/.mlx-node/agent` (override with `PI_CODING_AGENT_DIR`) — it holds `settings.json`, saved sessions, extensions, skills, prompts, and themes. A project-local `.pi/` directory still works for per-repo overrides. `mlx agent` also seeds `PI_SKIP_VERSION_CHECK=1` and `MLX_PAGED_PREFILL_CHUNK_SIZE=2048` (both only when unset) so long prompts keep bounded time-to-first-token on the default paged path.

### Permission gate

pi has no permission system of its own, so `mlx agent` installs a safety gate: every `bash`, `write`, and `edit` tool call must be approved before it runs. In an interactive session it prompts (`Yes` / `Always (this session)` / `No`). Without an attached UI — headless print or `--mode json` runs — it blocks those tools unless you opt in with `MLX_AGENT_AUTO_APPROVE=1`:

```bash
MLX_AGENT_AUTO_APPROVE=1 mlx agent -p 'run the test suite and report failures' --no-session
```

### Extensions and skills

The leading positional commands pass through to pi and manage what lives under the agent config home:

```bash
mlx agent install <source>   # add a pi extension / theme / skill
mlx agent remove <name>      # remove one (alias: uninstall)
mlx agent list               # list installed
mlx agent config             # edit which are enabled
```

`mlx agent update` is intentionally blocked (it maps to pi's npm self-update, which would fight the installed `@mlx-node/cli`); update `@mlx-node/cli` through your package manager instead. `mlx agent -h`/`--help` prints the mlx options above and then pi's full flag list. `mlx agent --version`/`-v` and `mlx agent --export <session>` are answered by pi directly — no local model needed, so the first-run wizard stays out — and `--version` prints the embedded pi version, not `@mlx-node/cli`'s (`mlx --version`).
