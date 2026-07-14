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

### Custom tools

A pi extension file can define its own tools with `defineTool` + `pi.registerTool`; `examples/echo-tool.ts` is the in-tree template (parameter schema, async execute, and an image-returning tool). Three properties matter for instrument-style setups that replace the coding toolset entirely:

- `--no-builtin-tools --extension <file.ts>` yields a session whose ONLY tools are the extension's — no bash/write/edit.
- Custom tools are never gated: the permission gate intercepts `bash`/`write`/`edit` by name, so extension tools run without prompts and without `MLX_AGENT_AUTO_APPROVE`, headless included. A tool needing its own approval flow implements it inside `execute`.
- Headless `-p` runs drive multi-turn tool loops to completion (call → result → continuation), not just one turn.

A tool result may carry images: return `{ type: 'image', data: <base64>, mimeType: 'image/png' }` parts alongside text. On a VLM the provider delivers the pixels to the model (internally hoisted onto a synthetic user turn to match the Qwen-VL trained format); text-only models reject image-bearing turns with a typed error.

```bash
mlx agent --model mlx/<model> --no-builtin-tools --extension examples/echo-tool.ts \
  -p "call echo_tool with the word hello, then tell me what it returned"
```

### Observing and driving a session

Two channels expose a running session's mind (think blocks, tool calls, streaming text); both are verified in this tree.

**`--mode rpc` — delta-level, and the administration channel.** The agent reads JSON commands on stdin and emits JSON lines on stdout: command `response`s plus `AgentSessionEvent`s as they occur. Verified event flow for one tool-using prompt:

```
response(prompt) → agent_start → turn_start
  → message_start/message_update*/message_end   (assistant; updates carry
     text_start/text_delta/text_end, thinking_start/thinking_delta/thinking_end,
     toolcall_start/toolcall_delta/toolcall_end sub-events)
  → tool_execution_start {toolName, args} → tool_execution_end
  → message_start/message_end                    (toolResult)
→ turn_end → turn_start … → agent_end → agent_settled
```

Commands include `prompt`, `steer`, `abort`, `new_session`, `set_model`, `set_thinking_level`, `compact`, `get_state` — enough to script an entire administration without the TUI. Example: `mlx agent --model mlx/<model> --mode rpc --no-builtin-tools --extension tools.ts`, then write `{"type":"prompt","message":"..."}` lines to stdin.

**Session JSONL — message-level, attach from outside.** Sessions persist under `~/.mlx-node/agent/sessions/` as JSONL, appended once per COMPLETED message (`appendFileSync`), so a live tail sees each think block and each tool call the moment it finishes — a tool call lands before its result executes. Stable record shape: every line has `{type, id, parentId, timestamp}` (ISO-8601); `type:"message"` lines carry a pi `message` with `role` `user` / `assistant` / `toolResult`; assistant `content` parts are `{type:"thinking", thinking}`, `{type:"text", text}`, `{type:"toolCall", id, name, arguments}` plus `stopReason` and `usage`; toolResult messages carry `toolCallId`, `toolName`, `isError`, and text/image parts. `parentId` chains records into pi's session tree (forks share the file).

`examples/watch-session.ts` is the attach script — point it at a session file or the sessions directory and it prints each think block, tool call, and tool result as records land:

```bash
oxnode examples/watch-session.ts ~/.mlx-node/agent/sessions/<project-dir>/
```

Note: models emit `<think>` blocks per their thinking level — set `--thinking low|medium|high` if the mind panel should have deliberation to show.

**Per-turn token accounting (measurement-grade).** Every assistant record's `usage` carries engine-real counts, not estimates: `input` (prompt tokens MINUS the KV-cache-served prefix — add `cacheRead` back for the full prompt length), `output` (completion tokens), `reasoning` (thinking tokens: sampled-token count up to the `</think>` boundary, computed by the engine's `ReasoningTracker` during decode; a subset of `output`; 0 when no think block), and `totalTokens` (full prompt + completion — this drives pi's auto-compaction). The same numbers ride `message_end` rpc events. Verified: the same prompt at `--thinking off` vs `medium` yields `reasoning` 0 vs >0 with `output` differing accordingly.

### Clean-room persona (replacing the coding-assistant frame)

`--system-prompt` fully REPLACES pi's coding-assistant prompt (the separate `--append-system-prompt` exists for prepend-style use). In the custom-prompt branch pi also ignores tool snippets and guidelines, so extension tools cannot add prompt text. Two caveats, verified byte-for-byte:

- pi UNCONDITIONALLY appends two lines to every system prompt: `Current date: YYYY-MM-DD` and `Current working directory: <cwd>`. The composed message is byte-identical to your text **plus exactly that suffix** — removing it would require forking pi. Launch from a neutral cwd if the path would break the persona frame.
- Project context (`AGENTS.md`/`CLAUDE.md`) is appended INSIDE the system prompt unless `-nc`/`--no-context-files` is passed — with a marked AGENTS.md in cwd, the marker appears without `-nc` and is absent with it.

The full clean-room incantation, plus the verification hook (`MLX_AGENT_DUMP_SYSTEM=<path>` writes the exact composed system prompt the model receives, each turn):

```bash
MLX_AGENT_DUMP_SYSTEM=/tmp/system-as-received.txt mlx agent \
  --model mlx/<model> \
  --system-prompt "$(cat persona.txt)" \
  -nc --no-skills --no-extensions -e participant-tools.ts \
  --no-builtin-tools
```

(`--no-extensions` disables discovery from the config home — third-party extensions could otherwise register tools; explicit `-e` paths still load.)

### Temperature and reproducibility

`MLX_AGENT_TEMPERATURE=<n>` overrides the launch preset's sampling temperature per run (an explicit pi-side option still wins). The native sampler treats temperature ≤ 1e-6 as greedy argmax, so `MLX_AGENT_TEMPERATURE=0` selects deterministic decoding. There is currently **no sampling seed on the chat path** — temp-0 is the only reproducible mode; seeded sampling at temp > 0 would be a small native feature (the RNG plumbing exists training-side as `GrpoEngineConfig.seed`).

**Reproducibility on Thor/CUDA, as measured (2026-07-14, Ornith-1.0-35B-8bit):** `-p` at temperature 0 is byte-repeatable — two identical runs (thinking medium, ~150 generated tokens) produced byte-identical stdout. This rests on more than the paired test: since the qmm_sm80 cp_async race fix (mlx `caadb9da7`), the MoE forward on this hardware is bit-deterministic at the logits level (30/30 identical prefill+decode across runs; guard test `qmm_determinism_test` in genmlx), and greedy argmax plus the deterministic penalty transforms preserve that through sampling. Caveats that bound the claim:

- The system prompt embeds the **current date and cwd** — runs are prompt-identical only same-day and same-cwd. Cross-day divergence is a *prompt* change, not nondeterminism.
- Claims are **per-binary and per-checkpoint**: any mlx-node/MLX rebuild or model change re-baselines them; pin SHAs per administration batch.
- Exit codes are unreliable (a known CUDA teardown abort can fire after output is complete) — compare bytes or the session JSONL, never exit codes.

### Extensions and skills

The leading positional commands pass through to pi and manage what lives under the agent config home:

```bash
mlx agent install <source>   # add a pi extension / theme / skill
mlx agent remove <name>      # remove one (alias: uninstall)
mlx agent list               # list installed
mlx agent config             # edit which are enabled
```

`mlx agent update` is intentionally blocked (it maps to pi's npm self-update, which would fight the installed `@mlx-node/cli`); update `@mlx-node/cli` through your package manager instead. `mlx agent -h`/`--help` prints the mlx options above and then pi's full flag list. `mlx agent --version`/`-v` and `mlx agent --export <session>` are answered by pi directly — no local model needed, so the first-run wizard stays out — and `--version` prints the embedded pi version, not `@mlx-node/cli`'s (`mlx --version`).
