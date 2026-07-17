/**
 * Per-call `ChatConfig` assembly for the provider bridge.
 *
 * Base sampling + output budget come from `@mlx-node/server`'s
 * `LAUNCH_PRESETS` (the ONLY allowed server import in this package —
 * presets/preset types, nothing else) extended by the agent-local
 * {@link AGENT_LAUNCH_PRESETS}, then pi's per-call `SimpleStreamOptions`
 * overlay on top.
 */

import type { SimpleStreamOptions, ThinkingLevel } from '@earendil-works/pi-ai';
import type { ChatConfig, ModelType, ToolDefinition } from '@mlx-node/lm';
import { LAUNCH_PRESETS, type LaunchPreset } from '@mlx-node/server/presets';

/**
 * Agent-local launch presets for model types `LAUNCH_PRESETS` does not
 * cover (kept here — this package must not fork `packages/server`).
 *
 * `lfm2_moe` (LFM2.5-8B-A1B): LiquidAI's HF model card for the MoE
 * checkpoint recommends temperature 0.2 / top_k 80 — deliberately NOT
 * the dense `lfm2` preset (LFM2.5-1.2B guidance: temperature 0.05 /
 * top_k 50). repetitionPenalty 1.05 and the 8192-token output budget
 * match the dense family entry.
 *
 * `qwen3_next` (Qwen3-Coder-Next) is served by the server table's own
 * entry — added there so `mlx launch claude` gets the family too.
 */
const AGENT_LAUNCH_PRESETS: Partial<Record<ModelType, LaunchPreset>> = {
  lfm2_moe: {
    sampling: {
      temperature: 0.2,
      topP: 1.0,
      topK: 80,
      minP: 0.0,
      presencePenalty: 0.0,
      repetitionPenalty: 1.05,
    },
    maxOutputTokens: 8192,
  },
};

/**
 * Preset lookup — agent-local entries win over `LAUNCH_PRESETS` (they
 * exist precisely because the server table has no correct entry for the
 * type). This is the ONE preset resolution shared by discovery
 * (`models.ts`) and per-call config assembly, so a model can never be
 * discovered without also being streamable (and vice versa).
 */
export function launchPresetFor(modelType: ModelType): LaunchPreset | undefined {
  return AGENT_LAUNCH_PRESETS[modelType] ?? LAUNCH_PRESETS[modelType];
}

/**
 * pi thinking level → native `reasoningEffort`. pi never delivers 'off'
 * here (the agent loop converts it to `undefined` before the provider
 * sees it), so `undefined` is the "thinking disabled" signal → 'none'.
 */
const THINKING_LEVEL_TO_EFFORT: Record<ThinkingLevel, 'low' | 'medium' | 'high'> = {
  minimal: 'low',
  low: 'low',
  medium: 'medium',
  high: 'high',
  xhigh: 'high',
  max: 'high',
};

export function buildChatConfig(
  modelType: ModelType,
  options: SimpleStreamOptions | undefined,
  tools: ToolDefinition[] | undefined,
): ChatConfig {
  const preset = launchPresetFor(modelType);
  if (!preset) {
    const known = [...new Set([...Object.keys(LAUNCH_PRESETS), ...Object.keys(AGENT_LAUNCH_PRESETS)])].join(', ');
    throw new Error(`buildChatConfig: no launch preset for model type "${modelType}" (known types: ${known})`);
  }

  const config: ChatConfig = {
    ...preset.sampling,
    maxNewTokens: preset.maxOutputTokens,
    reasoningEffort: options?.reasoning === undefined ? 'none' : THINKING_LEVEL_TO_EFFORT[options.reasoning],
  };
  if (options?.maxTokens !== undefined) config.maxNewTokens = options.maxTokens;
  // Per-run sampling override for measurement work (pi has no temperature
  // flag and never sets SimpleStreamOptions.temperature itself, so without
  // this the preset always wins). Explicit pi options still take precedence.
  // The native sampler treats temperature <= 1e-6 as greedy argmax, so
  // MLX_AGENT_TEMPERATURE=0 selects deterministic decoding.
  const envTemp = process.env.MLX_AGENT_TEMPERATURE;
  if (envTemp !== undefined && envTemp !== '') {
    const parsed = Number(envTemp);
    if (!Number.isFinite(parsed) || parsed < 0) {
      throw new Error(`MLX_AGENT_TEMPERATURE must be a finite number >= 0, got "${envTemp}"`);
    }
    config.temperature = parsed;
  }
  if (options?.temperature !== undefined) config.temperature = options.temperature;
  // Thinking-token budget for measurement sweeps (genmlx-fy9j): the native
  // ReasoningTracker forces </think> at the cap, giving intermediate points
  // between "no think" and "unlimited think". Only meaningful when thinking
  // is enabled — on the qwen3.5 family `--thinking none|low` disables
  // thinking at the template level (low == off), so pair this knob with
  // `--thinking medium` or higher. usage.reasoning verifies each cap.
  const envBudget = process.env.MLX_AGENT_THINKING_BUDGET;
  if (envBudget !== undefined && envBudget !== '') {
    const parsed = Number(envBudget);
    if (!Number.isInteger(parsed) || parsed < 0) {
      throw new Error(`MLX_AGENT_THINKING_BUDGET must be an integer >= 0, got "${envBudget}"`);
    }
    config.thinkingTokenBudget = parsed;
  }
  if (tools && tools.length > 0) config.tools = tools;
  // `reuseCache` is deliberately NOT set: ChatSession.mergeConfig forces it on.
  return config;
}
