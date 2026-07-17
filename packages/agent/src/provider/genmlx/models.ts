/**
 * genmlx model discovery — the PURE flat-dir walk (genmlx-djw6).
 *
 * Deliberately reimplements the family gate over raw `config.json` reads
 * instead of importing `detectModelType` from `@mlx-node/lm`: that import
 * chain dlopens `@mlx-node/core` at module load, and discovery runs at CLI
 * startup for BOTH providers — a genmlx run must reach its first model use
 * with no native addon loaded (the process-purity gate; see native-owner.ts).
 *
 * Families served = exactly what the GenMLX OWNED forward implements
 * (genmlx.llm.forward supported-model-types): qwen3, qwen3_5, qwen3_5_moe.
 * qwen3_next is NOT served (its 80B checkpoints route to the native MoE
 * forward, which this provider never loads); vision input is excluded until
 * genmlx-5aah (entries are `input: ['text']` and image turns are rejected
 * at the session). Trait values byte-match the v1 table so the same
 * checkpoint advertises the same window under either provider.
 */

import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { LAUNCH_PRESETS } from '@mlx-node/server/presets';

import type { DiscoveredModelLike } from '../../types.js';

/** Pi `ProviderModelConfig`-shaped entry (structural; avoids a pi type dep here). */
export interface GenmlxPiModel {
  id: string;
  name: string;
  reasoning: boolean;
  input: ['text'];
  cost: { input: 0; output: 0; cacheRead: 0; cacheWrite: 0 };
  contextWindow: number;
  maxTokens: number;
}

export interface GenmlxModelInfo {
  discovered: DiscoveredModelLike;
  piModel: GenmlxPiModel;
}

/** Owned-forward families only — values mirror v1 FAMILY_TRAITS. */
const OWNED_FAMILY_TRAITS: Record<string, { reasoning: boolean; fallbackContextWindow: number }> = {
  qwen3: { reasoning: true, fallbackContextWindow: 40960 },
  qwen3_5: { reasoning: true, fallbackContextWindow: 262144 },
  qwen3_5_moe: { reasoning: true, fallbackContextWindow: 262144 },
};

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

interface ParsedConfig {
  modelType: string | undefined;
  contextWindow: number | undefined;
}

async function readModelConfig(modelPath: string): Promise<ParsedConfig | undefined> {
  try {
    const raw = await readFile(join(modelPath, 'config.json'), 'utf-8');
    const config = JSON.parse(raw) as Record<string, unknown>;
    const modelType = typeof config.model_type === 'string' ? config.model_type : undefined;
    let contextWindow = positiveInteger(config.max_position_embeddings);
    if (contextWindow === undefined) {
      const textConfig = config.text_config;
      if (typeof textConfig === 'object' && textConfig !== null && !Array.isArray(textConfig)) {
        contextWindow = positiveInteger((textConfig as Record<string, unknown>).max_position_embeddings);
      }
    }
    return { modelType, contextWindow };
  } catch {
    return undefined;
  }
}

/**
 * Scan `modelsDir` for owned-forward-servable checkpoints. Same tolerance
 * contract as the v1 walk: unreadable dir → `[]`; undetectable/unsupported
 * entries are skipped silently. No weights are touched, no addon loads.
 */
export async function discoverGenmlxModels(modelsDir: string): Promise<GenmlxModelInfo[]> {
  let entries;
  try {
    entries = await readdir(modelsDir, { withFileTypes: true });
  } catch {
    return [];
  }
  const models: GenmlxModelInfo[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
    const path = join(modelsDir, entry.name);
    const config = await readModelConfig(path);
    if (config?.modelType === undefined) continue;
    const traits = OWNED_FAMILY_TRAITS[config.modelType];
    const preset = LAUNCH_PRESETS[config.modelType];
    if (traits === undefined || preset === undefined) continue;
    models.push({
      discovered: {
        name: entry.name,
        path,
        modelType: config.modelType as DiscoveredModelLike['modelType'],
      },
      piModel: {
        id: entry.name,
        name: entry.name,
        reasoning: traits.reasoning,
        input: ['text'],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: config.contextWindow ?? traits.fallbackContextWindow,
        maxTokens: preset.maxOutputTokens,
      },
    });
  }
  models.sort((a, b) => a.discovered.name.localeCompare(b.discovered.name));
  return models;
}
