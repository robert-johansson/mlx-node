/**
 * `runAgent` — the boot shell that hands control to pi's `main()` with the
 * mlx and genmlx providers, permission gate, local subagents, and terminal
 * branding installed.
 *
 * Spike-proven boot contract:
 * - The env vars below must be set BEFORE any runtime import of
 *   `@earendil-works/pi-coding-agent` (pi reads its config env at
 *   import/call time) — hence the dynamic import and the type-only
 *   top-level pi import here.
 * - pi's `main()` RETURNS on the happy path but `process.exit()`s on
 *   help/error/package-command paths, so nothing critical may run after
 *   `await main()`.
 * - In print/json mode pi takes over stdout (our writes are rerouted to
 *   stderr) and reads non-TTY stdin to EOF as prompt input — this
 *   wrapper must never consume or hold stdin. Paged-config temp cleanup in
 *   `finally` is best-effort for normal returns/rejections, never correctness-
 *   critical on pi's hard `process.exit()` paths.
 * - NOTHING here may statically reach a native addon: boot registers BOTH
 *   providers, and a `--model genmlx/*` run must arrive at its first model
 *   use with no `@mlx-node/core` dlopen behind it (genmlx-djw6). That covers
 *   `@mlx-node/lm`'s pure-JS exports too — the barrel itself loads the addon.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

import type { InlineExtension } from '@earendil-works/pi-coding-agent';

import { createPermissionGateExtension } from './extensions/permission-gate.js';
import { createSubagentExtension } from './extensions/subagent.js';
import { createTerminalTitleExtension } from './extensions/terminal-title.js';
import { createTraceNoticeExtension } from './extensions/trace-notice.js';
import { createGenmlxProviderExtension } from './provider/genmlx/index.js';
import type { GenmlxModelInfo } from './provider/genmlx/models.js';
import { createMlxProviderExtension } from './provider/index.js';
import { MlxModelHost } from './provider/model-host.js';
import {
  type FilterableModelRegistryConstructor,
  installMlxOnlyModelRegistryFilter,
} from './provider/model-registry-filter.js';
import type { MlxModelInfo } from './provider/models.js';

/** Shape of pi's `main(argv, { extensionFactories })` — also the test seam. */
export type RunAgentMain = (args: string[], opts: { extensionFactories: InlineExtension[] }) => Promise<void>;

export interface RunAgentPi {
  main: RunAgentMain;
  ModelRegistry: FilterableModelRegistryConstructor;
}

/** @internal Narrow lifecycle seam for the agent's paged config overlays. */
export interface AgentPagedConfigOverrides {
  resolve(modelPath: string, modelType?: string): Promise<string>;
  cleanup(): Promise<void>;
}

export interface RunAgentOptions {
  /** Resolved models directory (context for callers/diagnostics — discovery already ran). */
  modelsDir: string;
  /** Discovered models to serve through the in-process `mlx` provider. */
  models: MlxModelInfo[];
  /** Discovered models to serve through the in-process `genmlx` provider
   *  (owned-forward families; genmlx-djw6). Empty/omitted → the provider
   *  still registers with no models, which pi treats as unavailable. */
  genmlxModels?: GenmlxModelInfo[];
  /** Passthrough args handed to pi's `main()` verbatim. */
  argv: string[];
  /** Native inference-log path to surface after Pi takes over the TUI. */
  traceLogFile?: string;
  /** @internal Test seam; when set, the pi dynamic import is skipped entirely. */
  piImpl?: RunAgentPi;
  /** @internal Test seam for paged model-path resolution and cleanup. */
  pagedConfigOverrides?: AgentPagedConfigOverrides;
}

/** @internal Exact opt-in parser kept separate so non-`1` values stay disabled. */
export function agentGemmaDraftEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.MLX_AGENT_ENABLE_GEMMA_DRAFT === '1';
}

/**
 * @internal Whether block-paged attention can be active on this host.
 *
 * The paged KV primitives are Metal-only. On the CUDA/Linux build
 * `mlx_metal_is_available()` is false, every model constructor returns early
 * from `initialize_paged_adapter`, and `hasBlockPagedCache()` reports false —
 * so both halves of the paged policy below are pointless there: the config
 * overlay could never switch paging on, and requiring paging would reject
 * every single model load.
 */
export function agentPagedCacheSupported(platform: NodeJS.Platform = process.platform): boolean {
  return platform === 'darwin';
}

/**
 * The agent's paged-overlay seam.
 *
 * Force every paged-capable agent family through an isolated config clone.
 * This includes quantized LFM2 (whose standalone default is deliberately
 * flat) and Qwen3.5 dense/MoE (whose text-only defaults are flat). Gemma4's
 * paged overlay intentionally hides an embedded draft/ directory: the
 * current speculative executor is flat-cache-only and can regress quantized
 * agent workloads. Users may explicitly opt back into that native behavior.
 *
 * Two properties ride here. Where paging cannot run at all
 * ({@link agentPagedCacheSupported}) the seam is an identity resolve, which
 * also leaves an embedded Gemma4 `draft/` visible to native auto-discovery —
 * correct on that build, where every cache is flat anyway. And the manager
 * itself materializes LAZILY, on the first `mlx` model resolution: it is pure
 * JS but only reachable through the `@mlx-node/lm` barrel, whose import
 * dlopens `@mlx-node/core`, and `MlxModelHost` claims the native-owner latch
 * before it calls this (genmlx-djw6).
 */
function createPagedConfigOverrides(preserveEmbeddedGemmaDraft: boolean): AgentPagedConfigOverrides {
  if (!agentPagedCacheSupported()) {
    return {
      resolve: (modelPath) => Promise.resolve(modelPath),
      cleanup: () => Promise.resolve(),
    };
  }
  // `PagedConfigOverrideManager` satisfies AgentPagedConfigOverrides as-is.
  let manager: Promise<AgentPagedConfigOverrides> | undefined;
  const load = (): Promise<AgentPagedConfigOverrides> => {
    manager ??= import('@mlx-node/lm').then((lm) => new lm.PagedConfigOverrideManager({ preserveEmbeddedGemmaDraft }));
    return manager;
  };
  return {
    resolve: async (modelPath, modelType) => (await load()).resolve(modelPath, modelType),
    cleanup: async () => {
      // Never load the barrel just to clean up: with no `mlx` model load there
      // is no overlay to remove. A failed load already rejected `resolve`.
      const pending = manager;
      if (pending === undefined) return;
      await pending.then(
        (loaded) => loaded.cleanup(),
        () => undefined,
      );
    },
  };
}

/**
 * Seed the pi/mlx environment (never clobbering user-set values) and run
 * pi's `main()` with the mlx inline extensions. May not return: pi
 * `process.exit()`s on help/error paths.
 *
 * This is a dedicated-process entrypoint: pi owns stdin/stdout, signal
 * handlers, and process exit while `main()` runs. Do not run it concurrently
 * with another pi SDK/runtime in the same process. The temporary registry
 * policy is restored when `main()` returns or rejects.
 */
export async function runAgent(opts: RunAgentOptions): Promise<void> {
  process.env.PI_CODING_AGENT_DIR ??= join(homedir(), '.mlx-node', 'agent');
  process.env.PI_SKIP_VERSION_CHECK ??= '1';
  // Mirrors `mlx launch claude`: chunked paged prefill keeps long-prompt
  // TTFT bounded on the default paged path.
  process.env.MLX_PAGED_PREFILL_CHUNK_SIZE ??= '2048';

  const pagedConfigOverrides = opts.pagedConfigOverrides ?? createPagedConfigOverrides(agentGemmaDraftEnabled());
  const modelHost = new MlxModelHost(
    opts.models.map((model) => model.discovered),
    {
      resolveModelPathFn: (model) => pagedConfigOverrides.resolve(model.path, model.modelType),
      // Only where paging can actually be active: on a non-Metal build every
      // model reports a flat cache, so enforcing this would reject them all.
      requirePagedCache: agentPagedCacheSupported(),
    },
  );

  // Keep the pi import strictly behind the seam. The seam carries BOTH main
  // and its registry class, so tests and production exercise the same policy
  // installation/lifecycle instead of being able to bypass it accidentally.
  const pi: RunAgentPi = opts.piImpl ?? (await import('@earendil-works/pi-coding-agent'));
  // BOTH local providers' models must survive the policy. The registry's
  // unscoped reads back Tab, `/models`, RPC enumeration and session restore,
  // and an omitted id simply vanishes from all of them with no error — so a
  // genmlx-only inventory would look like an agent with no models at all.
  const restoreModelRegistry = installMlxOnlyModelRegistryFilter(pi.ModelRegistry, [
    ...opts.models.map((model) => model.discovered.name),
    ...(opts.genmlxModels ?? []).map((model) => model.discovered.name),
  ]);
  // Subagents resolve `mlx/<id>` models against the parent's MlxModelHost, so
  // they stay tied to the v1 provider's inventory (genmlx-djw6 rider 2).
  const subagentsEnabled =
    opts.models.length > 0 && !opts.argv.includes('--no-extensions') && !opts.argv.includes('-ne');
  try {
    await pi.main(opts.argv, {
      extensionFactories: [
        createMlxProviderExtension(opts.models, modelHost),
        createGenmlxProviderExtension(opts.genmlxModels ?? []),
        createPermissionGateExtension(),
        ...(subagentsEnabled ? [createSubagentExtension()] : []),
        ...(opts.traceLogFile !== undefined ? [createTraceNoticeExtension(opts.traceLogFile)] : []),
        createTerminalTitleExtension(),
      ],
    });
  } finally {
    try {
      restoreModelRegistry();
    } finally {
      await pagedConfigOverrides.cleanup();
    }
  }
}
