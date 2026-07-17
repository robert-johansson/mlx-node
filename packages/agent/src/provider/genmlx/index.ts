/**
 * `createGenmlxProviderExtension` — the pi inline extension that registers
 * the in-process `genmlx` provider (genmlx-djw6): completions come from
 * GenMLX's OWNED forward (branch-ledger sessions, delta prefill) behind
 * the nbb bridge, through the SAME stream-adapter/TurnEmitter machinery
 * as the v1 `mlx` provider. `--model genmlx/<dir-name>` selects it;
 * registration is strictly additive next to `mlx` (rider 2: v1 stays the
 * baseline arm).
 *
 * Import discipline (load-bearing, same as the v1 extension): type-only pi
 * imports at top level; and NOTHING here may reach a native addon —
 * discovery is the pure config.json walk, and the engine (with its
 * `@genmlx/core` dlopen) loads lazily on first model use behind the
 * native-owner latch.
 */

import type { ExtensionAPI, InlineExtension } from '@earendil-works/pi-coding-agent';

import { GenmlxModelHost } from './genmlx-model-host.js';
import type { GenmlxModelInfo } from './models.js';
import { makeMlxStreamSimple } from '../stream-adapter.js';

export function createGenmlxProviderExtension(models: GenmlxModelInfo[], host?: GenmlxModelHost): InlineExtension {
  const resolvedHost = host ?? new GenmlxModelHost(models.map((m) => m.discovered));
  const streamSimple = makeMlxStreamSimple(resolvedHost);
  return {
    name: 'genmlx-provider',
    factory: (pi: ExtensionAPI) => {
      pi.registerProvider('genmlx', {
        api: 'mlx',
        baseUrl: 'genmlx://local',
        apiKey: 'genmlx-local',
        streamSimple,
        models: models.map((m) => m.piModel),
      });
    },
  };
}
