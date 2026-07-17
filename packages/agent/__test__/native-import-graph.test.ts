/**
 * Process-purity import-graph gate (genmlx-djw6, spec §2c): NO module in
 * the agent package may STATICALLY import a native-addon-loading package —
 * `@mlx-node/lm` (→ `@mlx-node/core` dlopen), `@mlx-node/core` itself,
 * `@mlx-node/server` (its barrel pulls lm), or `@genmlx/core`. Both model
 * hosts reach their addon only through dynamic `import()` behind the
 * native-owner latch, so registering both providers touches no native
 * code and a `--model genmlx/*` run never dlopens `@mlx-node/core`.
 *
 * Checked over the BUILT dist (what the CLI actually executes): type-only
 * imports are erased there, so any surviving static `import ... from`
 * of a forbidden package is a real runtime dlopen chain.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

const AGENT_DIST = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist');

/** Barrels whose static import means an addon dlopen at module load. */
const FORBIDDEN = ['@mlx-node/lm', '@mlx-node/core', '@mlx-node/server', '@genmlx/core'];

/** Native-free subpaths that are explicitly allowed. */
const ALLOWED = new Set(['@mlx-node/server/presets']);

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else if (name.endsWith('.js')) out.push(full);
  }
  return out;
}

// Static ESM import/re-export forms; dynamic `import('x')` deliberately NOT matched.
const STATIC_IMPORT = /(?:^|\n)\s*(?:import|export)\s[^;]*?from\s*['"]([^'"]+)['"]/g;

describe('agent package native import graph', () => {
  it('has a built dist to check (run `yarn build:ts` first)', () => {
    expect(statSync(AGENT_DIST).isDirectory()).toBe(true);
  });

  it('no module statically imports a native-addon-loading package', () => {
    const offenders: string[] = [];
    for (const file of walk(AGENT_DIST)) {
      const source = readFileSync(file, 'utf-8');
      for (const match of source.matchAll(STATIC_IMPORT)) {
        const spec = match[1]!;
        if (ALLOWED.has(spec)) continue;
        if (FORBIDDEN.some((pkg) => spec === pkg || spec.startsWith(`${pkg}/`))) {
          offenders.push(`${file.slice(AGENT_DIST.length + 1)} -> ${spec}`);
        }
      }
    }
    expect(offenders, `static native import chains:\n${offenders.join('\n')}`).toEqual([]);
  });
});
