/**
 * genmlx-host: repo resolution + engine-API validation seams (genmlx-djw6).
 * Native/nbb loading is NOT exercised here — that is the live acceptance
 * path; these tests pin the pure logic around it.
 */
import { mkdirSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { resolveGenmlxHome } from '../src/provider/genmlx/genmlx-host.js';

describe('resolveGenmlxHome', () => {
  it('honors GENMLX_HOME when it looks like the genmlx repo', () => {
    // A real directory with the src/genmlx marker — not a hardcoded host path,
    // which only resolves on the machine it was written on.
    const home = mkdtempSync(join(tmpdir(), 'genmlx-home-'));
    mkdirSync(join(home, 'src', 'genmlx'), { recursive: true });
    expect(resolveGenmlxHome({ GENMLX_HOME: home })).toBe(home);
  });

  it('rejects a GENMLX_HOME without src/genmlx', () => {
    expect(() => resolveGenmlxHome({ GENMLX_HOME: '/tmp' })).toThrowError(/does not look like the genmlx repo/);
  });

  it('defaults to the parent of the mlx-node tree (submodule layout)', () => {
    const home = resolveGenmlxHome({});
    expect(home.endsWith('/genmlx')).toBe(true);
  });
});
