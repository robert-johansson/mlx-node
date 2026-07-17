/**
 * heapRelaunchTargetMb — the pure decision behind the one-shot `mlx agent`
 * heap re-exec (genmlx-djw6): relaunch exactly when running under real Node,
 * not already relaunched, not disabled, and the current V8 old-space limit
 * is below target.
 */

import { describe, expect, it } from 'vite-plus/test';

import { DEFAULT_AGENT_HEAP_MB, heapRelaunchTargetMb } from '../src/heap-relaunch.js';

describe('heapRelaunchTargetMb', () => {
  it('relaunches at the default target when the limit is Node-default sized', () => {
    expect(heapRelaunchTargetMb({}, 4192, false)).toBe(DEFAULT_AGENT_HEAP_MB);
  });

  it('does not relaunch when the limit already meets the target', () => {
    expect(heapRelaunchTargetMb({}, DEFAULT_AGENT_HEAP_MB, false)).toBeNull();
    expect(heapRelaunchTargetMb({}, 16384, false)).toBeNull();
  });

  it('never recurses: the relaunched child skips the check', () => {
    expect(heapRelaunchTargetMb({ MLX_AGENT_HEAP_RELAUNCHED: '1' }, 4192, false)).toBeNull();
  });

  it('skips under Bun (heap not governed by V8 flags)', () => {
    expect(heapRelaunchTargetMb({}, 4192, true)).toBeNull();
  });

  it('honors an override target', () => {
    expect(heapRelaunchTargetMb({ MLX_AGENT_MAX_OLD_SPACE_MB: '8192' }, 4192, false)).toBe(8192);
    expect(heapRelaunchTargetMb({ MLX_AGENT_MAX_OLD_SPACE_MB: '8192' }, 8192, false)).toBeNull();
  });

  it('0 disables the relaunch entirely', () => {
    expect(heapRelaunchTargetMb({ MLX_AGENT_MAX_OLD_SPACE_MB: '0' }, 1024, false)).toBeNull();
  });

  it('a malformed override disables rather than guessing', () => {
    expect(heapRelaunchTargetMb({ MLX_AGENT_MAX_OLD_SPACE_MB: 'lots' }, 4192, false)).toBeNull();
  });

  it('an empty override falls back to the default target', () => {
    expect(heapRelaunchTargetMb({ MLX_AGENT_MAX_OLD_SPACE_MB: '' }, 4192, false)).toBe(DEFAULT_AGENT_HEAP_MB);
  });
});
