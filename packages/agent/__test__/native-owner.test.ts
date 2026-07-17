/**
 * The native-host latch (SIGTRAP wall): first claim pins the process,
 * same-host claims are idempotent, cross-host claims throw BEFORE any
 * dlopen could happen (genmlx-djw6 process-purity gate).
 */
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { claimNativeOwner, nativeOwner, resetNativeOwnerForTests } from '../src/provider/native-owner.js';

describe('native-owner latch', () => {
  afterEach(() => {
    resetNativeOwnerForTests();
  });

  it('starts unpinned', () => {
    expect(nativeOwner()).toBeNull();
  });

  it('first claim pins the process', () => {
    claimNativeOwner('mlx');
    expect(nativeOwner()).toBe('mlx');
  });

  it('same-host claims are idempotent', () => {
    claimNativeOwner('genmlx');
    claimNativeOwner('genmlx');
    expect(nativeOwner()).toBe('genmlx');
  });

  it('cross-host claim throws a clear error naming both hosts', () => {
    claimNativeOwner('mlx');
    expect(() => claimNativeOwner('genmlx')).toThrowError(/already loaded the 'mlx' native host.*'genmlx'/s);
    expect(nativeOwner()).toBe('mlx');
  });

  it('cross-host claim throws in the other direction too', () => {
    claimNativeOwner('genmlx');
    expect(() => claimNativeOwner('mlx')).toThrowError(/SIGTRAP/);
  });
});
