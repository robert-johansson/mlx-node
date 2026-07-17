/**
 * GenmlxModelHost contract over a stubbed engine loader (genmlx-djw6):
 * lazy engine load + model residency, serialized chain, dirty/invalidate
 * semantics — the StreamSimpleHost mechanics mirrored from MlxModelHost,
 * with zero native/nbb involvement.
 */
import { describe, expect, it, vi } from 'vite-plus/test';

import type { GenmlxTurnEngine } from '../src/provider/genmlx/genmlx-host.js';
import { GenmlxModelHost } from '../src/provider/genmlx/genmlx-model-host.js';
import type { GenmlxSession } from '../src/provider/genmlx/genmlx-session.js';
import type { DiscoveredModelLike } from '../src/types.js';

const MODELS: DiscoveredModelLike[] = [
  { name: 'ornith-35b', path: '/models/ornith-35b', modelType: 'qwen3_5_moe' },
  { name: 'qwen-small', path: '/models/qwen-small', modelType: 'qwen3_5' },
];

function makeFakeEngine(): GenmlxTurnEngine {
  let minted = 0;
  return {
    loadModel: vi.fn(async (path: string) => JSON.stringify({ path })),
    newSession: vi.fn(() => `s${++minted}`),
    turnStream: vi.fn(async () => '{}'),
    abort: vi.fn(),
    dispose: vi.fn(),
  };
}

function makeHost(): { host: GenmlxModelHost; engine: GenmlxTurnEngine; loadEngineFn: ReturnType<typeof vi.fn> } {
  const engine = makeFakeEngine();
  const loadEngineFn = vi.fn(async () => engine);
  return { host: new GenmlxModelHost(MODELS, { loadEngineFn }), engine, loadEngineFn };
}

function getSession(host: GenmlxModelHost, id: string): Promise<GenmlxSession> {
  return host.runWithResident(id, async (session) => session);
}

describe('GenmlxModelHost', () => {
  it('loads the engine + model lazily once and reuses the resident session', async () => {
    const { host, engine, loadEngineFn } = makeHost();
    expect(host.residentId).toBeNull();
    expect(loadEngineFn).not.toHaveBeenCalled();

    const first = await getSession(host, 'ornith-35b');
    const second = await getSession(host, 'ornith-35b');
    expect(second).toBe(first);
    expect(loadEngineFn).toHaveBeenCalledTimes(1);
    expect(engine.loadModel).toHaveBeenCalledTimes(1);
    expect(engine.loadModel).toHaveBeenCalledWith('/models/ornith-35b');
    expect(host.residentId).toBe('ornith-35b');
  });

  it('rejects unknown ids without loading anything', async () => {
    const { host, loadEngineFn } = makeHost();
    await expect(getSession(host, 'nope')).rejects.toThrow(/unknown model "nope".*ornith-35b.*qwen-small/s);
    expect(loadEngineFn).not.toHaveBeenCalled();
  });

  it('swap points the engine at the new checkpoint and resets the old session', async () => {
    const { host, engine } = makeHost();
    const s1 = await getSession(host, 'ornith-35b');
    const resetSpy = vi.spyOn(s1, 'reset');
    const s2 = await getSession(host, 'qwen-small');
    expect(s2).not.toBe(s1);
    expect(resetSpy).toHaveBeenCalledTimes(1);
    expect(engine.loadModel).toHaveBeenNthCalledWith(2, '/models/qwen-small');
    expect(host.residentId).toBe('qwen-small');
  });

  it('dirty mark/consume semantics match the v1 host', async () => {
    const { host } = makeHost();
    await getSession(host, 'ornith-35b');
    expect(host.consumeResidentDirty('ornith-35b')).toBe(false);
    host.markResidentDirty('ornith-35b');
    expect(host.consumeResidentDirty('ornith-35b')).toBe(true);
    expect(host.consumeResidentDirty('ornith-35b')).toBe(false);
    host.markResidentDirty('other');
    expect(host.consumeResidentDirty('ornith-35b')).toBe(false);
  });

  it('invalidateResident drops the resident so the next call reloads', async () => {
    const { host, engine } = makeHost();
    await getSession(host, 'ornith-35b');
    host.invalidateResident('ornith-35b');
    expect(host.residentId).toBeNull();
    await getSession(host, 'ornith-35b');
    expect(engine.loadModel).toHaveBeenCalledTimes(2);
  });

  it('serializes callbacks FIFO (a parked first callback blocks the second)', async () => {
    const { host } = makeHost();
    const order: string[] = [];
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const p1 = host.runWithResident('ornith-35b', async () => {
      order.push('1-start');
      await gate;
      order.push('1-end');
    });
    const p2 = host.runWithResident('ornith-35b', async () => {
      order.push('2-start');
    });
    for (let i = 0; i < 8; i++) await new Promise<void>((resolve) => setTimeout(resolve, 0));
    expect(order).toEqual(['1-start']);
    release();
    await p1;
    await p2;
    expect(order).toEqual(['1-start', '1-end', '2-start']);
  });
});
