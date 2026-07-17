/**
 * `GenmlxModelHost` — the genmlx provider's `StreamSimpleHost`
 * (genmlx-djw6). Mirrors `MlxModelHost`'s serialized-chain mechanics
 * (one resident, one promise chain, dirty/invalidate semantics — see that
 * class's docs for why) with the ChatSession swapped for a
 * {@link GenmlxSession} over the nbb-loaded CLJS turn engine.
 *
 * Native discipline: nothing here touches an addon until the first
 * `runWithResident` — `loadGenmlxEngine()` claims the process latch and
 * dlopens `@genmlx/core` lazily (the process-purity gate).
 */

import type { DiscoveredModelLike } from '../../types.js';
import { loadGenmlxEngine } from './genmlx-host.js';
import type { GenmlxTurnEngine } from './genmlx-host.js';
import { GenmlxSession } from './genmlx-session.js';

export interface GenmlxModelHostOptions {
  /** Injectable engine loader so tests can stub the nbb bridge. */
  loadEngineFn?: () => Promise<GenmlxTurnEngine>;
}

interface ResidentModel {
  id: string;
  session: GenmlxSession;
  /** Post-error flag: next turn full-resets (fresh branch) instead of delta-prefilling. */
  dirty: boolean;
}

export class GenmlxModelHost {
  private readonly byName = new Map<string, DiscoveredModelLike>();
  private readonly loadEngineFn: () => Promise<GenmlxTurnEngine>;
  private resident: ResidentModel | null = null;
  private chain: Promise<unknown> = Promise.resolve();

  constructor(models: DiscoveredModelLike[], opts: GenmlxModelHostOptions = {}) {
    for (const model of models) this.byName.set(model.name, model);
    this.loadEngineFn = opts.loadEngineFn ?? loadGenmlxEngine;
  }

  get residentId(): string | null {
    return this.resident?.id ?? null;
  }

  modelInfo(modelId: string): DiscoveredModelLike | undefined {
    return this.byName.get(modelId);
  }

  runWithResident<T>(modelId: string, fn: (session: GenmlxSession) => Promise<T>): Promise<T> {
    const entry = this.byName.get(modelId);
    if (!entry) {
      const known = [...this.byName.keys()].join(', ');
      return Promise.reject(new Error(`GenmlxModelHost: unknown model "${modelId}" (known models: ${known})`));
    }
    return this.runSerialized(async () => {
      let session: GenmlxSession;
      if (this.resident?.id === modelId) {
        session = this.resident.session;
      } else {
        // Swap: drop the old session first (its engine branches go with the
        // engine-side model swap), then point the engine at the new checkpoint.
        this.resident?.session.reset();
        this.resident = null;
        const engine = await this.loadEngineFn();
        await engine.loadModel(entry.path);
        session = new GenmlxSession(engine);
        this.resident = { id: modelId, session, dirty: false };
      }
      return await fn(session);
    });
  }

  markResidentDirty(modelId: string): void {
    if (this.resident?.id === modelId) {
      this.resident.dirty = true;
    }
  }

  consumeResidentDirty(modelId: string): boolean {
    if (this.resident?.id !== modelId) {
      return false;
    }
    const wasDirty = this.resident.dirty;
    this.resident.dirty = false;
    return wasDirty;
  }

  invalidateResident(modelId: string): void {
    if (this.resident?.id === modelId) {
      this.resident = null;
    }
  }

  private runSerialized<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.chain.then(fn);
    this.chain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }
}
