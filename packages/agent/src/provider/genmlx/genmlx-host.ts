/**
 * The nbb bridge to GenMLX — loads the CLJS turn engine
 * (`src/genmlx/llm/pi_provider.cljs` in the genmlx repo) into THIS process
 * and hands back its `#js` API (bean genmlx-djw6; bridge proven 2026-07-14:
 * nbb programmatic import ~92ms, engine namespace + `@genmlx/core` dlopen +
 * first GPU eval ~0.5s — trivial next to model load).
 *
 * Process shape: pi SDK + nbb-loaded GenMLX + `@genmlx/core` ONLY. The
 * {@link claimNativeOwner} call pins the process BEFORE the dlopen, so a
 * process that already runs the v1 `@mlx-node/lm` host gets a clear error
 * instead of the two-runtimes SIGTRAP.
 *
 * Resolution notes:
 * - The genmlx repo is found via `GENMLX_HOME`, defaulting to the parent of
 *   the mlx-node tree (mlx-node is genmlx's submodule).
 * - nbb is a pinned dependency OF THE GENMLX REPO (1.4.208) — resolved from
 *   its `node_modules`, not ours.
 * - nbb resolves `(js/require "@genmlx/core")` against the process cwd, so
 *   engine startup runs inside a scoped `chdir(genmlxHome)` (restored in
 *   `finally`). Only startup needs this; turns never re-require.
 * - Thor/CUDA hosts need the usual native-loading env (GLIBC_TUNABLES
 *   static-TLS etc.) exactly as every other addon path; see docs/cli.md.
 */

import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { claimNativeOwner } from '../native-owner.js';

/**
 * The `#js` API exported by `pi_provider.cljs` (its last form). Everything
 * crossing the seam is JSON strings plus one per-delta callback — no
 * CLJS↔JS structure marshaling at the boundary.
 */
export interface GenmlxTurnEngine {
  /** Load (or swap to) the flat model dir; resolves to a model-info JSON. */
  loadModel(modelPath: string): Promise<string>;
  /** Create a session (branch-ledger backed); returns the session id. */
  newSession(optsJson: string): string;
  /**
   * Run one turn: render `messagesJson` (pi ChatMessage[] shape), delta-
   * prefill against the session's committed prefix, decode with per-token
   * `onDelta(deltaJson)` callbacks, resolve to the final-result JSON
   * (text/thinking/toolCallText/finishReason/token accounting).
   *
   * Image bytes ride `images` (genmlx-5aah) — the seam's non-JSON leg:
   * messages reference them via `imageRefs` indices, and the engine
   * reattaches before the chat-template render. An image-set change
   * rebuilds the session branch through the owned VLM prefill.
   */
  turnStream(
    sessionId: string,
    messagesJson: string,
    configJson: string,
    onDelta: (deltaJson: string) => void,
    images?: Uint8Array[],
    /** Best-of-K verifier (genmlx-maww): called once with all K candidates
     *  when config.bestOfK > 1; see genmlx-verifier.ts for the contract. */
    verifier?: (candidatesJson: string) => Promise<string> | string,
  ): Promise<string>;
  /** Cooperative cancel of the session's in-flight decode loop. */
  abort(sessionId: string): void;
  /** Drop the session's KV branch + registry entry. */
  dispose(sessionId: string): void;
}

interface NbbApi {
  addClassPath(path: string): void;
  loadFile(path: string): Promise<unknown>;
}

/**
 * Locate the genmlx repo: `GENMLX_HOME` wins; otherwise walk up from this
 * module to the `mlx-node` package root and take its parent. Validated by
 * the presence of `src/genmlx` (a wrong guess must fail loudly here, not as
 * a classpath miss inside nbb).
 */
export function resolveGenmlxHome(env: NodeJS.ProcessEnv = process.env): string {
  const fromEnv = env.GENMLX_HOME;
  if (fromEnv !== undefined && fromEnv !== '') {
    if (!existsSync(join(fromEnv, 'src', 'genmlx'))) {
      throw new Error(`genmlx-host: GENMLX_HOME=${fromEnv} does not look like the genmlx repo (no src/genmlx)`);
    }
    return fromEnv;
  }
  let dir = dirname(fileURLToPath(import.meta.url));
  while (dir !== dirname(dir)) {
    const pkg = join(dir, 'package.json');
    if (existsSync(pkg)) {
      try {
        if ((JSON.parse(readFileSync(pkg, 'utf8')) as { name?: string }).name === 'mlx-node') {
          const home = dirname(dir);
          if (existsSync(join(home, 'src', 'genmlx'))) {
            return home;
          }
          throw new Error(
            `genmlx-host: found mlx-node at ${dir} but its parent has no src/genmlx — ` +
              `is mlx-node checked out as the genmlx submodule? Set GENMLX_HOME explicitly.`,
          );
        }
      } catch (err) {
        if (err instanceof SyntaxError) {
          // Unparseable package.json on the walk — keep climbing.
        } else {
          throw err;
        }
      }
    }
    dir = dirname(dir);
  }
  throw new Error('genmlx-host: could not locate the mlx-node package root; set GENMLX_HOME');
}

let enginePromise: Promise<GenmlxTurnEngine> | null = null;

/**
 * Load the CLJS turn engine (once per process; concurrent callers share the
 * same in-flight promise). A startup failure clears the memo so the next
 * call retries — but the native-owner claim sticks (a partial dlopen may
 * already have happened).
 */
export function loadGenmlxEngine(): Promise<GenmlxTurnEngine> {
  if (enginePromise === null) {
    enginePromise = startEngine().catch((err: unknown) => {
      enginePromise = null;
      throw err;
    });
  }
  return enginePromise;
}

async function startEngine(): Promise<GenmlxTurnEngine> {
  claimNativeOwner('genmlx');
  const home = resolveGenmlxHome();

  const genmlxRequire = createRequire(join(home, 'package.json'));
  let nbbEntry: string;
  try {
    nbbEntry = genmlxRequire.resolve('nbb');
  } catch {
    throw new Error(
      `genmlx-host: nbb is not installed in ${home}/node_modules — ` +
        `run \`bun install\` in the genmlx repo (nbb is a pinned dependency there).`,
    );
  }
  const nbb = (await import(pathToFileURL(nbbEntry).href)) as NbbApi;

  for (const cp of ['src', join('malli', 'src'), join('instaparse', 'src'), 'test.check']) {
    const p = join(home, cp);
    if (existsSync(p)) {
      nbb.addClassPath(p);
    }
  }

  // Scoped chdir: nbb resolves (js/require "@genmlx/core") against cwd.
  const prevCwd = process.cwd();
  let api: unknown;
  try {
    process.chdir(home);
    api = await nbb.loadFile(join(home, 'src', 'genmlx', 'llm', 'pi_provider.cljs'));
  } finally {
    process.chdir(prevCwd);
  }

  const engine = api as Partial<GenmlxTurnEngine> | null | undefined;
  for (const method of ['loadModel', 'newSession', 'turnStream', 'abort', 'dispose'] as const) {
    if (typeof engine?.[method] !== 'function') {
      throw new Error(
        `genmlx-host: pi_provider.cljs did not return the engine API (missing ${method}) — ` +
          `its last form must be the #js api object.`,
      );
    }
  }
  return engine as GenmlxTurnEngine;
}

/** Test-only: forget the memoized engine (does NOT unload native state). */
export function resetGenmlxEngineForTests(): void {
  enginePromise = null;
}
