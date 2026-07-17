/**
 * Best-of-K verifier registration (genmlx-maww) — SEAM CONTRACT v1.
 *
 * On the genmlx provider, `bestOfK > 1` decodes K candidate turns in one
 * batched forward and calls an external verifier ONCE with all K:
 *
 *   verifier(candidatesJson) -> resultJson (string or Promise<string>)
 *     candidatesJson: {"candidates": [{"index", "text", "finishReason",
 *                                      "toolCalls", "toolCallErrors"}, ...]}
 *     resultJson:     {"winner": <index>}  OR  {"scores": [K numbers]}
 *                     (argmax; ties -> lowest index)
 *
 * Degenerate semantics (engine-side, part of the contract): verifier
 * absent / throwing / timing out (`verifierTimeoutMs`, default 10000) /
 * answering malformed -> candidate 0; `bestOfK: 1` (or unset) is the
 * scalar path, byte-identical to the core provider.
 *
 * Registration channels:
 * - `setGenmlxToolVerifier(fn)` / `setGenmlxBestOfK(k)` — for in-process
 *   callers (tests, embedders, future run-agent wiring).
 * - `globalThis.__GENMLX_TOOL_VERIFIER__` — for pi EXTENSION files, which
 *   can only import `@earendil-works/*` (the jiti alias set): assign the
 *   verifier in the extension body. See `examples/best-of-k-verifier.ts`.
 * - `MLX_AGENT_BEST_OF_K` env — the CLI-facing K knob (registry wins).
 *
 * v1 scope mirrors the engine: no thinking / per-argument-grammar
 * composition with K>1 (typed engine errors); candidate turns do not
 * stream token-by-token (the winner arrives as one delta).
 */

export type GenmlxToolVerifier = (candidatesJson: string) => Promise<string> | string;

const VERIFIER_GLOBAL = '__GENMLX_TOOL_VERIFIER__';

let registeredVerifier: GenmlxToolVerifier | null = null;
let registeredBestOfK: number | null = null;

/** Register (or clear, with null) the process-wide tool verifier. */
export function setGenmlxToolVerifier(fn: GenmlxToolVerifier | null): void {
  registeredVerifier = fn;
}

/** Register (or clear, with null) the process-wide best-of-K width. */
export function setGenmlxBestOfK(k: number | null): void {
  registeredBestOfK = k;
}

/** The verifier to use this turn: explicit registration wins, then the
 *  extension-facing global. */
export function resolveGenmlxVerifier(): GenmlxToolVerifier | undefined {
  if (registeredVerifier !== null) {
    return registeredVerifier;
  }
  const fromGlobal = (globalThis as Record<string, unknown>)[VERIFIER_GLOBAL];
  return typeof fromGlobal === 'function' ? (fromGlobal as GenmlxToolVerifier) : undefined;
}

/** The best-of-K width this turn: registration wins, then MLX_AGENT_BEST_OF_K.
 *  Returns null when unset/invalid/<=1 (the scalar path). */
export function resolveGenmlxBestOfK(env: Record<string, string | undefined> = process.env): number | null {
  const k = registeredBestOfK ?? Number(env.MLX_AGENT_BEST_OF_K ?? Number.NaN);
  return Number.isInteger(k) && k > 1 ? k : null;
}
