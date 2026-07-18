/**
 * `GenmlxSession` — the genmlx provider's {@link StreamableSession}
 * (genmlx-djw6). A thin TS face over ONE engine session (a branch-ledger
 * branch CLJS-side): `primeHistory` stores the pi-converted messages,
 * `startFromHistoryStream` runs one engine turn and adapts its per-delta
 * callbacks + final JSON into the `ChatStreamEvent` protocol the shared
 * stream-adapter/TurnEmitter already speak.
 *
 * Semantics mirrored from the v1 native session:
 * - an ABORTED turn ends the stream with NO final event (the adapter
 *   synthesizes the aborted terminal from the signal) — the engine's
 *   'aborted' final is consumed here, not yielded;
 * - an in-band error becomes a `done` event with `finishReason: 'error'`
 *   (the adapter routes it to onError + marks the resident dirty);
 * - `reset()` disposes the engine session (branch + registry) so the next
 *   turn starts from a fresh branch — the post-error rebuild path. The
 *   warm path is a no-op here beyond the TS field wipe: the engine's
 *   token-diff delta prefill IS the warm reuse, structurally.
 *
 * The five `private` fields at the top exist so the shared warm-reuse
 * helper (which wipes them through a structural cast) stays a single
 * implementation across both providers; the genmlx session's real state
 * lives CLJS-side keyed by session id, so wiping them is always safe.
 */

import type { ToolCallResult } from '@mlx-node/core';
import type { ChatConfig, ChatMessage, ChatStreamEvent } from '@mlx-node/lm';

import type { GenmlxTurnEngine } from './genmlx-host.js';
import { resolveGenmlxBestOfK, resolveGenmlxVerifier } from './genmlx-verifier.js';

/** Engine delta JSON shape (pi_provider.cljs emit!). */
interface EngineDelta {
  text: string;
  isReasoning?: boolean;
}

/** Engine final JSON shape (pi_provider.cljs finish-payload). */
interface EngineFinal {
  text: string;
  thinking: string | null;
  rawText: string;
  finishReason: string;
  errorMessage?: string;
  toolCalls: Array<{ id: string; name: string; arguments: Record<string, string>; status: string }>;
  toolCallErrors: string[];
  promptTokens: number;
  numTokens: number;
  reasoningTokens: number;
  cachedTokens: number;
}

export class GenmlxSession {
  // Warm-reuse-touched fields (see module doc) — names are load-bearing,
  // byte-matching ChatSession's privates and WARM_REUSE_TOUCHED_FIELDS.
  private inFlight = false;
  private history: ChatMessage[] = [];
  private lastImagesKey: string | null = null;
  private turnCount = 0;
  private unresolvedOkToolCallCount: number | null = null;

  /**
   * Engine sessions keyed by pi session id (genmlx-lin9). Callers that
   * carry no `options.sessionId` share the `''` key — exactly the old
   * one-memoized-session behavior.
   */
  private readonly sessionIds = new Map<string, string>();
  /** Fork hints: new pi session id -> source session FILE (session_start). */
  private readonly pendingForks = new Map<string, string>();

  constructor(private readonly engine: GenmlxTurnEngine) {}

  /** Dispose every engine session (branches + registry) and wipe TS state. */
  reset(): void {
    for (const engineId of this.sessionIds.values()) {
      this.engine.dispose(engineId);
    }
    this.sessionIds.clear();
    this.pendingForks.clear();
    this.inFlight = false;
    this.history = [];
    this.lastImagesKey = null;
    this.turnCount = 0;
    this.unresolvedOkToolCallCount = null;
  }

  /**
   * Record an in-process pi session fork (genmlx-lin9): when pi's runtime
   * forks a session (`session_start` reason "fork"), the NEW pi session's
   * first turn should mint its engine session as an O(1) `branch-from`
   * fork of the source's — the shared committed prefix then DELTA-PREFILLS
   * instead of cold-replaying. The hint is consulted (and consumed) at
   * engine-session mint time; an unresolvable source falls back to a cold
   * session, never an error. A fork cut at an interior entry yields a
   * history that is a strict PREFIX of the source's committed tokens — the
   * engine's rebuild rule then full-prefills correctly, so hints are O(1)
   * for leaf forks (counterfactual administration) and merely neutral
   * elsewhere.
   */
  noteFork(newPiSessionId: string, previousSessionFile: string): void {
    this.pendingForks.set(newPiSessionId, previousSessionFile);
  }

  /** Pi session ids embed in session file names: `<timestamp>_<uuid>.jsonl`. */
  private static piSessionIdFromFile(file: string): string | null {
    const match = /_([0-9a-f-]{36})\.jsonl$/i.exec(file);
    return match?.[1] ?? null;
  }

  /** Existing engine session for `piKey`, else mint one (fork-aware). */
  private ensureEngineSession(piKey: string): string {
    const existing = this.sessionIds.get(piKey);
    if (existing !== undefined) {
      return existing;
    }
    const sourceFile = this.pendingForks.get(piKey);
    this.pendingForks.delete(piKey);
    if (sourceFile !== undefined) {
      const sourcePiId = GenmlxSession.piSessionIdFromFile(sourceFile);
      const sourceEngineId = sourcePiId !== null ? this.sessionIds.get(sourcePiId) : undefined;
      if (sourceEngineId !== undefined) {
        try {
          const forked = this.engine.newSession(JSON.stringify({ forkFrom: sourceEngineId }));
          this.sessionIds.set(piKey, forked);
          return forked;
        } catch {
          // e.g. :fork-source-busy — fall through to a cold session; the
          // engine's delta prefill still reuses nothing worse than v1 would.
        }
      }
    }
    const fresh = this.engine.newSession('{}');
    this.sessionIds.set(piKey, fresh);
    return fresh;
  }

  primeHistory(messages: ChatMessage[]): void {
    this.history = messages;
  }

  async *startFromHistoryStream(
    config?: ChatConfig,
    signal?: AbortSignal,
    piSessionId?: string,
  ): AsyncGenerator<ChatStreamEvent> {
    if (this.inFlight) {
      throw new Error('GenmlxSession: a turn is already in flight');
    }
    // Images ride the seam's non-JSON leg (genmlx-5aah): bytes go as a flat
    // Uint8Array array, messages carry imageRefs indices into it (a JSON-
    // stringified Uint8Array would explode into an index-keyed object). The
    // engine reattaches them before the Rust chat-template render.
    const imageBytes: Uint8Array[] = [];
    const wireMessages = this.history.map((message) => {
      const images = (message as { images?: Uint8Array[] }).images;
      if (images === undefined || images.length === 0) {
        return message;
      }
      const imageRefs = images.map((bytes) => {
        imageBytes.push(bytes);
        return imageBytes.length - 1;
      });
      return { ...message, images: undefined, imageRefs };
    });
    const sessionId = this.ensureEngineSession(piSessionId ?? '');

    // Callback-push → async-generator-pull bridge.
    const queue: ChatStreamEvent[] = [];
    let wake: (() => void) | null = null;
    let finalJson: string | null = null;
    let failure: unknown = null;
    let settled = false;
    const notify = () => {
      wake?.();
      wake = null;
    };
    const onAbort = () => {
      this.engine.abort(sessionId);
    };
    if (signal?.aborted) {
      onAbort();
    } else {
      signal?.addEventListener('abort', onAbort, { once: true });
    }

    this.inFlight = true;
    try {
      // Best-of-K (genmlx-maww): K + the verifier ride the seam per turn —
      // K into the config JSON, the verifier on the non-JSON leg.
      const bestOfK = resolveGenmlxBestOfK();
      const wireConfig: Record<string, unknown> = { ...(config ?? {}) };
      if (bestOfK !== null) {
        wireConfig.bestOfK = bestOfK;
        const timeoutMs = Number(process.env.MLX_AGENT_VERIFIER_TIMEOUT_MS ?? Number.NaN);
        if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
          wireConfig.verifierTimeoutMs = timeoutMs;
        }
      }
      this.engine
        .turnStream(
          sessionId,
          JSON.stringify(wireMessages),
          JSON.stringify(wireConfig),
          (deltaJson) => {
            const delta = JSON.parse(deltaJson) as EngineDelta;
            queue.push({ text: delta.text, done: false, isReasoning: delta.isReasoning === true });
            notify();
          },
          imageBytes,
          resolveGenmlxVerifier(),
        )
        .then(
          (json) => {
            finalJson = json;
            settled = true;
            notify();
          },
          (err: unknown) => {
            failure = err;
            settled = true;
            notify();
          },
        );

      while (!settled || queue.length > 0) {
        if (queue.length > 0) {
          yield queue.shift()!;
        } else {
          await new Promise<void>((resolve) => {
            wake = resolve;
          });
        }
      }
      if (failure !== null) {
        throw failure;
      }
      const final = JSON.parse(finalJson!) as EngineFinal;
      this.turnCount += 1;
      if (final.finishReason === 'aborted') {
        // Native parity: aborted streams end cleanly with NO final event.
        return;
      }
      if (final.finishReason === 'error') {
        // Throw instead of yielding the in-band error final: the adapter's
        // catch path marks the resident dirty either way, but a thrown error
        // carries the ENGINE's message into the pi-visible terminal.
        throw new Error(final.errorMessage ?? 'genmlx engine turn failed');
      }
      yield {
        text: final.text,
        done: true,
        finishReason: final.finishReason,
        toolCalls: final.toolCalls.map(
          (call): ToolCallResult => ({
            id: call.id,
            name: call.name,
            arguments: call.arguments,
            status: call.status === 'ok' ? 'ok' : 'parse_error',
            rawContent: '',
          }),
        ),
        thinking: final.thinking,
        numTokens: final.numTokens,
        promptTokens: final.promptTokens,
        reasoningTokens: final.reasoningTokens,
        rawText: final.rawText,
        cachedTokens: final.cachedTokens,
      } as ChatStreamEvent;
    } finally {
      this.inFlight = false;
      signal?.removeEventListener('abort', onAbort);
    }
  }
}
