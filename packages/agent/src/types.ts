import type { ChatConfig, ChatMessage, ChatStreamEvent, ModelType } from '@mlx-node/lm';

/** Structural mirror of the CLI's DiscoveredModel — avoids a cli↔agent dependency cycle. */
export interface DiscoveredModelLike {
  name: string;
  path: string;
  modelType: ModelType;
}

/**
 * The minimal session surface the stream adapter drives — kept structural
 * so BOTH providers' sessions satisfy it: the v1 `ChatSession` as-is, and
 * the genmlx provider's `GenmlxSession` (whose real turn state lives
 * CLJS-side behind the nbb bridge; genmlx-djw6). All type-only imports —
 * referencing this interface never dlopens a native addon.
 */
export interface StreamableSession {
  /** Full reset: native caches + JS history (the post-error rebuild path). */
  reset(): void | Promise<void>;
  /** Replace the session's committed history wholesale (turnCount must be 0). */
  primeHistory(messages: ChatMessage[]): void;
  /**
   * Run one turn from the primed history as a ChatStreamEvent stream.
   * `piSessionId` (pi's `options.sessionId`, genmlx-lin9) lets a session
   * implementation key per-conversation engine state; the v1 `ChatSession`
   * takes two parameters and satisfies this structurally (fewer-params
   * assignability) — it simply never sees the id.
   */
  startFromHistoryStream(
    config?: ChatConfig,
    signal?: AbortSignal,
    piSessionId?: string,
  ): AsyncGenerator<ChatStreamEvent>;
}
