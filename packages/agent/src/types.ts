import type { ChatConfig, ChatMessage, ChatStreamEvent, ModelType, SessionContextLimits } from '@mlx-node/lm';

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
  /** Load-time physical context snapshot, when the session exposes one. */
  contextLimits(): SessionContextLimits | undefined;
  /** Authoritative image-input capability of the resident model. */
  supportsImages(): boolean;
  /** Replace the session's committed history wholesale (turnCount must be 0). */
  primeHistory(messages: ChatMessage[]): void;
  /**
   * Run one turn from the primed history as a ChatStreamEvent stream.
   * Per-conversation identity travels inside `config` (`cacheOwnerId`,
   * pi's `options.sessionId`), which is what lets a session implementation
   * key engine state per conversation and honor fork hints (genmlx-lin9).
   */
  startFromHistoryStream(config?: ChatConfig, signal?: AbortSignal): AsyncGenerator<ChatStreamEvent>;
}
