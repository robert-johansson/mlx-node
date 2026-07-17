/**
 * GenmlxSession contract over a scripted fake engine (genmlx-djw6): the
 * callback→generator bridge, the ChatStreamEvent mapping, native-parity
 * abort semantics (no final event), engine-error surfacing, the image
 * wire shape (genmlx-5aah), and reset/dispose bookkeeping. No nbb, no
 * native code.
 */
import type { ChatMessage, ChatStreamEvent } from '@mlx-node/lm';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { GenmlxTurnEngine } from '../src/provider/genmlx/genmlx-host.js';
import { GenmlxSession } from '../src/provider/genmlx/genmlx-session.js';
import { setGenmlxBestOfK, setGenmlxToolVerifier } from '../src/provider/genmlx/genmlx-verifier.js';

interface FakeTurn {
  deltas: Array<{ text: string; isReasoning?: boolean }>;
  final: Record<string, unknown>;
  reject?: unknown;
}

interface FakeEngine extends GenmlxTurnEngine {
  turns: Array<{
    sessionId: string;
    messages: ChatMessage[];
    config: Record<string, unknown>;
    images: Uint8Array[];
    verifier: unknown;
  }>;
  aborted: string[];
  disposed: string[];
  sessionsMinted: number;
}

function makeFinal(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    text: 'hi',
    thinking: null,
    rawText: 'hi',
    finishReason: 'stop',
    toolCalls: [],
    toolCallErrors: [],
    promptTokens: 10,
    numTokens: 2,
    reasoningTokens: 0,
    cachedTokens: 4,
    ...overrides,
  };
}

function makeEngine(script: FakeTurn[]): FakeEngine {
  let turnIndex = 0;
  const engine: FakeEngine = {
    turns: [],
    aborted: [],
    disposed: [],
    sessionsMinted: 0,
    loadModel: vi.fn(async (path: string) => JSON.stringify({ path })),
    newSession: vi.fn(() => `fake-s${++engine.sessionsMinted}`),
    turnStream: vi.fn(async (sessionId: string, messagesJson: string, configJson: string, onDelta, images, verifier) => {
      const turn = script[turnIndex++];
      if (!turn) throw new Error('fake engine: script exhausted');
      engine.turns.push({
        sessionId,
        messages: JSON.parse(messagesJson) as ChatMessage[],
        config: JSON.parse(configJson) as Record<string, unknown>,
        images: images ?? [],
        verifier,
      });
      for (const delta of turn.deltas) onDelta(JSON.stringify(delta));
      if (turn.reject !== undefined) throw turn.reject;
      return JSON.stringify(turn.final);
    }),
    abort: vi.fn((sessionId: string) => {
      engine.aborted.push(sessionId);
    }),
    dispose: vi.fn((sessionId: string) => {
      engine.disposed.push(sessionId);
    }),
  };
  return engine;
}

async function collect(session: GenmlxSession, config = {}, signal?: AbortSignal): Promise<ChatStreamEvent[]> {
  const events: ChatStreamEvent[] = [];
  for await (const event of session.startFromHistoryStream(config, signal)) events.push(event);
  return events;
}

const HISTORY: ChatMessage[] = [
  { role: 'system', content: 'sys' },
  { role: 'user', content: 'hello' },
];

describe('GenmlxSession', () => {
  it('streams deltas then a mapped final; engine sees the primed history + config', async () => {
    const engine = makeEngine([
      {
        deltas: [
          { text: 'think…', isReasoning: true },
          { text: 'hi', isReasoning: false },
        ],
        final: makeFinal({
          toolCalls: [{ id: 'call_1', name: 'echo_tool', arguments: { text: 'x' }, status: 'ok' }],
          finishReason: 'toolUse',
        }),
      },
    ]);
    const session = new GenmlxSession(engine);
    session.primeHistory(HISTORY);
    const events = await collect(session, { temperature: 0 });

    expect(events).toHaveLength(3);
    expect(events[0]).toEqual({ text: 'think…', done: false, isReasoning: true });
    expect(events[1]).toEqual({ text: 'hi', done: false, isReasoning: false });
    const final = events[2] as Extract<ChatStreamEvent, { done: true }>;
    expect(final.done).toBe(true);
    expect(final.finishReason).toBe('toolUse');
    expect(final.cachedTokens).toBe(4);
    expect(final.toolCalls).toEqual([
      { id: 'call_1', name: 'echo_tool', arguments: { text: 'x' }, status: 'ok', rawContent: '' },
    ]);

    expect(engine.turns).toHaveLength(1);
    expect(engine.turns[0]!.messages).toEqual(HISTORY);
    expect(engine.turns[0]!.config).toEqual({ temperature: 0 });
    expect(engine.newSession).toHaveBeenCalledTimes(1);
  });

  it('reuses one engine session across turns; reset disposes it and mints fresh', async () => {
    const engine = makeEngine([
      { deltas: [], final: makeFinal() },
      { deltas: [], final: makeFinal() },
      { deltas: [], final: makeFinal() },
    ]);
    const session = new GenmlxSession(engine);
    session.primeHistory(HISTORY);
    await collect(session);
    await collect(session);
    expect(engine.newSession).toHaveBeenCalledTimes(1);
    expect(engine.turns[1]!.sessionId).toBe(engine.turns[0]!.sessionId);

    session.reset();
    expect(engine.disposed).toEqual([engine.turns[0]!.sessionId]);
    session.primeHistory(HISTORY);
    await collect(session);
    expect(engine.newSession).toHaveBeenCalledTimes(2);
  });

  it('aborted finals end the stream with NO final event (native parity)', async () => {
    const engine = makeEngine([{ deltas: [{ text: 'par' }], final: makeFinal({ finishReason: 'aborted' }) }]);
    const session = new GenmlxSession(engine);
    session.primeHistory(HISTORY);
    const events = await collect(session);
    expect(events).toEqual([{ text: 'par', done: false, isReasoning: false }]);
  });

  it('an abort signal reaches engine.abort; a pre-aborted signal aborts immediately', async () => {
    const engine = makeEngine([
      { deltas: [], final: makeFinal({ finishReason: 'aborted' }) },
      { deltas: [], final: makeFinal({ finishReason: 'aborted' }) },
    ]);
    const session = new GenmlxSession(engine);
    session.primeHistory(HISTORY);
    const controller = new AbortController();
    controller.abort();
    await collect(session, {}, controller.signal);
    expect(engine.aborted).toHaveLength(1);
  });

  it('engine error-finals throw with the engine message', async () => {
    const engine = makeEngine([
      { deltas: [], final: makeFinal({ finishReason: 'error', errorMessage: 'branch exploded' }) },
    ]);
    const session = new GenmlxSession(engine);
    session.primeHistory(HISTORY);
    await expect(collect(session)).rejects.toThrow('branch exploded');
  });

  it('a rejected engine turn propagates', async () => {
    const engine = makeEngine([{ deltas: [], final: makeFinal(), reject: new Error('bridge died') }]);
    const session = new GenmlxSession(engine);
    session.primeHistory(HISTORY);
    await expect(collect(session)).rejects.toThrow('bridge died');
  });

  it('images ride the non-JSON leg: bytes stripped to a flat array, messages carry imageRefs (5aah)', async () => {
    const engine = makeEngine([{ deltas: [], final: makeFinal() }]);
    const session = new GenmlxSession(engine);
    const imageA = new Uint8Array([1, 2]);
    const imageB = new Uint8Array([3, 4, 5]);
    const imageC = new Uint8Array([6]);
    session.primeHistory([
      { role: 'user', content: 'look', images: [imageA, imageB] } as unknown as ChatMessage,
      { role: 'assistant', content: 'ok' } as ChatMessage,
      { role: 'user', content: 'and this', images: [imageC] } as unknown as ChatMessage,
    ]);
    await collect(session);
    const turn = engine.turns[0]!;
    // Flat bytes array, message order preserved, indices correct.
    expect(turn.images).toEqual([imageA, imageB, imageC]);
    const wire = turn.messages as Array<{ images?: unknown; imageRefs?: number[] }>;
    expect(wire[0]!.imageRefs).toEqual([0, 1]);
    expect(wire[0]!.images).toBeUndefined();
    expect(wire[1]!.imageRefs).toBeUndefined();
    expect(wire[2]!.imageRefs).toEqual([2]);
    // The session's own history is untouched (no images stripped in place).
    expect((session as unknown as { history: Array<{ images?: unknown }> }).history[0]!.images).toEqual([
      imageA,
      imageB,
    ]);
  });

  describe('best-of-K seam (genmlx-maww)', () => {
    afterEach(() => {
      setGenmlxBestOfK(null);
      setGenmlxToolVerifier(null);
      delete (globalThis as Record<string, unknown>).__GENMLX_TOOL_VERIFIER__;
      delete process.env.MLX_AGENT_BEST_OF_K;
    });

    it('K rides the config JSON, the verifier rides the non-JSON leg', async () => {
      const engine = makeEngine([{ deltas: [], final: makeFinal() }]);
      const session = new GenmlxSession(engine);
      const verifier = () => JSON.stringify({ winner: 0 });
      setGenmlxBestOfK(3);
      setGenmlxToolVerifier(verifier);
      session.primeHistory(HISTORY);
      await collect(session);
      const turn = engine.turns[0]!;
      expect(turn.config.bestOfK).toBe(3);
      expect(turn.verifier).toBe(verifier);
    });

    it('scalar default: no bestOfK key, no verifier', async () => {
      const engine = makeEngine([{ deltas: [], final: makeFinal() }]);
      const session = new GenmlxSession(engine);
      session.primeHistory(HISTORY);
      await collect(session);
      const turn = engine.turns[0]!;
      expect('bestOfK' in turn.config).toBe(false);
      expect(turn.verifier).toBeUndefined();
    });

    it('MLX_AGENT_BEST_OF_K is the CLI channel; the extension global is the verifier fallback', async () => {
      const engine = makeEngine([{ deltas: [], final: makeFinal() }]);
      const session = new GenmlxSession(engine);
      const globalVerifier = () => JSON.stringify({ winner: 1 });
      process.env.MLX_AGENT_BEST_OF_K = '4';
      (globalThis as Record<string, unknown>).__GENMLX_TOOL_VERIFIER__ = globalVerifier;
      session.primeHistory(HISTORY);
      await collect(session);
      const turn = engine.turns[0]!;
      expect(turn.config.bestOfK).toBe(4);
      expect(turn.verifier).toBe(globalVerifier);
    });

    it('K=1 and malformed env stay scalar', async () => {
      const engine = makeEngine([
        { deltas: [], final: makeFinal() },
        { deltas: [], final: makeFinal() },
      ]);
      const session = new GenmlxSession(engine);
      session.primeHistory(HISTORY);
      process.env.MLX_AGENT_BEST_OF_K = '1';
      await collect(session);
      process.env.MLX_AGENT_BEST_OF_K = 'many';
      await collect(session);
      expect(engine.turns.every((turn) => !('bestOfK' in turn.config))).toBe(true);
    });
  });

  it('warm-reuse-touched fields exist (shared helper contract)', () => {
    const session = new GenmlxSession(makeEngine([]));
    const internals = session as unknown as Record<string, unknown>;
    for (const field of ['inFlight', 'history', 'lastImagesKey', 'turnCount', 'unresolvedOkToolCallCount']) {
      expect(field in internals, `missing warm-reuse field ${field}`).toBe(true);
    }
  });
});
