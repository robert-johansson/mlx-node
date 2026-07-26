/**
 * Gated live smoke for the assembled provider seam: `makeMlxStreamSimple`
 * over a REAL `MlxModelHost` + real weights — one text turn plus one
 * tool round-trip (call → fabricated result → continuation).
 *
 * Availability convention (spike B): the smallest local qwen3.5
 * checkpoint, overridable via `MLX_AGENT_TEST_MODEL`. Skips cleanly when
 * no candidate exists. Turns run strictly sequentially on one shared
 * host — GPU work is never concurrent.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, join } from 'node:path';

import type {
  Api,
  AssistantMessage,
  AssistantMessageEvent,
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
  Tool,
  ToolResultMessage,
  TSchema,
} from '@earendil-works/pi-ai';
import { detectModelType } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

import { MlxModelHost } from '../src/provider/model-host.js';
import { makeMlxStreamSimple } from '../src/provider/stream-adapter.js';
import type { DiscoveredModelLike } from '../src/types.js';

/**
 * Resolve a Hugging Face hub snapshot directory (`.../snapshots/<sha>/`). The
 * sha is content-addressed, so it cannot be hard-coded.
 */
function hfSnapshot(repo: string): string | undefined {
  const root = join(homedir(), '.cache', 'huggingface', 'hub', repo, 'snapshots');
  if (!existsSync(root)) return undefined;
  for (const sha of readdirSync(root)) {
    const dir = join(root, sha);
    if (existsSync(join(dir, 'config.json'))) return dir;
  }
  return undefined;
}

/**
 * Prefer the GENUINELY bf16 Qwen3.5-0.8B.
 *
 * The `qwen3.5-0.8b-mlx-bf16` directories under ~/.mlx-node/models and
 * ~/.cache/models are misnamed: both report
 * `quantization: {bits: 4, group_size: 64}` in their own config.json. Running
 * this suite against 4-bit weights under a bf16 name is how the tool
 * round-trip came to be permanently red here — a 4-bit 0.8B does not reliably
 * emit the qwen3.5 `<tool_call><function=…>` form, so it returned stopReason
 * 'stop' and looked like a provider defect for months. They stay as fallbacks
 * so the suite still runs where the real checkpoint is absent, but they are
 * no longer the default.
 */
const CANDIDATES = [
  process.env.MLX_AGENT_TEST_MODEL,
  hfSnapshot('models--mlx-community--Qwen3.5-0.8B-bf16'),
  join(homedir(), '.mlx-node', 'models', 'qwen3.5-0.8b-mlx-bf16'),
  join(homedir(), '.cache', 'models', 'qwen3.5-0.8b-mlx-bf16'),
].filter((p): p is string => typeof p === 'string' && p.length > 0);

const MODEL_PATH = CANDIDATES.find((p) => existsSync(join(p, 'config.json')));

/**
 * The tool round-trip needs a checkpoint that can actually EMIT a tool call —
 * a strictly stronger requirement than the text turn, and one the default
 * candidates do not meet on this box.
 *
 * Every locally available `qwen3.5-0.8b-mlx-bf16` reports
 * `quantization: {bits: 4, group_size: 64}` in its own config.json: the
 * `-bf16` suffix is a lie (the same trap recorded for this checkpoint
 * elsewhere in the project). A 4-bit 0.8B does not reliably produce the
 * qwen3.5 `<tool_call><function=…>` form, so the round-trip returns
 * stopReason 'stop'. That is model weakness, not a provider defect — it was
 * recorded as such the day this file landed (genmlx-rhuk), the suite is
 * local-only (CI resolves models to a repo-relative path this test never
 * searches, so `skipIf` always fires there), and it has therefore never been
 * green on Linux/CUDA.
 *
 * So the round-trip resolves its OWN model and skips when none is present,
 * rather than the assertion being relaxed to accept 'stop'. Set
 * MLX_AGENT_TOOL_TEST_MODEL to point at any tool-capable checkpoint.
 */
const TOOL_CANDIDATES = [
  process.env.MLX_AGENT_TOOL_TEST_MODEL,
  hfSnapshot('models--mlx-community--Qwen3.5-0.8B-bf16'),
  join(homedir(), '.mlx-node', 'models', 'qwen3.6-35b-a3b-4bit'),
].filter((p): p is string => typeof p === 'string' && p.length > 0);

const TOOL_MODEL_PATH = TOOL_CANDIDATES.find((p) => existsSync(join(p, 'config.json')));

const TURN_TIMEOUT = 240_000;
const OPTIONS: SimpleStreamOptions = { maxTokens: 128, temperature: 0 }; // no `reasoning` → reasoningEffort 'none'
const SYSTEM = 'You are a concise assistant. Answer in at most two sentences.';

async function collect(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
  const events: AssistantMessageEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

function finalMessage(events: AssistantMessageEvent[]): AssistantMessage {
  const last = events[events.length - 1]!;
  if (last.type === 'done') return last.message;
  if (last.type === 'error') return last.error;
  throw new Error(`stream did not terminate: last event ${last.type}`);
}

function visibleText(message: AssistantMessage): string {
  return message.content
    .filter((part): part is Extract<AssistantMessage['content'][number], { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('\n');
}

async function bindHost(modelPath: string): Promise<{
  streamSimple: ReturnType<typeof makeMlxStreamSimple>;
  model: Model<Api>;
}> {
  const discovered: DiscoveredModelLike = {
    name: basename(modelPath),
    path: modelPath,
    modelType: await detectModelType(modelPath),
  };
  // Name the weights that produced the result: a green run must never be
  // attributable to a checkpoint nobody can identify afterwards.
  const cfg = JSON.parse(readFileSync(join(modelPath, 'config.json'), 'utf8')) as {
    quantization?: unknown;
  };
  console.log(
    `[provider-live] ${discovered.name} type=${discovered.modelType} ` +
      `quantization=${cfg.quantization ? JSON.stringify(cfg.quantization).slice(0, 60) : 'none'}`,
  );
  return {
    streamSimple: makeMlxStreamSimple(new MlxModelHost([discovered])),
    model: {
      id: discovered.name,
      name: discovered.name,
      api: 'mlx',
      provider: 'mlx',
      baseUrl: 'mlx://local',
      reasoning: true,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 262144,
      maxTokens: 128,
    },
  };
}

describe.skipIf(!MODEL_PATH)('mlx provider live smoke', () => {
  let streamSimple: ReturnType<typeof makeMlxStreamSimple>;
  let model: Model<Api>;

  beforeAll(async () => {
    ({ streamSimple, model } = await bindHost(MODEL_PATH!));
  });

  it(
    'streams a real text turn to a stop final',
    async () => {
      const context: Context = {
        systemPrompt: SYSTEM,
        messages: [
          {
            role: 'user',
            content: 'What is the capital of France? Answer in one short sentence.',
            timestamp: Date.now(),
          },
        ],
      };
      const events = await collect(streamSimple(model, context, OPTIONS));

      expect(events[0]!.type).toBe('start');
      expect(events[events.length - 1]!.type).toBe('done');
      expect(events.some((e) => e.type === 'text_delta')).toBe(true);

      const message = finalMessage(events);
      expect(message.stopReason).toBe('stop');
      expect(visibleText(message)).toMatch(/paris/i);
      expect(message.usage.output).toBeGreaterThan(0);
      expect(message.usage.totalTokens).toBeGreaterThan(0);
    },
    TURN_TIMEOUT,
  );

});


// Separate describe: the round-trip binds a TOOL-CAPABLE checkpoint, not the
// smallest one. Gating on capability keeps `expect(stopReason).toBe('toolUse')`
// exactly as written instead of relaxing it to accept the 'stop' a 4-bit 0.8B
// actually returns.
describe.skipIf(!TOOL_MODEL_PATH)('mlx provider live smoke — tool round-trip', () => {
  let streamSimple: ReturnType<typeof makeMlxStreamSimple>;
  let model: Model<Api>;

  beforeAll(async () => {
    ({ streamSimple, model } = await bindHost(TOOL_MODEL_PATH!));
  });

  it(
    'completes a tool round-trip: toolUse final, then a continuation quoting the fabricated result',
    async () => {
      const weatherTool: Tool = {
        name: 'get_weather',
        description: 'Get current weather for a city',
        parameters: {
          type: 'object',
          properties: { location: { type: 'string', description: 'City name' } },
          required: ['location'],
        } as unknown as TSchema,
      };
      const context: Context = {
        systemPrompt: SYSTEM,
        messages: [
          {
            role: 'user',
            content:
              'What is the current weather in Paris? You must call the get_weather tool — do not answer from memory.',
            timestamp: Date.now(),
          },
        ],
        tools: [weatherTool],
      };

      // Turn 1: the model must emit a tool call.
      const callEvents = await collect(streamSimple(model, context, OPTIONS));
      const callMessage = finalMessage(callEvents);
      expect(callMessage.stopReason).toBe('toolUse');
      expect(callEvents.some((e) => e.type === 'toolcall_end')).toBe(true);

      const toolCall = callMessage.content.find(
        (part): part is Extract<AssistantMessage['content'][number], { type: 'toolCall' }> => part.type === 'toolCall',
      );
      expect(toolCall).toBeDefined();
      expect(toolCall!.name).toBe('get_weather');
      expect(String(toolCall!.arguments.location ?? '')).toMatch(/paris/i);

      // Turn 2: replay with the fabricated tool result appended.
      const toolResult: ToolResultMessage = {
        role: 'toolResult',
        toolCallId: toolCall!.id,
        toolName: 'get_weather',
        content: [{ type: 'text', text: '{"location":"Paris","condition":"sunny","temp_c":22}' }],
        isError: false,
        timestamp: Date.now(),
      };
      const continueContext: Context = {
        ...context,
        messages: [...context.messages, callMessage, toolResult],
      };
      const continueEvents = await collect(streamSimple(model, continueContext, OPTIONS));
      const continueMessage = finalMessage(continueEvents);
      expect(continueMessage.stopReason).toBe('stop');
      expect(visibleText(continueMessage)).toMatch(/sunny|22/i);
      // Warm replay on the shared prefix: the second call must reuse KV.
      expect(continueMessage.usage.cacheRead).toBeGreaterThan(0);
    },
    TURN_TIMEOUT,
  );
});
