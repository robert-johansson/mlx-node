/**
 * pi `Context` → native `ChatMessage[]` / `ToolDefinition[]` conversion.
 *
 * The provider bridge replays pi's full message history through
 * `ChatSession.primeHistory()` on every LLM call, so this conversion must
 * be deterministic and byte-stable: an unstable rendering (key-order
 * churn, nondeterministic joins) would change the token prefix between
 * replays and silently kill native KV-cache reuse.
 */

import type { Context, ImageContent, Message, TextContent, Tool } from '@earendil-works/pi-ai';
import type { ChatMessage, ToolDefinition } from '@mlx-node/lm';

/**
 * Fixed text of the synthetic user message that carries tool-result
 * images to the model (see {@link contextToChatMessages}). MUST stay
 * byte-stable and count-independent: it is replayed on every turn, so
 * any variation (image counts, tool names, timestamps) would change
 * the token prefix and kill native KV reuse for the rest of the
 * session.
 */
const TOOL_IMAGE_HOIST_TEXT = 'The image output of the preceding tool result is attached.';

/** Split pi content parts into joined text and decoded image bytes.
 * Within-message interleaving is NOT preserved: the native Jinja
 * serializer renders one text part followed by the image parts
 * (matching mlx-vlm's ordering), so text is joined first and images
 * follow. Cross-message ordering is exact. */
function splitParts(parts: ReadonlyArray<TextContent | ImageContent>): { text: string; images: Uint8Array[] } {
  const texts: string[] = [];
  const images: Uint8Array[] = [];
  for (const part of parts) {
    if (part.type === 'image') {
      images.push(new Uint8Array(Buffer.from(part.data, 'base64')));
    } else {
      texts.push(part.text);
    }
  }
  return { text: texts.join('\n'), images };
}

/** Per-message conversion (byte-stable joins). Never drops — the drop / orphan
 * repair lives in {@link contextToChatMessages}, mirroring pi's transformMessages. */
function convertMessage(message: Message): ChatMessage {
  switch (message.role) {
    case 'user': {
      if (typeof message.content === 'string') {
        return { role: 'user', content: message.content };
      }
      const { text, images } = splitParts(message.content);
      const converted: ChatMessage = { role: 'user', content: text };
      if (images.length > 0) converted.images = images;
      return converted;
    }
    case 'assistant': {
      // Thinking blocks are dropped: the native chat template re-renders
      // reasoning through its own <think> handling, and replayed thinking
      // would invalidate the KV prefix of every later turn.
      const text = message.content
        .filter((part): part is TextContent => part.type === 'text')
        .map((part) => part.text)
        .join('\n');
      const toolCalls = message.content
        .filter((part) => part.type === 'toolCall')
        .map((part) => ({ id: part.id, name: part.name, arguments: JSON.stringify(part.arguments) }));
      const converted: ChatMessage = { role: 'assistant', content: text };
      if (toolCalls.length > 0) converted.toolCalls = toolCalls;
      return converted;
    }
    case 'toolResult': {
      // Images are NOT attached here: the native Jinja serializer emits
      // image parts only for user-role messages, matching the model's
      // trained format (Qwen-VL vision tokens live in user turns; tool
      // responses are text inside <tool_response>). Tool-result images
      // are hoisted onto a synthetic user message by
      // {@link contextToChatMessages} instead.
      const { text } = splitParts(message.content);
      return {
        role: 'tool',
        content: text,
        toolCallId: message.toolCallId,
        isError: message.isError,
      };
    }
  }
}

/** Decoded image bytes of a pi tool result, or `null` when it has none. */
function toolResultImages(parts: ReadonlyArray<TextContent | ImageContent>): Uint8Array[] | null {
  const { images } = splitParts(parts);
  return images.length > 0 ? images : null;
}

/**
 * Convert a pi `Context` into the `ChatMessage[]` accepted by
 * `ChatSession.primeHistory()`.
 *
 * - `systemPrompt` becomes the leading `system` message.
 * - USER image parts ride the converted message's `images` field: the
 *   native session start extracts them in message order and the Jinja
 *   serializer emits one vision part per image at the message's position,
 *   so multiple images and images at multiple history positions both work.
 * - TOOL-RESULT image parts are HOISTED onto a synthetic user message
 *   (fixed text {@link TOOL_IMAGE_HOIST_TEXT}) pushed directly after the
 *   tool message. The native serializer emits image parts only for
 *   user-role messages — the model's trained format (Qwen-VL vision
 *   tokens live in user turns) — so attaching them to the tool message
 *   would render zero placeholders while the engine still extracted the
 *   bytes: a feature/placeholder count mismatch. The hoist is
 *   provider-internal; pi extensions keep returning images in tool
 *   results per the pi content model.
 * - Text-only models: an image-bearing history is rejected by the engine
 *   with a typed `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-prefixed error
 *   before rendering (no silent placeholder substitution).
 *
 * Two-pass mirror of pi's canonical `transformMessages` (pi-ai
 * `dist/api/transform-messages.js`). That transform normally sanitizes the
 * history INSIDE pi's built-in providers, but our custom `streamSimple` bypasses
 * it (and `defaultConvertToLlm` filters by role only), so the same two passes
 * must run here or a failed/interrupted turn reaches `primeHistory` unchanged:
 *
 *  1. DROP every assistant turn whose `stopReason` is `error` or `aborted` —
 *     partial or not. These incomplete turns (partial text, a half-emitted tool
 *     call) must not be replayed: after a native error (R2-3 resets the native
 *     cache) or an Esc/abort, priming the invalid partial turn garbles the
 *     continuation or leaves a dangling `<tool_call>` and corrupts the native
 *     `unresolvedOkToolCallCount`. A dropped turn's tool calls are NOT tracked.
 *  2. ORPHAN-REPAIR: track the tool-call ids of each RETAINED assistant and,
 *     before every following user/assistant message and at the end, synthesize a
 *     native tool result (`{ role: 'tool', content: 'No result provided',
 *     isError: true }`) for any tracked call with no matching `toolResult`
 *     (pi's `insertSyntheticToolResults`), so no assistant tool call is left
 *     unanswered in the primed history.
 *
 * The happy path (every assistant completes, every tool call answered) is
 * untouched, so the byte-stable joins that keep the replayed KV prefix stable
 * are preserved.
 */
export function contextToChatMessages(context: Context): ChatMessage[] {
  const messages: ChatMessage[] = [];
  if (context.systemPrompt) {
    messages.push({ role: 'system', content: context.systemPrompt });
  }

  // Orphan-repair state: the tool-call ids awaiting a result from the most
  // recent RETAINED assistant, and the result ids seen since.
  let pendingToolCallIds: string[] = [];
  let seenToolResultIds = new Set<string>();

  const flushOrphans = (): void => {
    if (pendingToolCallIds.length === 0) return;
    for (const id of pendingToolCallIds) {
      if (!seenToolResultIds.has(id)) {
        messages.push({ role: 'tool', content: 'No result provided', toolCallId: id, isError: true });
      }
    }
    pendingToolCallIds = [];
    seenToolResultIds = new Set();
  };

  for (const message of context.messages) {
    switch (message.role) {
      case 'user':
        flushOrphans();
        messages.push(convertMessage(message));
        break;
      case 'assistant': {
        flushOrphans();
        if (message.stopReason === 'error' || message.stopReason === 'aborted') {
          break; // dropped: not primed, and its tool calls are NOT tracked
        }
        const converted = convertMessage(message);
        messages.push(converted);
        if (converted.toolCalls && converted.toolCalls.length > 0) {
          // Native ToolCall.id is optional; only ids can be matched against a
          // tool result, so an id-less call is never tracked for orphan repair.
          pendingToolCallIds = converted.toolCalls.map((tc) => tc.id).filter((id): id is string => id !== undefined);
          seenToolResultIds = new Set();
        }
        break;
      }
      case 'toolResult': {
        seenToolResultIds.add(message.toolCallId);
        messages.push(convertMessage(message));
        // Tool-image hoist: the image bytes ride a synthetic user message
        // DIRECTLY after the tool result, preserving temporal order across
        // multiple image-bearing tool calls. Pushed without flushing orphan
        // state — sibling tool calls from the same assistant fan-out may
        // still be awaiting results, and the hoist must not trigger
        // synthetic "No result provided" entries for them.
        const images = toolResultImages(message.content);
        if (images) {
          messages.push({ role: 'user', content: TOOL_IMAGE_HOIST_TEXT, images });
        }
        break;
      }
    }
  }
  flushOrphans();
  return messages;
}

/**
 * Convert pi `Tool[]` (TypeBox-built plain JSON Schema objects) into the
 * native OpenAI-style `ToolDefinition[]`.
 *
 * The NAPI layer requires `parameters.properties` as a JSON string;
 * `JSON.stringify` preserves the schema's own key order, keeping the
 * rendered tool block byte-stable across replays. Returns `undefined`
 * for an absent or empty tool list so `ChatConfig.tools` stays unset.
 */
export function toolsToDefinitions(tools: Tool[] | undefined): ToolDefinition[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  return tools.map((tool) => {
    // pi's Tool.parameters is a TSchema — at runtime a plain JSON Schema
    // object (TypeBox kind markers live on symbols, which JSON ignores).
    const schema = tool.parameters as { properties?: Record<string, unknown>; required?: string[] };
    return {
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: {
          type: 'object' as const,
          properties: JSON.stringify(schema.properties ?? {}),
          required: schema.required,
        },
      },
    };
  });
}
