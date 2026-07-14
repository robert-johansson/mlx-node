/**
 * Minimal `mlx agent` custom-tool extension — the template for
 * participant-tool extensions (work-order item 3, genmlx-jmjo).
 *
 * Run it:
 *
 *   mlx agent --model mlx/<model> --no-builtin-tools --extension examples/echo-tool.ts \
 *     -p "call echo_tool with the word hello, then tell me what it returned"
 *
 * What this demonstrates:
 *
 * - `pi.registerTool(defineTool({...}))` — tool name, TypeBox parameter
 *   schema, async execute. With `--no-builtin-tools` the session exposes
 *   ONLY the tools registered here (no bash/write/edit).
 *
 * - PERMISSION GATE: the agent's gate intercepts the built-in `bash` /
 *   `write` / `edit` tools BY NAME (`GATED_TOOLS` in
 *   `packages/agent/src/extensions/permission-gate.ts`). Custom tools are
 *   never gated — they run without approval prompts and without
 *   `MLX_AGENT_AUTO_APPROVE`, in both interactive and `-p` runs. If your
 *   extension needs its own approval flow, implement it inside `execute`.
 *
 * - IMAGES IN TOOL RESULTS (`show_image`): return an `ImageContent` part
 *   (`{ type: 'image', data: <base64>, mimeType: 'image/png' }`) alongside
 *   text. On a VLM the provider carries the bytes to the model (tool-result
 *   images are hoisted onto a synthetic user message internally — the
 *   Qwen-VL trained format keeps vision tokens in user turns; your
 *   extension just returns image parts per the pi content model). On a
 *   text-only model an image-bearing turn is rejected with a typed error.
 *
 * - Imports resolve anywhere on disk: the extension loader (jiti) aliases
 *   `@earendil-works/*` to the agent's own copies, so an extension file
 *   needs no node_modules of its own.
 */
import { readFileSync } from 'node:fs';

import { Type } from '@earendil-works/pi-ai';
import { defineTool, type ExtensionAPI } from '@earendil-works/pi-coding-agent';

export default function echoToolExtension(pi: ExtensionAPI) {
  pi.registerTool(
    defineTool({
      name: 'echo_tool',
      label: 'Echo',
      description: 'Echoes back the provided text, uppercased. Use it when asked to echo something.',
      parameters: Type.Object({
        text: Type.String({ description: 'The text to echo back' }),
      }),
      async execute(_toolCallId, params) {
        return {
          content: [{ type: 'text', text: `ECHO: ${params.text.toUpperCase()}` }],
          details: {},
        };
      },
    }),
  );

  pi.registerTool(
    defineTool({
      name: 'show_image',
      label: 'Show image',
      description: 'Shows the image at the given path. Call this to see an image file.',
      parameters: Type.Object({
        path: Type.String({ description: 'Filesystem path of a PNG or JPEG image' }),
      }),
      async execute(_toolCallId, params) {
        const bytes = readFileSync(params.path);
        return {
          content: [
            { type: 'text', text: 'Image attached.' },
            { type: 'image', data: bytes.toString('base64'), mimeType: 'image/png' },
          ],
          details: {},
        };
      },
    }),
  );
}
