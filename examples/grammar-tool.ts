/**
 * Per-argument grammar constraint example (genmlx-3g0t) — the seam
 * contract for extension authors.
 *
 * On the genmlx provider, a JSON-Schema `pattern` on a string tool
 * parameter is compiled (with the whole declared toolset) into a
 * tool-call DFA and applied as a per-decode-step logit mask: a tool call
 * whose argument violates the pattern is UNREPRESENTABLE at sampling
 * time — no typecheck-and-retry loop needed. TypeBox's standard
 * `pattern` option is all it takes; nothing provider-internal leaks into
 * the extension.
 *
 * Notes:
 * - The regex runs on the grammar engine's dialect (printable ASCII,
 *   char classes, alternation, `{m,n}` repetition; no lookahead).
 * - Constraint activation is annotation-presence: a toolset with no
 *   `pattern` anywhere decodes exactly as before.
 * - On the v1 `mlx` provider the annotation is inert (documentation
 *   only) — pi's own validation still applies.
 * - `x-genmlx-grammar: "cljs"` (reader-level ClojureScript constraint)
 *   is reserved and currently rejected with a typed error.
 *
 * Run:
 *   mlx agent --model genmlx/<model> --no-builtin-tools \
 *     --extension examples/grammar-tool.ts \
 *     -p "Use set_point to set the point at row 999999, column 888888."
 * The emitted xy is grammar-bound to at most 3 digits per coordinate.
 */
import { Type } from '@earendil-works/pi-ai';
import { defineTool, type ExtensionAPI } from '@earendil-works/pi-coding-agent';

export default function grammarToolExtension(pi: ExtensionAPI) {
  pi.registerTool(
    defineTool({
      name: 'set_point',
      label: 'Set point',
      description: 'Set a 2D point on the board.',
      parameters: Type.Object({
        xy: Type.String({
          description: 'Coordinates ROW,COL — each an integer with at most 3 digits',
          pattern: '-?[0-9]{1,3},-?[0-9]{1,3}',
        }),
      }),
      async execute(_toolCallId, params) {
        return {
          content: [{ type: 'text', text: `Point set at ${params.xy}.` }],
          details: {},
        };
      },
    }),
  );
}
