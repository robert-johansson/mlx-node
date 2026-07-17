/**
 * Best-of-K verifier example (genmlx-maww) — the seam contract for
 * extension authors.
 *
 * On the genmlx provider with `MLX_AGENT_BEST_OF_K=<K>`, each turn decodes
 * K candidate completions in ONE batched forward and calls the registered
 * verifier once with all of them; the winner streams as the turn's reply.
 *
 * Extension files can only import `@earendil-works/*` (pi's jiti alias
 * set), so the verifier registers through the documented global:
 * `globalThis.__GENMLX_TOOL_VERIFIER__`. In-process embedders use
 * `setGenmlxToolVerifier` from `@mlx-node/agent` instead.
 *
 * Contract (see provider/genmlx/genmlx-verifier.ts):
 *   in:  {"candidates": [{"index", "text", "finishReason",
 *                         "toolCalls", "toolCallErrors"}, ...]}
 *   out: {"winner": <index>}  or  {"scores": [K numbers]} (argmax,
 *        ties -> lowest index)
 * Fallbacks: verifier absent / throws / times out / malformed -> candidate 0.
 *
 * This example scores executability the way an instrument-style verifier
 * would: a candidate with a cleanly-parsed tool call beats prose; parse
 * errors disqualify. Swap `score` for a real verifier (run the candidate
 * against the scene, typecheck the result) without touching the provider.
 *
 * Run:
 *   MLX_AGENT_BEST_OF_K=4 mlx agent --model genmlx/<model> \
 *     --no-builtin-tools --extension examples/echo-tool.ts \
 *     --extension examples/best-of-k-verifier.ts \
 *     -p "call echo_tool with the word hello"
 */
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';

interface Candidate {
  index: number;
  text: string;
  finishReason: string;
  toolCalls: Array<{ name: string }>;
  toolCallErrors: string[];
}

function score(candidate: Candidate): number {
  if (candidate.toolCallErrors.length > 0) return -1;
  if (candidate.toolCalls.length > 0) return 1;
  return 0;
}

export default function bestOfKVerifierExtension(_pi: ExtensionAPI) {
  (globalThis as Record<string, unknown>).__GENMLX_TOOL_VERIFIER__ = (candidatesJson: string): string => {
    const { candidates } = JSON.parse(candidatesJson) as { candidates: Candidate[] };
    return JSON.stringify({ scores: candidates.map(score) });
  };
}
