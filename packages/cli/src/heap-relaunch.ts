/**
 * One-shot heap-headroom re-exec for `mlx agent` (genmlx-djw6).
 *
 * The genmlx provider's owned-forward model load runs inside THIS process
 * (nbb bridge -> `@genmlx/core`), and it needs more JS heap than Node's
 * default ~4GB old space — without it the load dies in an OOM mark-compact.
 * V8 cannot raise the limit after startup (`v8.setFlagsFromString
 * ('--max-old-space-size=…')` is a no-op on Node 26 — measured: the limit
 * stays 4192MB and allocation aborts at 4GB), so the agent command re-execs
 * itself ONCE with `--max-old-space-size` when the current limit is below
 * target. The v1 mlx provider doesn't need the headroom but is unaffected
 * beyond the extra fork (the limit is a cap, not a reservation), and the
 * wizard may pick a genmlx model after startup — so the relaunch is
 * unconditional for the agent command rather than gated on `--model`.
 *
 * Knobs: `MLX_AGENT_MAX_OLD_SPACE_MB` overrides the 12288 target
 * (`0` disables the relaunch entirely — e.g. for embedders that manage
 * their own flags). `MLX_AGENT_HEAP_RELAUNCHED=1` is the internal marker
 * that stops recursion.
 */

import { spawnSync } from 'node:child_process';
import v8 from 'node:v8';

/** Default `--max-old-space-size` target (MB) — the measured-working value
 *  for owned 35B-class model load under the nbb bridge. */
export const DEFAULT_AGENT_HEAP_MB = 12288;

/**
 * Pure decision: the `--max-old-space-size` MB the agent process should be
 * relaunched with, or `null` when no relaunch is needed (already relaunched,
 * running under Bun — whose heap is not V8-flag-governed —, disabled, a
 * malformed override, or the current limit already meets the target).
 */
export function heapRelaunchTargetMb(
  env: Record<string, string | undefined>,
  currentLimitMb: number,
  isBun: boolean,
): number | null {
  if (isBun || env.MLX_AGENT_HEAP_RELAUNCHED === '1') {
    return null;
  }
  const raw = env.MLX_AGENT_MAX_OLD_SPACE_MB;
  const targetMb = raw === undefined || raw === '' ? DEFAULT_AGENT_HEAP_MB : Number(raw);
  if (!Number.isFinite(targetMb) || targetMb <= 0) {
    return null;
  }
  return currentLimitMb >= targetMb ? null : targetMb;
}

/**
 * Re-exec `mlx agent` with heap headroom when needed. Returns the child's
 * exit code to propagate, or `null` when the caller should just keep running
 * in this process. Blocking by design: the child owns the TTY (stdio
 * inherit) for the interactive TUI and print mode alike.
 */
export function relaunchAgentWithHeapHeadroom(agentArgs: string[]): number | null {
  const currentLimitMb = v8.getHeapStatistics().heap_size_limit / (1024 * 1024);
  const targetMb = heapRelaunchTargetMb(process.env, currentLimitMb, process.versions.bun !== undefined);
  if (targetMb === null) {
    return null;
  }
  const result = spawnSync(
    process.execPath,
    [`--max-old-space-size=${targetMb}`, process.argv[1], 'agent', ...agentArgs],
    {
      stdio: 'inherit',
      env: { ...process.env, MLX_AGENT_HEAP_RELAUNCHED: '1' },
    },
  );
  if (result.signal !== null) {
    // Killed by signal (Ctrl-C lands on the whole foreground group) —
    // conventional shell encoding of a signal death.
    return 128 + (result.signal === 'SIGINT' ? 2 : result.signal === 'SIGKILL' ? 9 : 15);
  }
  return result.status ?? 1;
}
