/**
 * Process-level native-host latch — the SIGTRAP wall, made explicit.
 *
 * One agent process may load EITHER `@mlx-node/core` (via `@mlx-node/lm`,
 * the v1 ChatSession provider) OR `@genmlx/core` (via the nbb bridge, the
 * genmlx provider), never both: the two addons carry duplicate native MLX
 * runtimes and loading the second dlopen aborts the process (observed
 * SIGTRAP; see bean genmlx-w68q). This module turns that hard crash into a
 * clear, catchable error thrown BEFORE the offending dlopen.
 *
 * Contract: every code path that is about to trigger a native addon load
 * calls {@link claimNativeOwner} first — the first claim wins and pins the
 * process; a later claim for the OTHER host throws. Both model hosts import
 * their addon lazily (dynamic `import()` on first model use), so merely
 * registering both providers never touches native code.
 */

export type NativeOwner = 'mlx' | 'genmlx';

let owner: NativeOwner | null = null;

/**
 * Pin this process to `candidate` (idempotent for the same host). Throws —
 * before any dlopen — when the process is already pinned to the other host.
 */
export function claimNativeOwner(candidate: NativeOwner): void {
  if (owner === null) {
    owner = candidate;
    return;
  }
  if (owner !== candidate) {
    throw new Error(
      `native-owner: this process already loaded the '${owner}' native host; ` +
        `loading '${candidate}' too would dlopen a second MLX runtime and abort the process (SIGTRAP). ` +
        `Run models from one provider per process — start a fresh \`mlx agent\` for '${candidate}' models.`,
    );
  }
}

/** The host this process is pinned to, or null before any model use. */
export function nativeOwner(): NativeOwner | null {
  return owner;
}

/** Test-only: unpin the process latch. */
export function resetNativeOwnerForTests(): void {
  owner = null;
}
