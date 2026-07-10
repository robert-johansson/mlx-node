#!/usr/bin/env node
// Build the GenMLX superset NAPI addon -> packages/genmlx-core/index.node.
//
// This is the addon GenMLX loads at runtime (@genmlx/core -> ./index.node).
// It is NOT produced by `yarn build:native` (that builds the sibling
// @mlx-node/core). Run it with:  yarn workspace @genmlx/core build
//
// It reproduces the historical ad-hoc command (commit 5f3c8ee):
//   napi build --release --manifest-path crates/genmlx-core/Cargo.toml \
//     -o packages/genmlx-core
// (no --platform => the output name stays index.node, matching package.json
// "main"), then colocates the Metal shader libs NEXT TO the addon — but ONLY on
// darwin. On Linux/CUDA there are no metallibs (MLX loads CUDA kernels at
// runtime), so the colocation step is skipped entirely. See
// docs/cuda-port-runbook.md §3c.

import { spawnSync } from 'node:child_process';
import { readdir, stat, copyFile, cp, rm } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url)); // packages/genmlx-core
const repoRoot = join(here, '..', '..'); // mlx-node/
const manifest = join(repoRoot, 'crates', 'genmlx-core', 'Cargo.toml');
const napiBin = join(repoRoot, 'node_modules', '.bin', 'napi');

// Bare napi build (NO --platform): emits index.node (target-agnostic name).
// Extra CLI args pass through, e.g. `yarn workspace @genmlx/core build --target
// aarch64-unknown-linux-gnu` on a cross host (native build is preferred — CUDA
// is not cross-compilable from macOS, runbook §3a).
const args = [
  'build',
  '--release',
  '--manifest-path',
  manifest,
  '-o',
  here,
  ...process.argv.slice(2),
];
const r = spawnSync(napiBin, args, { stdio: 'inherit', cwd: repoRoot });
if (r.status !== 0) process.exit(r.status ?? 1);

// Metallibs are a macOS/Metal concern only. On Linux/CUDA, colocate the CUDA
// JIT headers instead: MLX's NVRTC JIT resolves cute/cutlass/cccl includes via
// `current_binary_dir()/../include` (jit_module.cpp include_path_args), i.e.
// packages/include for this statically-linked addon. The mlx-sys cmake build
// installs that header tree into its OUT_DIR; without this copy every
// cute-dependent kernel (gather_qmm / qmm_naive / qmm_sm80...) dies at NVRTC
// with `cannot open source file "cute/numeric/numeric_types.hpp"` on a cold
// PTX cache (genmlx-q5uq).
if (process.platform !== 'darwin') {
  await colocateJitHeaders();
  process.exit(0);
}
await colocateMetallibs();

// Copy the newest mlx-sys OUT_DIR include tree (mlx/, cute/, cutlass/, cccl/)
// to packages/include, where include_path_args() probes for it. Handles both
// build layouts: target/release/build/ and target/<arch>/release/build/.
async function colocateJitHeaders() {
  const targetDir = join(repoRoot, 'target');
  const candidates = [];
  const scanBuildDir = async (buildDir) => {
    let dirs;
    try {
      dirs = await readdir(buildDir);
    } catch {
      return;
    }
    for (const d of dirs) {
      if (!d.startsWith('mlx-sys-')) continue;
      const inc = join(buildDir, d, 'out', 'include');
      try {
        const [cute, st] = [await stat(join(inc, 'cute')), await stat(inc)];
        if (cute.isDirectory()) candidates.push({ inc, mtime: st.mtimeMs });
      } catch {
        // no installed headers in this OUT_DIR; keep looking
      }
    }
  };
  await scanBuildDir(join(targetDir, 'release', 'build'));
  let archDirs = [];
  try {
    archDirs = await readdir(targetDir);
  } catch {
    // no target dir at all — nothing to scan
  }
  for (const arch of archDirs) {
    await scanBuildDir(join(targetDir, arch, 'release', 'build'));
  }
  if (candidates.length === 0) {
    throw new Error(
      '[genmlx-core] no mlx-sys OUT_DIR include tree with cute/ found under ' +
        'target/{,<arch>/}release/build/mlx-sys-*/out/include — the CUDA JIT ' +
        'would fail on cute-dependent kernels (gather_qmm). Did the mlx-sys ' +
        'cmake install step run?',
    );
  }
  candidates.sort((a, b) => b.mtime - a.mtime);
  const src = candidates[0].inc;
  const dst = join(repoRoot, 'packages', 'include');
  await rm(dst, { recursive: true, force: true });
  await cp(src, dst, { recursive: true });
  console.log(`[genmlx-core] colocated CUDA JIT headers ${src} -> ${dst}`);
}

// Find + copy mlx.metallib + paged_attn.metallib next to index.node. Both are
// required on darwin (MLX/paged-attn load them via dladdr next to the binary).
// Robust to the two build-output layouts seen in the tree:
//   mlx.metallib       : target/<arch>/release/build/mlx-sys-*/out/lib/
//   paged_attn.metallib: same libDir, OR target/<arch>/release/build/mlx-paged-attn-*/out/
async function colocateMetallibs() {
  const targetDir = join(repoRoot, 'target');
  const mlx = await findUnder(targetDir, (releaseBuild) =>
    matchInDirs(releaseBuild, 'mlx-sys-', ['out', 'lib', 'mlx.metallib']),
  );
  if (!mlx) {
    throw new Error(
      '[genmlx-core] mlx.metallib not found under target/<arch>/release/build/mlx-sys-*/out/lib/',
    );
  }
  let paged = await findUnder(targetDir, (releaseBuild) =>
    matchInDirs(releaseBuild, 'mlx-sys-', ['out', 'lib', 'paged_attn.metallib']),
  );
  if (!paged) {
    paged = await findUnder(targetDir, (releaseBuild) =>
      matchInDirs(releaseBuild, 'mlx-paged-attn-', ['out', 'paged_attn.metallib']),
    );
  }
  if (!paged) {
    throw new Error(
      '[genmlx-core] paged_attn.metallib not found under target/<arch>/release/build/' +
        '{mlx-sys-*/out/lib,mlx-paged-attn-*/out}/ — check that build.rs ran ' +
        'compile_paged_attn_metallib (runbook §3b).',
    );
  }
  for (const [name, src] of [
    ['mlx.metallib', mlx],
    ['paged_attn.metallib', paged],
  ]) {
    const dst = join(here, name);
    await copyFile(src, dst);
    console.log(`[genmlx-core] copied ${name} -> ${dst}`);
  }
}

// Walk target/<arch>/release/build and return the first path `match` resolves.
async function findUnder(targetDir, match) {
  let archDirs;
  try {
    archDirs = await readdir(targetDir);
  } catch {
    return null;
  }
  for (const arch of archDirs) {
    const releaseBuild = join(targetDir, arch, 'release', 'build');
    const hit = await match(releaseBuild);
    if (hit) return hit;
  }
  return null;
}

// In `releaseBuild`, find a sub-dir starting with `prefix` whose `...rest` path
// exists; return that full path or null.
async function matchInDirs(releaseBuild, prefix, rest) {
  let dirs;
  try {
    dirs = await readdir(releaseBuild);
  } catch {
    return null;
  }
  for (const d of dirs) {
    if (!d.startsWith(prefix)) continue;
    const p = join(releaseBuild, d, ...rest);
    try {
      await stat(p);
      return p;
    } catch {
      // not at this path; keep looking
    }
  }
  return null;
}
