/**
 * Attach to a RUNNING (or finished) `mlx agent` session and print each
 * think block, tool call, and tool result as it lands — the JSONL
 * observation channel (work-order item 2, genmlx-ylgv).
 *
 * Usage:
 *
 *   oxnode examples/watch-session.ts <session.jsonl | sessions-dir>
 *
 * With a directory, the most recently modified `*.jsonl` inside is
 * followed. Records are appended by pi via `appendFileSync` once per
 * COMPLETED message (session-manager.ts), so granularity is per think
 * block / tool call — a tool call appears before its result executes.
 * For token-level deltas use `--mode rpc` instead (see docs/cli.md
 * "Observing and driving a session").
 *
 * Session JSONL schema (pi 0.80.6, stable fields relied on here):
 *   every line: { type, id, parentId?, timestamp }
 *   type "session":  + { cwd, version }              (first line)
 *   type "message":  + { message: pi Message }
 *     message.role "user":      content: string | parts
 *     message.role "assistant": content parts:
 *         { type: "thinking", thinking }             — think block
 *         { type: "text", text }                     — spoken text
 *         { type: "toolCall", id, name, arguments }  — tool call
 *       + stopReason, usage { input, output, ... }
 *     message.role "toolResult": toolCallId, toolName, isError,
 *         content parts: { type: "text" | "image", ... }
 *   type "model_change" / "thinking_level_change": administrative.
 * `parentId` chains records into pi's session TREE (forks share a file).
 */
import { readdirSync, readSync, openSync, closeSync, statSync, fstatSync } from 'node:fs';
import { join } from 'node:path';

function resolveTarget(arg: string): string {
  if (statSync(arg).isDirectory()) {
    const files = readdirSync(arg)
      .filter((f) => f.endsWith('.jsonl'))
      .map((f) => join(arg, f))
      .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
    if (files.length === 0) throw new Error(`no .jsonl files in ${arg}`);
    return files[0]!;
  }
  return arg;
}

function stamp(ts: unknown): string {
  // pi writes ISO-8601 strings (e.g. "2026-07-14T03:24:45.552Z").
  if (typeof ts === 'string') return ts.slice(11, 23);
  return typeof ts === 'number' ? new Date(ts).toISOString().slice(11, 23) : '--:--:--.---';
}

function clip(s: string, n = 2000): string {
  return s.length > n ? `${s.slice(0, n)} …[${s.length - n} more chars]` : s;
}

function render(rec: Record<string, any>): void {
  const t = stamp(rec.timestamp);
  if (rec.type !== 'message') {
    if (rec.type === 'session') console.log(`${t}  === session ${rec.id} (cwd ${rec.cwd}) ===`);
    else console.log(`${t}  [${rec.type}]`);
    return;
  }
  const msg = rec.message ?? {};
  if (msg.role === 'user') {
    const text = typeof msg.content === 'string'
      ? msg.content
      : (msg.content ?? []).map((p: any) => (p.type === 'image' ? '<image>' : p.text)).join('\n');
    console.log(`${t}  [user] ${clip(text, 400)}`);
    return;
  }
  if (msg.role === 'toolResult') {
    const parts = (msg.content ?? []).map((p: any) =>
      p.type === 'image' ? `<image ${Buffer.from(p.data ?? '', 'base64').length} bytes>` : p.text,
    );
    console.log(`${t}  [tool-result${msg.isError ? ' ERROR' : ''}] ${msg.toolName}#${msg.toolCallId}: ${clip(parts.join(' | '), 600)}`);
    return;
  }
  if (msg.role === 'assistant') {
    for (const part of msg.content ?? []) {
      if (part.type === 'thinking') console.log(`${t}  [think] ${clip(part.thinking)}`);
      else if (part.type === 'text') console.log(`${t}  [say] ${clip(part.text)}`);
      else if (part.type === 'toolCall') console.log(`${t}  [tool-call] ${part.name}#${part.id} ${clip(JSON.stringify(part.arguments), 600)}`);
    }
    if (msg.stopReason) console.log(`${t}  [end-of-message] stopReason=${msg.stopReason}`);
  }
}

async function main(): Promise<void> {
  const arg = process.argv[2];
  if (!arg) {
    console.error('usage: oxnode examples/watch-session.ts <session.jsonl | sessions-dir>');
    process.exit(1);
  }
  const path = resolveTarget(arg);
  console.log(`watching ${path}`);
  const fd = openSync(path, 'r');
  let offset = 0;
  let tail = '';
  const drain = (): void => {
    const size = fstatSync(fd).size;
    if (size <= offset) return;
    const buf = Buffer.alloc(size - offset);
    readSync(fd, buf, 0, buf.length, offset);
    offset = size;
    tail += buf.toString('utf8');
    let nl: number;
    while ((nl = tail.indexOf('\n')) >= 0) {
      const line = tail.slice(0, nl).trim();
      tail = tail.slice(nl + 1);
      if (!line) continue;
      try {
        render(JSON.parse(line));
      } catch {
        console.log(`  [unparseable line] ${clip(line, 120)}`);
      }
    }
  };
  drain();
  // appendFileSync-per-record makes 200ms polling both simple and robust
  // (fs.watch coalesces and misses on some filesystems).
  const timer = setInterval(drain, 200);
  process.on('SIGINT', () => {
    clearInterval(timer);
    closeSync(fd);
    process.exit(0);
  });
}

void main();
