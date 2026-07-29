/**
 * Minimal ambient declaration for Bun's built-in SQLite module. This package
 * compiles with `types: ["node"]` and no Bun type package, so the conditional
 * `await import('bun:sqlite')` in `db/open.ts` needs a local declaration.
 * Only the surface `db/open.ts` touches is declared — the module itself ships
 * inside the Bun binary and is imported exclusively when running under Bun.
 */
declare module 'bun:sqlite' {
  export class Database {
    constructor(filename: string);
    exec(sql: string): void;
    prepare(sql: string): {
      get(...params: unknown[]): unknown;
      all(...params: unknown[]): unknown[];
      run(...params: unknown[]): unknown;
    };
    close(): void;
  }
}
