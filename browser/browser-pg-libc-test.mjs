// **The chibicc card's separate-compilation path, in the wasm cdylib** (#1392) — the same exports
// `web/play.js` calls, driven from Node so the *wasm* build of them is gated (the Rust
// `pg_libc_asset.rs` suite gates the native build).
//
// The card used to compile the seeded playground libc into every program on every Run — guest C,
// identical every time, and nearly the whole compile cost. Now the libc's bodies are a committed
// prebuilt unit (`web/assets/pg_libc.temeno`), resident for the life of the page, and the user's C is
// compiled *decls-only* against the headers' prototypes and linked against it.
//
// This asserts the flow and prints the in-browser-engine numbers both ways:
//   1. `temen_link_lib_open(pg_libc.temeno)`            — once
//   2. `temen_run_onramp_fs(…, flags = -g | PROGRAM_UNIT)` — the user's TU → a small program unit
//   3. `temen_link_encode_lib(h, unit, "main")`         — → runnable module bytes (no text round trip)
//   4. `temen_run_onramp(module)`                       — it prints
// …against the old path (`temen_run_onramp_fs` whole-program → `temen_parse` → `temen_run_onramp`).
//
// Run: node browser/browser-pg-libc-test.mjs   (needs the wasm32 cdylib + both assets; SKIPs otherwise)
import { readFileSync, existsSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { engineImports } from './engine-imports.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const WASM = join(HERE, 'target/wasm32-unknown-unknown/release/temen_browser.wasm');
const CHIBICC = join(HERE, 'web/assets/chibicc.temen');
const PG_LIBC = join(HERE, 'web/assets/pg_libc.temeno');
for (const [p, what] of [[WASM, 'the wasm32 cdylib'], [CHIBICC, 'chibicc.temen'], [PG_LIBC, 'pg_libc.temeno']]) {
  if (!existsSync(p)) {
    console.log(`SKIP: ${what} not built (${p})`);
    process.exit(0);
  }
}

let failed = false;
const ok = (m) => console.log(`  ok: ${m}`);
const fail = (m) => { failed = true; console.log(`  FAIL: ${m}`); };

const CHIBICC_DEBUG_INFO = 1, CHIBICC_PROGRAM_UNIT = 2;
const SRC = `#include <stdio.h>
#include <stdlib.h>
int main(void) {
  for (int i = 1; i <= 3; i++) printf("i=%d\\n", i);
  char *b = malloc(32);
  snprintf(b, 32, "pi=%.2f", 3.14159);
  puts(b);
  fprintf(stdout, "and stdout\\n");
  return 42;
}
`;
const EXPECT = 'i=1\ni=2\ni=3\npi=3.14\nand stdout\n';

const { instance } = await WebAssembly.instantiate(readFileSync(WASM), engineImports());
const ex = instance.exports;
const u8 = () => new Uint8Array(ex.memory.buffer);
const enc = new TextEncoder(), dec = new TextDecoder();
// Every load re-reads `ex.memory.buffer`: `temen_alloc` can grow (and so detach) linear memory.
const load = (b) => { const p = Number(ex.temen_alloc(b.length)); u8().set(b, p); return p; };
const outBytes = () => u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + Number(ex.temen_stdout_len()));
const outText = () => dec.decode(outBytes());
const errText = () => dec.decode(u8().slice(Number(ex.temen_stderr_ptr()), Number(ex.temen_stderr_ptr()) + Number(ex.temen_stderr_len())));

const chibicc = new Uint8Array(readFileSync(CHIBICC));
const pgLibc = new Uint8Array(readFileSync(PG_LIBC));
const src = enc.encode(SRC);

// 1 — the prebuilt unit goes resident, once per page.
const tOpen = performance.now();
const libP = load(pgLibc);
const h = ex.temen_link_lib_open(libP, pgLibc.length);
const openMs = performance.now() - tOpen;
h >= 0 ? ok(`pg_libc.temeno resident (handle ${h}, ${pgLibc.length} B, ${openMs.toFixed(0)} ms once)`)
       : fail(`temen_link_lib_open declined: status ${ex.temen_status()} — stale asset? see AGENTS.md`);

// A helper for pass 1, either mode. Returns { ms, ir, status }.
const compile = (flags) => {
  const cP = load(chibicc), sP = load(src);
  const t = performance.now();
  ex.temen_run_onramp_fs(cP, chibicc.length, 0, 0, sP, src.length, flags);
  const ms = performance.now() - t;
  const status = Number(ex.temen_status());
  const ir = outBytes();
  ex.temen_dealloc(cP, chibicc.length);
  ex.temen_dealloc(sP, src.length);
  return { ms, ir, status };
};
const run = (moduleBytes) => {
  const mP = load(moduleBytes);
  const t = performance.now();
  const rv = ex.temen_run_onramp(mP, moduleBytes.length, 0, 0);
  const ms = performance.now() - t;
  const status = Number(ex.temen_status());
  const stdout = outText();
  ex.temen_dealloc(mP, moduleBytes.length);
  return { ms, rv: Number(rv), status, stdout };
};

// 2 + 3 — the new path: a program unit, linked against the resident libc straight to module bytes.
if (h >= 0) {
  const c = compile(CHIBICC_DEBUG_INFO | CHIBICC_PROGRAM_UNIT);
  c.status === 0 || c.status === 5 ? ok(`compiled a program unit: ${c.ir.length} B IR in ${c.ms.toFixed(0)} ms`)
                                  : fail(`program-unit compile: status ${c.status} — ${errText()}`);
  const unitP = load(c.ir), entry = enc.encode('main');
  const entryP = load(entry);
  const tLink = performance.now();
  const lok = ex.temen_link_encode_lib(h, unitP, c.ir.length, entryP, entry.length);
  const linkMs = performance.now() - tLink;
  const module = lok === 0 ? outBytes() : new Uint8Array();
  ex.temen_dealloc(unitP, c.ir.length);
  ex.temen_dealloc(entryP, entry.length);
  lok === 0 ? ok(`linked against the resident unit: ${module.length} B module in ${linkMs.toFixed(0)} ms`)
            : fail(`temen_link_encode_lib: status ${ex.temen_status()}`);

  if (lok === 0) {
    const r = run(module);
    r.stdout === EXPECT && r.rv === 42 && (r.status === 0 || r.status === 5)
      ? ok(`ran it: returned ${r.rv}, printed through the linked libc in ${r.ms.toFixed(0)} ms`)
      : fail(`run: ${JSON.stringify({ status: r.status, rv: r.rv, stdout: r.stdout })}`);
  }

  // 4 — the old path, for the number that matters: the libc's bodies compiled into the program.
  const w = compile(CHIBICC_DEBUG_INFO);
  const wIr = w.ir;
  const irP = load(wIr);
  const parsed = ex.temen_parse(irP, wIr.length) === 1
    ? u8().slice(Number(ex.temen_parse_ptr()), Number(ex.temen_parse_ptr()) + Number(ex.temen_parse_len()))
    : new Uint8Array();
  ex.temen_dealloc(irP, wIr.length);
  const wr = parsed.length ? run(parsed) : { stdout: '', rv: -1, status: -1, ms: 0 };
  wr.stdout === EXPECT
    ? ok('the whole-program path still agrees — same output either way')
    : fail(`whole-program parity: ${JSON.stringify({ status: wr.status, rv: wr.rv, stdout: wr.stdout })}`);

  console.log(
    `\n#1392 in the wasm engine: whole program ${w.ms.toFixed(0)} ms / ${wIr.length} B IR  →  ` +
    `program unit ${c.ms.toFixed(0)} ms / ${c.ir.length} B IR + link ${linkMs.toFixed(0)} ms  ` +
    `(${(w.ms / (c.ms + linkMs)).toFixed(1)}x, libc resident in ${openMs.toFixed(0)} ms once)`,
  );
  // 5 — the debugger's half (`temen_link_text_lib`): the linked program's IR *text*, carrying both
  // units' merged debug info, is what a DAP session launches from. Without the merge this text would
  // have no `debug.*` at all and stepping a separately-compiled program would be impossible.
  {
    const u = load(c.ir), e = load(enc.encode('main'));
    const tText = performance.now();
    const tok = ex.temen_link_text_lib(h, u, c.ir.length, e, 4);
    const textMs = performance.now() - tText;
    const text = tok === 0 ? outText() : '';
    ex.temen_dealloc(u, c.ir.length);
    ex.temen_dealloc(e, 4);
    tok === 0 && text.includes('debug.loc') && text.includes('"/in.c"') && text.includes('__pg_stdio_impl.h')
      ? ok(`linked IR text for a debug session: ${text.length} B in ${textMs.toFixed(0)} ms, ` +
           'carrying both units\' debug info')
      : fail(`temen_link_text_lib: status ${ex.temen_status()}, ` +
             `debug.loc=${text.includes('debug.loc')} in.c=${text.includes('"/in.c"')} ` +
             `libc=${text.includes('__pg_stdio_impl.h')}`);
  }

  ex.temen_link_lib_close(h);
  // A closed handle must decline rather than link against whatever is left in the slot.
  const entry2 = enc.encode('main'), e2 = load(entry2), u2 = load(c.ir);
  ex.temen_link_encode_lib(h, u2, c.ir.length, e2, entry2.length) < 0
    ? ok('a closed handle declines')
    : fail('a closed handle still linked');
}

ex.temen_dealloc(libP, pgLibc.length);
console.log(failed ? '\nFAILED' : '\nAll pg_libc card-path checks passed.');
process.exit(failed ? 1 : 0);
