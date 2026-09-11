// #1373 — per-run link floor: `temen_link_run` (decodes the library every call) vs
// `temen_link_lib_open` once + `temen_link_run_lib` per run. Same program, same library, best-of-N.
//
//   node browser/bench_link_lib.mjs <temen_browser.wasm> <lib.temen|.temt> <prog unit> [entry] [N]
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { engineImports } from './engine-imports.mjs';

const [wasmPath, libPath, progPath, entryName = '__jacl_entry', nArg = '30'] = process.argv.slice(2);
const N = Number(nArg);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), engineImports());
const ex = instance.exports, u8 = () => new Uint8Array(ex.memory.buffer), enc = new TextEncoder();
const load = (b) => { const p = Number(ex.temen_alloc(b.length)); u8().set(b, p); return p; };
const lib = new Uint8Array(readFileSync(libPath)), prog = new Uint8Array(readFileSync(progPath)), entry = enc.encode(entryName);
const libP = load(lib), progP = load(prog), entryP = load(entry);
const out = () => new TextDecoder().decode(u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + Number(ex.temen_stdout_len())));
const best = (f) => { let b = Infinity, o; for (let i = 0; i < N; i++) { const t = performance.now(); f(); const d = performance.now() - t; if (d < b) { b = d; o = out(); } } return { b, o }; };

const a = best(() => ex.temen_link_run(progP, prog.length, libP, lib.length, entryP, entry.length, 0, 0));
const sa = ex.temen_status();
const tOpen = performance.now(); const h = ex.temen_link_lib_open(libP, lib.length); const openMs = performance.now() - tOpen;
const b = best(() => ex.temen_link_run_lib(h, progP, prog.length, entryP, entry.length, 0, 0));
const sb = ex.temen_status();
ex.temen_link_lib_close(h);
console.log(`lib ${lib.length} B, prog ${prog.length} B, entry ${entryName}, best of ${N}`);
console.log(`temen_link_run        : ${a.b.toFixed(2)} ms  (status ${sa})`);
console.log(`temen_link_lib_open   : ${openMs.toFixed(2)} ms once (handle ${h})`);
console.log(`temen_link_run_lib    : ${b.b.toFixed(2)} ms  (status ${sb})  → ${(a.b / b.b).toFixed(2)}x, saves ${(a.b - b.b).toFixed(2)} ms/run`);
console.log(`stdout parity: ${a.o === b.o ? 'OK' : 'MISMATCH'}`);
