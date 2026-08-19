// #839: bench the single-shot tier-up path vs the plain bytecode path, per card, with event
// counts — the "measure what the pump buys or costs" task. For each (card, workload) it runs:
//   1. plain bytecode:  svm_run_onramp
//   2. the playground path: runJitModule (whole-program JIT → tier-up pump → coop, as play.js)
// asserting stdout parity, and reporting wall-clock plus TIERUP/JIT_INVOKE/bounce counts (counted
// through a Proxy over the cdylib exports — zero engine changes). Cold (first Run, includes the
// open's emit + WebAssembly.compile) and warm (best of N, Module cached) are reported separately,
// the #753/#803 caching split. Runs on the threads cdylib on Node's V8, no Playwright.
//
// Usage:  node bench_tierup_cards.mjs [module.wasm]   (build the threads cdylib first)
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { runJitModule } from './web/wasmjit-module.js';
import { engineImports } from './engine-imports.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2] ?? join(ROOT, 'target/wasm32-unknown-unknown/release/svm_browser.wasm');
const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));

const enc = new TextEncoder();
const dec = new TextDecoder();
const u8 = () => new Uint8Array(memory.buffer);
const readStdout = () => dec.decode(u8().slice(
  Number(ex.svm_stdout_ptr()), Number(ex.svm_stdout_ptr()) + Number(ex.svm_stdout_len())));

// Event counters, observed from outside: one TIERUP service reads svm_coop_func once; a
// JIT_INVOKE reads svm_coop_jit_wasm_ptr once; every cross-tier bounce goes through
// svm_coop_call_interp (#1026: the coop driver is the one fallback tier; the pump is gone).
let forcePump = false;
const counts = { tierups: 0, invokes: 0, bounces: 0, path: 'interp' };
const resetCounts = () => { counts.tierups = 0; counts.invokes = 0; counts.bounces = 0; counts.path = 'interp'; };
const exCounted = Object.fromEntries(Object.entries(Object.getOwnPropertyDescriptors(ex)).map(([k, d]) => {
  const v = d.value;
  if (typeof v !== 'function') return [k, v];
  if (k === 'svm_coop_func') return [k, (...a) => { counts.tierups++; return v(...a); }];
  if (k === 'svm_coop_jit_wasm_ptr') return [k, (...a) => { counts.invokes++; return v(...a); }];
  if (k === 'svm_coop_call_interp') return [k, (...a) => { counts.bounces++; return v(...a); }];
  if (k === 'svm_onramp_jit_run_call_interp') return [k, (...a) => { counts.bounces++; return v(...a); }];
  if (k === 'svm_onramp_jit_run_open')
    return [k, (...a) => {
      if (forcePump) return -2; // pretend "_start not emittable" so runJitModule falls to coop
      const r = v(...a);
      if (r === 0) counts.path = 'whole-program';
      return r;
    }];
  if (k === 'svm_coop_open')
    return [k, (...a) => { const r = v(...a); if (r === 0) counts.path = 'coop'; return r; }];
  return [k, v];
}));

function bytecodeRun(modBytes, stdinBytes) {
  const mp = Number(ex.svm_alloc(modBytes.length));
  u8().set(modBytes, mp);
  const sp = Number(ex.svm_alloc(stdinBytes.length));
  u8().set(stdinBytes, sp);
  const t0 = performance.now();
  ex.svm_run_onramp(mp, modBytes.length, sp, stdinBytes.length);
  const ms = performance.now() - t0;
  const out = readStdout();
  const st = ex.svm_status();
  ex.svm_dealloc(mp, modBytes.length);
  ex.svm_dealloc(sp, stdinBytes.length);
  return { ms, out, st };
}

async function pumpRun(modBytes, stdinBytes, cacheKey) {
  resetCounts();
  const t0 = performance.now();
  await runJitModule(exCounted, memory, modBytes, stdinBytes, cacheKey);
  const ms = performance.now() - t0;
  return { ms, out: readStdout(), st: ex.svm_status(), ...counts };
}

const N = 3;
const cards = [
  ['hello_c', 'web/assets/hello_c.svmb', ''],
  ['lua', 'web/assets/lua_snapshot.svmb',
    'local s = 0\nfor i = 1, 300000 do s = s + i % 7 end\nprint(s)\n'],
  ['quickjs', 'web/assets/qjs_repl.svmb',
    'let s = 0; for (let i = 1; i <= 100000; i++) s += i % 7; print(s);\n'],
  ['sqlite', 'web/assets/sqlite_repl.svmb',
    'WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt LIMIT 50000) SELECT sum(x%7) FROM cnt;\n'],
  ['tcl', 'web/assets/tcl_repl.svmb',
    'set s 0\nfor {set i 1} {$i <= 20000} {incr i} { set s [expr {$s + $i % 7}] }\nputs $s\n'],
];

console.log(`cdylib: ${wasmPath}`);
for (const [name, rel, workload] of cards) {
  const path = join(ROOT, rel);
  if (!existsSync(path)) { console.log(`${name}: SKIP (missing ${rel})`); continue; }
  const modBytes = new Uint8Array(readFileSync(path));
  const stdin = enc.encode(workload);

  process.stderr.write(`${name}: interp...\n`); const b = [];
  for (let i = 0; i < N; i++) b.push(bytecodeRun(modBytes, stdin));
  const bBest = Math.min(...b.map((r) => r.ms));

  process.stderr.write(`${name}: pump...\n`);
  let cold, warmBest = Infinity, warm;
  try {
    cold = await pumpRun(modBytes, stdin, `bench#${name}`);
    for (let i = 0; i < N; i++) {
      const r = await pumpRun(modBytes, stdin, `bench#${name}`);
      if (r.ms < warmBest) { warmBest = r.ms; warm = r; }
    }
  } catch (e) {
    console.log(`${name}: bytecode best ${bBest.toFixed(1)}ms | pump REFUSED/failed (${e.message})`);
    continue;
  }
  const parity = warm.out === b[0].out && warm.st === b[0].st ? 'parity=OK' : `parity=MISMATCH (interp ${JSON.stringify(b[0].out)} vs pump ${JSON.stringify(warm.out)})`;
  console.log(
    `${name}: bytecode best ${bBest.toFixed(1)}ms | auto cold ${cold.ms.toFixed(1)}ms, warm best ${warmBest.toFixed(1)}ms ` +
    `(${(bBest / warmBest).toFixed(2)}x) | path ${warm.path} | tierups ${warm.tierups} invokes ${warm.invokes} bounces ${warm.bounces} | ${parity}`,
  );
  // The #839 column proper: the coop fallback tier forced (what a guest gets when the
  // whole-program emit refuses), same workload.
  forcePump = true;
  try {
    const pcold = await pumpRun(modBytes, stdin, `bench#${name}#coop`);
    let pBest = Infinity, pw;
    for (let i = 0; i < N; i++) {
      const r = await pumpRun(modBytes, stdin, `bench#${name}#coop`);
      if (r.ms < pBest) { pBest = r.ms; pw = r; }
    }
    const pparity = pw.out === b[0].out && pw.st === b[0].st ? 'parity=OK' : 'parity=MISMATCH';
    console.log(
      `${name}: coop-forced cold ${pcold.ms.toFixed(1)}ms, warm best ${pBest.toFixed(1)}ms ` +
      `(${(bBest / pBest).toFixed(2)}x) | path ${pw.path} | tierups ${pw.tierups} invokes ${pw.invokes} bounces ${pw.bounces} | ${pparity}`,
    );
  } catch (e) {
    console.log(`${name}: coop-forced REFUSED/failed (${e.message})`);
  }
  forcePump = false;
}
