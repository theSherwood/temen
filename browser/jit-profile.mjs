// End-to-end wall-clock profiler for a REAL guest on the single-shot module wasm-JIT (BROWSER.md
// Step 0 follow-up). For each guest it breaks the JIT run into phases — EMIT (our Rust emitter:
// decode+outline+IR→wasm, `svm_onramp_jit_run_open*`), V8 COMPILE (`WebAssembly.compile`), INSTANTIATE,
// and RUN (`f0` to completion) — and counts CROSS-TIER BOUNCES (`env.call_interp`, the wasm→interp
// relay) plus the wall-time inside them. This tells us where a real guest's time actually goes, vs the
// synthetic hot-loop kernels the earlier benches used. Also runs the interpreter (`svm_run_onramp`) for
// a total-time baseline + a stdout correctness anchor. Plain (non-shared) cdylib; Node/V8.
//   node jit-profile.mjs [path/to/svm_browser.wasm]
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { engineImports } from './engine-imports.mjs';

const wasmPath = process.argv[2] || 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const eng = (await WebAssembly.instantiate(await WebAssembly.compile(readFileSync(wasmPath)), engineImports())).exports;
const memory = eng.memory;
const u8 = () => new Uint8Array(memory.buffer);
const dec = (p, n) => new TextDecoder().decode(u8().slice(p, p + n));
const readStdout = () => dec(Number(eng.svm_stdout_ptr()), eng.svm_stdout_len());
const alloc = (bytes) => { const p = Number(eng.svm_alloc(bytes.length)); u8().set(bytes, p); return p; };

function interpRun(bytes, stdinBytes) {
  const mp = alloc(bytes);
  let sp = 0; const sl = stdinBytes ? stdinBytes.length : 0;
  if (sl) sp = alloc(stdinBytes);
  const t = performance.now();
  eng.svm_run_onramp(mp, bytes.length, sp, sl);
  const ms = performance.now() - t;
  const out = readStdout();
  eng.svm_dealloc(mp, bytes.length); if (sp) eng.svm_dealloc(sp, sl);
  return { ms, out };
}

// Drive an already-opened JIT run, timing each phase + counting bounces.
async function jitDrive() {
  const wptr = Number(eng.svm_onramp_jit_run_wasm_ptr());
  const wlen = eng.svm_onramp_jit_run_wasm_len();
  const emitted = u8().slice(wptr, wptr + wlen);
  const win = Number(eng.svm_onramp_jit_run_win_ptr());
  const envBytes = eng.svm_onramp_jit_run_env_bytes();
  const slots = [];
  for (let i = 0, n = eng.svm_onramp_jit_run_slot_count(); i < n; i++) slots.push(eng.svm_onramp_jit_run_slot(i));

  const tC = performance.now();
  const module = await WebAssembly.compile(emitted);
  const compileMs = performance.now() - tC;

  let bounces = 0, bounceMs = 0;
  const tI = performance.now();
  const instance = new WebAssembly.Instance(module, {
    env: {
      memory,
      trap: () => {},
      call_interp: (func, argsPtr) => {
        bounces++;
        const b = performance.now();
        const r = eng.svm_onramp_jit_run_call_interp(func, argsPtr);
        bounceMs += performance.now() - b;
        if (r !== 0) throw new Error('cross-tier stop');
      },
    },
  });
  const instMs = performance.now() - tI;

  const env = Number(eng.svm_alloc(envBytes));
  new DataView(memory.buffer).setBigInt64(env, 1n << 60n, true);
  const tR = performance.now();
  try { instance.exports.f0(win, env, ...slots); } catch { /* exit/trap unwinds f0 */ }
  const runMs = performance.now() - tR;
  eng.svm_dealloc(env, envBytes);
  eng.svm_onramp_jit_run_finish();
  eng.svm_onramp_jit_run_close();
  return { emittedBytes: wlen, compileMs, instMs, runMs, bounces, bounceMs, out: readStdout() };
}

// The interpreter baseline double-interprets (guest bytecode on the SVM interp), so it is
// pathologically slow for a compute-heavy guest (minutes for fib+loops on QuickJS). Off by default —
// the JIT phase breakdown is the deliverable; set INTERP=1 for the total-speedup + correctness anchor.
const WANT_INTERP = process.env.INTERP === '1';

async function profileStdin(name, modBytes, stdinBytes) {
  const interp = WANT_INTERP ? interpRun(modBytes, stdinBytes) : null;
  const modP = alloc(modBytes);
  const stdinP = stdinBytes.length ? alloc(stdinBytes) : 0;
  const tE = performance.now();
  const opened = eng.svm_onramp_jit_run_open(modP, modBytes.length, stdinP, stdinBytes.length, 0);
  const emitMs = performance.now() - tE;
  eng.svm_dealloc(modP, modBytes.length); if (stdinP) eng.svm_dealloc(stdinP, stdinBytes.length);
  if (opened !== 0) { console.log(`${name}: _start not emittable (status ${eng.svm_status()}) — interp-only`); return; }
  const jit = await jitDrive();
  report(name, emitMs, jit, interp);
}

// The chibicc "compile a C source" shape, via the memfs opener. NOTE: `open_fs` emits a module that
// imports SHARED memory, so it instantiates only against a **threads** cdylib build (shared memory);
// on the plain build here it LinkErrors on the memory import. Kept for use with that build.
async function profileCompile(name, modBytes, srcBytes) {
  const modP = alloc(modBytes);
  const srcP = alloc(srcBytes);
  const tE = performance.now();
  const opened = eng.svm_onramp_jit_run_open_fs(modP, modBytes.length, 0, 0, srcP, srcBytes.length, 0);
  const emitMs = performance.now() - tE;
  eng.svm_dealloc(modP, modBytes.length); eng.svm_dealloc(srcP, srcBytes.length);
  if (opened !== 0) { console.log(`${name}: _start not emittable (status ${eng.svm_status()})`); return; }
  const jit = await jitDrive();
  report(name, emitMs, jit, null);
}

function report(name, emitMs, jit, interp) {
  const total = emitMs + jit.compileMs + jit.instMs + jit.runMs;
  const pct = (x) => `${(100 * x / total).toFixed(1)}%`;
  console.log(`\n=== ${name} ===`);
  console.log(`emitted module: ${(jit.emittedBytes / 1024).toFixed(0)} KiB   stdout: ${JSON.stringify(jit.out.slice(0, 70))}`);
  if (interp) {
    const match = jit.out === interp.out ? 'IDENTICAL' : 'MISMATCH';
    console.log(`correctness vs interp: ${match}   interp total ${interp.ms.toFixed(1)}ms  ·  jit total ${total.toFixed(1)}ms  (${(interp.ms / total).toFixed(1)}× )`);
  }
  console.log(`phase                     ms        share`);
  console.log(`  EMIT (IR→wasm)          ${emitMs.toFixed(1).padStart(8)}   ${pct(emitMs)}`);
  console.log(`  V8 COMPILE              ${jit.compileMs.toFixed(1).padStart(8)}   ${pct(jit.compileMs)}`);
  console.log(`  INSTANTIATE             ${jit.instMs.toFixed(1).padStart(8)}   ${pct(jit.instMs)}`);
  console.log(`  RUN (f0 to completion)  ${jit.runMs.toFixed(1).padStart(8)}   ${pct(jit.runMs)}`);
  console.log(`    of which cross-tier   ${jit.bounceMs.toFixed(1).padStart(8)}   ${pct(jit.bounceMs)}  (${jit.bounces.toLocaleString()} bounces, ${(1000 * jit.bounceMs / Math.max(jit.bounces, 1)).toFixed(2)} µs each)`);
  console.log(`  TOTAL                   ${total.toFixed(1).padStart(8)}`);
}

// Two contrasting real workloads on QuickJS, to test which lever matters:
//  - COMPUTE-heavy: a hot loop + fib recursion, almost no host I/O (few bounces expected).
//  - OUTPUT-heavy: thousands of console.log calls (each a host write = a cross-tier bounce), to see
//    whether the wasm↔interp boundary ever dominates (the "schedule/IO-shaped guest" hypothesis).
const QJS_COMPUTE = `
let s=0; for(let i=0;i<1000000;i++){ s=(s*1103515245+12345+i)>>>0; }
function fib(n){ return n<2?n:fib(n-1)+fib(n-2); }
console.log('sum', s>>>0, 'fib28', fib(28));
`;
const QJS_OUTPUT = `
let s=0;
for(let i=0;i<20000;i++){ s=(s+i*2654435761)>>>0; console.log('row', i, s>>>0); }
`;

const dirAsset = (n) => readFileSync(new URL(`./web/assets/${n}.svmb`, import.meta.url));
const qjs = dirAsset('qjs_repl');
await profileStdin('qjs · COMPUTE (loop + fib, ~no I/O)', qjs, new TextEncoder().encode(QJS_COMPUTE));
await profileStdin('qjs · OUTPUT (20k console.log = 20k host writes)', qjs, new TextEncoder().encode(QJS_OUTPUT));
