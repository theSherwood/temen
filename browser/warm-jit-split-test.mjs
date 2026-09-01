// #1120 hot-function outlining — A/B measurement of first-Run tier-up on the **real QuickJS** warm+JIT
// card. Drives the shipping engine FFI + browser driver (`primeWarmJit`/`runWarmJit` from
// wasmjit-module.js) over the committed `qjs_snapshot.temen`, in Node's V8 — the same Liftoff→TurboFan
// dynamic tiering the browser card runs on (the `split_proto` findings that motivated this were also
// Node/V8).
//
//   node warm-jit-split-test.mjs [temen_browser.wasm] [qjs_snapshot.temen] [--runs N]
//
// The hypothesis (from split_proto): QuickJS's hot `JS_CallInternal` emits as one giant wasm function
// that V8 runs on Liftoff (baseline) for the first several evals while TurboFan compiles it in the
// background, so the first Run(s) are slow. Outlining it into K smaller wasm functions lets TurboFan
// finish on Run 1 → full speed immediately, with no steady-state tax.
//
// Method: two INDEPENDENT engine instances (split-off vs split-on; the emit is baked + cached at prime
// time, so the flag can't be flipped within one warm session). Each: warm_open → set_split → primeWarmJit
// (the shipping prime: emit + compile + instantiate + one dry eval), then run one compute-heavy program
// K times over the cached instance, recording each Run's wall-time. The per-Run curve shows when each
// variant reaches steady state. Distinct cacheKeys are MANDATORY — the instance cache in wasmjit-module.js
// is module-global keyed by cacheKey, so a shared key would hand variant B the instance bound to variant
// A's memory.
import { readFileSync, existsSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { engineImports } from './engine-imports.mjs';
import { runWarmJit, primeWarmJit } from './web/wasmjit-module.js';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const wasmPath = args[0] ?? 'target/wasm32-unknown-unknown/release/temen_browser.wasm';
const modPath = args[1] ?? 'web/assets/qjs_snapshot.temen';
const runsArg = process.argv.find((a) => a.startsWith('--runs='));
const RUNS = runsArg ? parseInt(runsArg.slice('--runs='.length), 10) : 8;
// `--variant=off|on` runs a SINGLE variant, so each can go in its own fresh Node process — the only way
// to measure Run-1 tier-up without a run-order confound (whichever variant runs first in a shared process
// pays V8/process cold-start, dwarfing the split effect). Omit to run both A/B in one process (quick look,
// confounded Run-1).
const variantArg = process.argv.find((a) => a.startsWith('--variant='));
const ONLY = variantArg ? variantArg.slice('--variant='.length) : null;

if (!existsSync(modPath)) { console.log(`SKIP: ${modPath} not built`); process.exit(0); }
if (!existsSync(wasmPath)) { console.log(`SKIP: ${wasmPath} not built`); process.exit(0); }

const engineWasm = await WebAssembly.compile(readFileSync(wasmPath));
const modBytes = readFileSync(modPath);
const enc = (s) => Buffer.from(s);

// A compute-heavy program that keeps `JS_CallInternal` hot across Runs: deep recursion (fib) is nearly
// all interpreter dispatch + call frames, so the hot function accumulates the V8 call/loop budget that
// drives tier-up. Sized so each Run is tens–hundreds of ms (tier-up transitions are visible, total is
// bounded).
// The eval must keep the guest's hot function hot across Runs so it accumulates V8's tier-up budget.
// Lua and QuickJS take different source syntaxes — pick by the asset name.
const isLua = /lua/i.test(modPath);
const PROGRAM = isLua
  ? 'local function fib(n) if n<2 then return n end return fib(n-1)+fib(n-2) end\n' +
    'print("fib28", fib(28))\n'
  : 'function fib(n){return n<2?n:fib(n-1)+fib(n-2);}\n' +
    'console.log("fib28", fib(28));\n';

async function measureVariant(split, cacheKey) {
  const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
  const ex = (await WebAssembly.instantiate(engineWasm, engineImports(memory))).exports;
  const membuf = () => (ex.memory ?? memory).buffer;
  const is64 = ex.temen_abi_is64() === 1;
  const N = (x) => (is64 ? BigInt(x) : Number(x));
  const readStdout = () => {
    const p = Number(ex.temen_stdout_ptr());
    const l = Number(ex.temen_stdout_len());
    return p && l ? Buffer.from(new Uint8Array(membuf(), p, l)).toString() : '';
  };

  // Open the warm session (runs QuickJS init once, snapshots the post-init image).
  {
    const p = ex.temen_alloc(N(modBytes.length));
    new Uint8Array(membuf()).set(modBytes, Number(p));
    const live = Number(ex.temen_warm_open(p, N(modBytes.length)));
    ex.temen_dealloc(p, N(modBytes.length));
    if (live < 0 || ex.temen_status() !== 0) throw new Error(`warm_open status ${ex.temen_status()}`);
  }

  // THE TOGGLE — before the first open/emit (primeWarmJit performs it), exactly as the snapshot worker does.
  if (!ex.temen_warm_jit_set_split) throw new Error('engine lacks temen_warm_jit_set_split (pre-#1131 build)');
  ex.temen_warm_jit_set_split(split ? 1 : 0);

  const tPrime = performance.now();
  const primed = await primeWarmJit(ex, ex.memory ?? memory, cacheKey, 1);
  const primeMs = performance.now() - tPrime;
  if (!primed) throw new Error('primeWarmJit returned false (eval_run should be emittable)');

  const wasmLen = Number(ex.temen_warm_jit_wasm_len());
  const entryFunc = Number(ex.temen_warm_jit_entry_func());

  const curve = [];
  for (let i = 0; i < RUNS; i++) {
    const t = performance.now();
    const status = await runWarmJit(ex, ex.memory ?? memory, enc(PROGRAM), cacheKey, 1);
    const ms = performance.now() - t;
    if (status !== 0 && status !== 5) throw new Error(`run ${i} status ${status}`);
    curve.push(ms);
  }
  const out = readStdout();
  ex.temen_warm_close();
  return { primeMs, wasmLen, entryFunc, curve, out };
}

console.error(`engine ${wasmPath} (${(readFileSync(wasmPath).length / (1 << 20)).toFixed(1)} MiB)`);
console.error(`asset  ${modPath} (${(modBytes.length / (1 << 20)).toFixed(1)} MiB), ${RUNS} runs/variant\n`);

const steady = (c) => Math.min(...c.slice(Math.max(1, c.length - 3))); // min of the last 3 runs
const fmt = (c) => c.map((x) => x.toFixed(0).padStart(6)).join('');
const reportOne = (label, v) => {
  console.log(`${label}: emit wasm=${v.wasmLen} bytes (${(v.wasmLen / (1 << 20)).toFixed(3)}MiB) entryFunc=${v.entryFunc} prime=${v.primeMs.toFixed(0)}ms`);
  console.log(`  per-Run ms (1…${RUNS}): ${fmt(v.curve)}`);
  console.log(`  Run-1=${v.curve[0].toFixed(0)}ms  steady(min last 3)=${steady(v.curve).toFixed(0)}ms  Run-1/steady=${(v.curve[0] / steady(v.curve)).toFixed(2)}x`);
  console.log(`  out=${JSON.stringify(v.out.trim())}`);
};

if (ONLY === 'off' || ONLY === 'on') {
  const split = ONLY === 'on';
  const v = await measureVariant(split, `${modPath}#eval-${ONLY}`);
  reportOne(`split-${ONLY}`, v);
} else {
  // Both in one process — quick look. Run-1 is run-order-confounded (see --variant); exact wasm_len byte
  // counts still tell us definitively whether the split engaged.
  const off = await measureVariant(false, `${modPath}#eval-off`);
  const on = await measureVariant(true, `${modPath}#eval-on`);
  const engaged = on.wasmLen !== off.wasmLen;
  console.log(`\nsplit engaged: ${engaged ? `YES — emit differs by ${on.wasmLen - off.wasmLen} bytes (eval_run outlined)` : 'NO — emit byte-identical; heuristic declined'}\n`);
  reportOne('split-off', off);
  reportOne('split-on ', on);
  console.log(`\n(NOTE: cross-variant Run-1 comparison in one process is confounded by run order — use --variant=off / --variant=on in separate processes.)`);
  if (off.out !== on.out) { console.error(`FAIL: output mismatch`); process.exit(1); }
}
