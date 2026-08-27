// #816 (ledger follow-up): bench the **warm-coop** tier vs the **warm-interp** fallback on the
// page-managing warm-snapshot cards (QuickJS, Tcl, …). The #816 machinery lets a page-managing warm
// guest tier up its eligible pure leaves onto emitted wasm (`runWarmCoop`, item 4), but no playground
// card routes to it — warm cards default to `temen_warm_eval` (the interpreter warm path). This
// measures whether flipping that default pays: per card it opens ONE warm session
// (`temen_warm_open`, init paid once) and then times repeated `eval_run`s of the same workload
// through:
//   1. warm-interp:  temen_warm_eval                       (today's default for a warm card)
//   2. warm-coop:    runWarmCoop → driveCoopTierupRun      (#816 item 4 — leaves on emitted wasm)
//   3. warm+JIT:     runWarmJit                            (WasmDriven whole-program — expected to
//                                                           DECLINE a page-managing guest; reported)
// asserting stdout/status parity against warm-interp, and reporting wall-clock (cold = first Run incl.
// the leaves' emit + WebAssembly.compile; warm = best of N, module cached — the #753/#803 split) plus
// TIERUP / JIT_INVOKE / bounce counts via a Proxy over the cdylib exports (zero engine changes). The
// speedup column is warm-interp/warm-coop: >1 means the coop tier is worth routing the card to.
// Runs on the threads cdylib on Node's V8 — no Playwright.
//
// Build the threads cdylib first (see browser-test.mjs header for the exact RUSTFLAGS), then:
//   node bench_warm_coop.mjs [module.wasm]
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { runWarmCoop, runWarmJit } from './web/wasmjit-module.js';
import { engineImports } from './engine-imports.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2] ?? join(ROOT, 'target/wasm32-unknown-unknown/release/temen_browser.wasm');
const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));

const enc = new TextEncoder();
const dec = new TextDecoder();
const u8 = () => new Uint8Array(memory.buffer);
const readStdout = () => dec.decode(u8().slice(
  Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + Number(ex.temen_stdout_len())));

// Event counters observed from outside (same Proxy trick as bench_tierup_cards.mjs): one TIERUP
// service reads temen_coop_func once; a JIT_INVOKE reads temen_coop_jit_wasm_ptr once; every
// cross-tier bounce goes through temen_coop_call_interp.
const counts = { tierups: 0, invokes: 0, bounces: 0 };
const resetCounts = () => { counts.tierups = 0; counts.invokes = 0; counts.bounces = 0; };
const exCounted = Object.fromEntries(Object.entries(Object.getOwnPropertyDescriptors(ex)).map(([k, d]) => {
  const v = d.value;
  if (typeof v !== 'function') return [k, v];
  if (k === 'temen_coop_func') return [k, (...a) => { counts.tierups++; return v(...a); }];
  if (k === 'temen_coop_jit_wasm_ptr') return [k, (...a) => { counts.invokes++; return v(...a); }];
  if (k === 'temen_coop_call_interp') return [k, (...a) => { counts.bounces++; return v(...a); }];
  return [k, v];
}));

// Open the warm session for `modBytes` (runs the guest's `warmup` once, snapshots the post-init
// image). Returns the live image length; throws on failure (not a warm-snapshot driver, or trap).
function warmOpen(modBytes) {
  const p = Number(ex.temen_alloc(modBytes.length));
  u8().set(modBytes, p);
  const live = Number(ex.temen_warm_open(p, modBytes.length));
  ex.temen_dealloc(p, modBytes.length);
  if (live < 0 || ex.temen_status() !== 0) {
    throw new Error(`warm_open failed: live ${live}, status ${ex.temen_status()}`);
  }
  return live;
}

// One warm-interp eval: restore the snapshot + eval the stdin source on the bytecode interpreter.
function warmInterpEval(stdinBytes) {
  let sp = 0;
  const len = stdinBytes ? stdinBytes.length : 0;
  if (len) { sp = Number(ex.temen_alloc(len)); u8().set(stdinBytes, sp); }
  const t0 = performance.now();
  const rv = Number(ex.temen_warm_eval(sp, len));
  const ms = performance.now() - t0;
  const out = readStdout();
  const st = ex.temen_status();
  if (sp) ex.temen_dealloc(sp, len);
  return { ms, out, st, rv };
}

// One warm-coop eval: restore the snapshot + drive the cooperative tier-up loop (eligible pure
// leaves on emitted wasm). Counts tier-ups/invokes/bounces for this eval.
async function warmCoopEval(stdinBytes, cacheKey) {
  resetCounts();
  const t0 = performance.now();
  const st = await runWarmCoop(exCounted, memory, stdinBytes, cacheKey);
  const ms = performance.now() - t0;
  return { ms, out: readStdout(), st, ...counts };
}

// Best-of-N warm timing after a cold Run, for an async eval fn (returns {ms,...}). Returns
// { cold, warm } where warm is the fastest of N repeats (module/instance cached across Runs).
async function timeAsync(evalFn, n) {
  const cold = await evalFn();
  let warm = null;
  for (let i = 0; i < n; i++) {
    const r = await evalFn();
    if (!warm || r.ms < warm.ms) warm = r;
  }
  return { cold, warm };
}
function timeSync(evalFn, n) {
  const cold = evalFn();
  let warm = null;
  for (let i = 0; i < n; i++) {
    const r = evalFn();
    if (!warm || r.ms < warm.ms) warm = r;
  }
  return { cold, warm };
}

const N = 4;
// Page-managing warm-snapshot cards. Each: [name, asset, workload]. The workload is compute-heavy so
// the guest's hot code has leaf-shaped work the coop tier can service (a pure interpreter dispatch
// loop with no eligible leaves is itself a finding: warm-coop can't help it).
const cards = [
  // Pure compute (little allocation): the eval_run stays whole-program-emittable, so warm+JIT opens.
  ['quickjs-compute', 'web/assets/qjs_snapshot.temen',
    'let s = 0; for (let i = 1; i <= 200000; i++) s = (s + i * 2654435761) % 1000003; print(s);\n'],
  // Allocation-heavy (grows the guest heap DURING eval): probes whether an in-eval grow makes the
  // WasmDriven warm+JIT open decline — the case where warm-coop is the only accelerated tier.
  ['quickjs-alloc', 'web/assets/qjs_snapshot.temen',
    'let a = []; for (let i = 0; i < 60000; i++) a.push((i * 2654435761) % 1000003); ' +
    'let s = 0; for (let i = 0; i < a.length; i++) s = (s + a[i]) % 1000003; print(s);\n'],
  ['tcl', 'web/assets/tcl_snapshot.temen',
    'set s 0\nfor {set i 1} {$i <= 40000} {incr i} { set s [expr {($s + $i) % 1000003}] }\nputs $s\n'],
];

console.log(`cdylib: ${wasmPath}`);
console.log(`(warm session opened once per card; per-eval best of ${N}, speedup = interp/coop)\n`);

for (const [name, rel, workload] of cards) {
  const path = join(ROOT, rel);
  if (!existsSync(path)) { console.log(`${name}: SKIP (missing ${rel})`); continue; }
  const modBytes = new Uint8Array(readFileSync(path));
  const stdin = enc.encode(workload);

  let live;
  try {
    live = warmOpen(modBytes);
  } catch (e) {
    console.log(`${name}: warm_open FAILED (${e.message}) — not a warm-snapshot driver?`);
    continue;
  }
  process.stderr.write(`${name}: warm session open (image ${(live / (1 << 20)).toFixed(1)} MiB)\n`);

  // 1. warm-interp (today's default).
  process.stderr.write(`${name}: warm-interp...\n`);
  const interp = timeSync(() => warmInterpEval(stdin), N);
  const base = interp.warm.out;

  // 2. warm-coop (#816 item 4).
  process.stderr.write(`${name}: warm-coop...\n`);
  let coop = null, coopErr = null;
  try {
    coop = await timeAsync(() => warmCoopEval(stdin, `warmcoop#${name}`), N);
  } catch (e) { coopErr = e; }

  // 3. warm+JIT (expected to decline a page-managing guest — reported either way).
  process.stderr.write(`${name}: warm+JIT probe...\n`);
  let jit = null, jitErr = null;
  try {
    jit = await timeAsync(async () => {
      const t0 = performance.now();
      const st = await runWarmJit(ex, memory, stdin, `warmjit#${name}`);
      return { ms: performance.now() - t0, out: readStdout(), st };
    }, N);
  } catch (e) { jitErr = e; }

  // Report.
  console.log(`── ${name} ─────────────────────────────────────────────`);
  console.log(`  warm-interp : cold ${interp.cold.ms.toFixed(1)}ms  warm ${interp.warm.ms.toFixed(2)}ms  (baseline)  rv=${interp.warm.rv} st=${interp.warm.st}`);
  if (coop) {
    const parity = coop.warm.out === base && coop.warm.st === interp.warm.st ? 'parity=OK' : 'parity=MISMATCH';
    const speedup = interp.warm.ms / coop.warm.ms;
    console.log(`  warm-coop   : cold ${coop.cold.ms.toFixed(1)}ms  warm ${coop.warm.ms.toFixed(2)}ms  (${speedup.toFixed(2)}x)  tierups=${coop.warm.tierups} invokes=${coop.warm.invokes} bounces=${coop.warm.bounces}  ${parity}`);
  } else {
    console.log(`  warm-coop   : DECLINED/failed (${coopErr.message})`);
  }
  if (jit) {
    const parity = jit.warm.out === base && jit.warm.st === interp.warm.st ? 'parity=OK' : 'parity=MISMATCH';
    const speedup = interp.warm.ms / jit.warm.ms;
    console.log(`  warm+JIT    : cold ${jit.cold.ms.toFixed(1)}ms  warm ${jit.warm.ms.toFixed(2)}ms  (${speedup.toFixed(2)}x)  ${parity}`);
  } else {
    console.log(`  warm+JIT    : DECLINED (${jitErr.message.split('\n')[0]})  ← page-managing ⇒ warm-coop is the relevant tier`);
  }
  console.log('');
  ex.temen_warm_close();
}
