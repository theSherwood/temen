// #1110 emit-split prototype — Node/V8 measurement harness.
//
// Loads the blobs emitted by `examples/split_proto.rs` (via manifest.json) and, for each configuration,
// instantiates every partition against ONE shared memory + ONE shared reserved funcref table, populates
// every table slot with the emitting instance's `f{i}` (the host-owns-the-table Model-B2 contract), then
// calls the entry `f0(win, env, n)` R times, timing each run. The per-run ms series exposes V8's
// Liftoff→TurboFan tier-up (the run where ms drops) and the steady-state cost.
//
// Configs (see the emitter): single (whole-program, ~10 MB), split_good (hot path in a tiny module,
// helper intra-module), split_bad (helper stranded in the cold module), split_xmod (f0 and f1 each in
// their own tiny module — isolates pure cross-module call_indirect cost).
//
// Run: node split_proto.mjs <dir> [n_iters] [runs]

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const dir = process.argv[2] ?? '.';
const N_ITERS = BigInt(process.argv[3] ?? 20_000_000);
const RUNS = Number(process.argv[4] ?? 14);
// Optional 5th arg: run ONLY this config (in a fresh process → no cross-config V8 code-cache sharing).
const ONLY = process.argv[5] || null;

const manifest = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8'));
const TABLE_SIZE = 1 << manifest.table_log2;
const WIN = 0;      // no memory access in the synthetic guest → window base is irrelevant
const ENV = 8;      // env cell ptr; untouched (no cross-tier leaves), fuel lives in a wasm global

// Pre-compile every distinct blob once (WebAssembly.Module is reusable across instantiations).
const moduleCache = new Map();
async function moduleFor(name) {
  if (!moduleCache.has(name)) {
    const bytes = readFileSync(join(dir, name));
    moduleCache.set(name, await WebAssembly.compile(bytes));
  }
  return moduleCache.get(name);
}

// Build a fresh domain (memory + table + instances) for `config` and return the entry function. A fresh
// domain per config keeps V8's per-module compiled-code state independent between configs.
async function instantiate(config) {
  const memory = new WebAssembly.Memory({ initial: 4 });
  const table = new WebAssembly.Table({ initial: TABLE_SIZE, maximum: TABLE_SIZE, element: 'anyfunc' });
  const env = {
    memory,
    trap: (code) => { throw new Error('guest trap ' + code); },
    call_interp: () => { throw new Error('unexpected call_interp'); },
    __indirect_function_table: table,
  };
  let entry = null;
  for (const part of config) {
    const mod = await moduleFor(part.wasm);
    const inst = await WebAssembly.instantiate(mod, { env });
    for (const i of part.funcs) {
      const f = inst.exports['f' + i];
      table.set(i, f);
      if (i === manifest.entry) entry = f;
    }
  }
  if (!entry) throw new Error('no partition emitted the entry');
  return entry;
}

// Correctness here is cross-config agreement: every config runs the *same* IR, so all must return the
// same i64 (the emitter's own interpreter-differential test, `tests/split.rs`, pins absolute correctness).
// The first config measured sets the reference.
let reference = null;

async function measure(name, config) {
  const entry = await instantiate(config);
  const series = [];
  for (let r = 0; r < RUNS; r++) {
    const t0 = performance.now();
    const got = entry(WIN, ENV, N_ITERS);
    const ms = performance.now() - t0;
    if (reference === null) reference = got;
    else if (got !== reference) throw new Error(`${name} run ${r}: ${got} != reference ${reference}`);
    series.push(ms);
  }
  return series;
}

function fmt(series) {
  return series.map((ms) => String(Math.round(ms)).padStart(5)).join(' ');
}

// The run index where the series first drops to within 1.4× of its own minimum — a proxy for "reached
// TurboFan". Steady-state = median of the last third.
function analyze(series) {
  const min = Math.min(...series);
  const tierRun = series.findIndex((ms) => ms <= min * 1.4);
  const tail = series.slice(Math.floor((series.length * 2) / 3)).sort((a, b) => a - b);
  const steady = tail[Math.floor(tail.length / 2)];
  return { tierRun, steady, first: series[0] };
}

console.log(`node ${process.version} — n_iters=${N_ITERS}, runs=${RUNS}, table=${TABLE_SIZE}\n`);
const order = (ONLY ? [ONLY] : ['single', 'split_good', 'split_xmod', 'split_bad']);
const results = {};
for (const name of order) {
  if (!manifest.configs[name]) continue;
  const series = await measure(name, manifest.configs[name]);
  results[name] = { series, ...analyze(series) };
  console.log(`${name.padEnd(11)} | per-run ms: ${fmt(series)}`);
}
console.log('');
console.log('config      | run0(cold) | tier-up@run | steady ms');
console.log('------------|------------|-------------|----------');
for (const name of order) {
  const r = results[name];
  if (!r) continue;
  console.log(
    `${name.padEnd(11)} | ${String(Math.round(r.first)).padStart(10)} | ${String(r.tierRun).padStart(11)} | ${String(Math.round(r.steady)).padStart(8)}`,
  );
}
console.log('');
if (results.single && results.split_good) {
  const s = results.single, g = results.split_good;
  console.log(`Q1 first-Run tier-up: single reaches steady at run ${s.tierRun}, split_good at run ${g.tierRun}`);
}
if (results.split_good && results.split_xmod) {
  const g = results.split_good.steady, x = results.split_xmod.steady;
  const pct = ((x - g) / g) * 100;
  console.log(`Q2 cross-module call cost (steady): intra ${Math.round(g)}ms vs cross ${Math.round(x)}ms  (${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%)`);
}
