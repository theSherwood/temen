// Dynamic touched-function count for the tier-up break-even question. Runs a real guest (QuickJS)
// on the BYTECODE interpreter (via svm_run_onramp) with the opt-in per-function call profiler armed,
// then reads the histogram: how many of the guest's functions are actually *called* (touched) vs the
// whole-module emit's total, and the call-frequency distribution (hot vs cold separation). Needs a
// cdylib built with `--features callprof`. Light workloads (the interp is slow) — the touched SET and
// distribution shape don't need heavy iteration. Run from browser/:
//   node callprof.mjs [path/to/svm_browser.wasm]  (default: the wasm32 build path)
import { readFileSync } from 'node:fs';
import { engineImports } from './engine-imports.mjs';

const wasmPath = process.argv[2] || 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const eng = (await WebAssembly.instantiate(await WebAssembly.compile(readFileSync(wasmPath)), engineImports())).exports;
if (!eng.svm_callprof_reset) { console.error('cdylib built without --features callprof — rebuild with it'); process.exit(2); }
const memory = eng.memory;
const u8 = () => new Uint8Array(memory.buffer);
const alloc = (bytes) => { const p = Number(eng.svm_alloc(bytes.length)); u8().set(bytes, p); return p; };
const dec = (p, n) => new TextDecoder().decode(u8().slice(p, p + n));

const N = 4096; // histogram size (covers qjs's ~1185 funcs; unused slots stay 0)

function run(name, guestBytes, stdinBytes, totalFuncs) {
  eng.svm_callprof_reset(N);
  const mp = alloc(guestBytes);
  const sp = stdinBytes.length ? alloc(stdinBytes) : 0;
  eng.svm_run_onramp(mp, guestBytes.length, sp, stdinBytes.length);
  const out = dec(Number(eng.svm_stdout_ptr()), eng.svm_stdout_len());
  eng.svm_dealloc(mp, guestBytes.length); if (sp) eng.svm_dealloc(sp, stdinBytes.length);
  eng.svm_callprof_dump();
  const ptr = Number(eng.svm_callprof_ptr()), len = eng.svm_callprof_len();
  const dv = new DataView(memory.buffer, ptr, len);
  const counts = [];
  for (let i = 0; i < len / 8; i++) counts.push(dv.getBigUint64(i * 8, true));
  report(name, counts, totalFuncs, out);
}

function report(name, counts, totalFuncs, out) {
  const touched = counts.filter((c) => c > 0n).length;
  const totalCalls = counts.reduce((a, c) => a + c, 0n);
  // Call-frequency histogram (hot/cold separation).
  const bucket = (c) => (c === 0n ? -1 : c < 10n ? 0 : c < 100n ? 1 : c < 1000n ? 2 : c < 10000n ? 3 : 4);
  const labels = ['1–9', '10–99', '100–999', '1k–9999', '10k+'];
  const hist = [0, 0, 0, 0, 0];
  counts.forEach((c) => { const b = bucket(c); if (b >= 0) hist[b]++; });
  const ranked = counts.map((c, i) => [i, c]).filter(([, c]) => c > 0n).sort((a, b) => (b[1] > a[1] ? 1 : -1));

  console.log(`\n=== ${name} ===`);
  console.log(`  stdout: ${JSON.stringify(out.slice(0, 55))}`);
  console.log(`  TOUCHED ${touched} of ${totalFuncs} functions (${(100 * touched / totalFuncs).toFixed(0)}%) · ${totalCalls.toLocaleString()} total calls`);
  console.log(`  call-count histogram:  ${labels.map((l, i) => `${l}:${hist[i]}`).join('   ')}`);
  console.log(`  hottest functions (call count):  ${ranked.slice(0, 6).map(([i, c]) => `f${i}=${c.toLocaleString()}`).join('  ')}`);
  console.log(`  → whole-module emit produces ${totalFuncs}; a lazy/gated trigger emits at most the ${touched} touched, and`);
  console.log(`    only the hot tail (top buckets) if gated — the rest never pay emit.`);
}

const qjs = readFileSync(new URL('./web/assets/qjs_repl.svmb', import.meta.url));
const QJS_FUNCS = 1185; // from tierup_analyze
// Light workloads (interp is slow): a compute script and an output/allocation script.
const COMPUTE = `let s=0; for(let i=0;i<3000;i++){ s=(s*1103515245+12345+i)>>>0; } function fib(n){return n<2?n:fib(n-1)+fib(n-2);} console.log('r', s>>>0, fib(18));`;
const OUTPUT = `for(let i=0;i<200;i++){ console.log('row', i, (i*2654435761)>>>0); }`;
run('qjs · COMPUTE (loop + fib)', qjs, new TextEncoder().encode(COMPUTE), QJS_FUNCS);
run('qjs · OUTPUT (200 console.log + string/array)', qjs, new TextEncoder().encode(OUTPUT), QJS_FUNCS);
