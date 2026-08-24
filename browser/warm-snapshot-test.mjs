// V8 (Node) end-to-end check of the **warm-runtime snapshot** across the wasm boundary, driving the
// shipping engine FFI the playground would use (`temen_warm_open` / `temen_warm_eval` / `temen_warm_close`).
// The native `crates/temen-llvm/examples/qjs_snapshot.rs` prototype proves the Rust logic; this proves the
// same warm-once / restore-per-Run round-trip works through the cdylib, over an owned window, and — the
// point of the feature — that a restored `eval_run` matches the cold `_start` path byte-for-byte
// (fresh-per-Run isolation) while skipping the QuickJS runtime rebuild.
//
//   node warm-snapshot-test.mjs [temen_browser.wasm] [qjs_snapshot.temen]
//
// Exits 0 on success (all parity checks pass), 1 on any mismatch.
import { readFileSync, existsSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { engineImports } from './engine-imports.mjs';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/temen_browser.wasm';
const modPath = process.argv[3] ?? 'web/assets/qjs_snapshot.temen';

// Skip cleanly if the (fetch-gated) two-phase QuickJS asset isn't present — same posture as the other
// on-ramp browser tests (see ci.yml: "skips cleanly if absent").
if (!existsSync(modPath)) {
  console.log(`SKIP: ${modPath} not built (QuickJS on-ramp asset absent)`);
  process.exit(0);
}

const mod = await WebAssembly.compile(readFileSync(wasmPath));
// The real-browser CI job builds the THREADS module (imports its linear memory as a shared memory, like
// web/par.js / pg_snapshot_test.mjs); a plain build owns its own and ignores the import. Pass one
// unconditionally so both work, and read through whichever memory the instance actually uses
// (`ex.memory` for the plain build's own export; the imported `memory` for the threads build).
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const ex = (await WebAssembly.instantiate(mod, engineImports(memory))).exports;
const membuf = () => (ex.memory ?? memory).buffer;
const is64 = ex.temen_abi_is64() === 1;
const N = (x) => (is64 ? BigInt(x) : Number(x));

// Alloc `bytes.length` in linear memory and copy in; re-fetch the view (alloc may grow/detach memory).
const put = (bytes) => {
  const p = ex.temen_alloc(N(bytes.length));
  new Uint8Array(membuf()).set(bytes, Number(p));
  return { p, len: N(bytes.length), free: () => ex.temen_dealloc(p, N(bytes.length)) };
};
const readStdout = () => {
  const p = Number(ex.temen_stdout_ptr());
  const l = Number(ex.temen_stdout_len());
  return p && l ? Buffer.from(new Uint8Array(membuf(), p, l)).toString() : '';
};
const fail = (m) => {
  console.error(`FAIL: ${m}`);
  process.exit(1);
};

const modBytes = readFileSync(modPath);

// COLD: the original one-shot path — `temen_run_onramp` runs `_start` (read → init → eval → print).
function cold(js) {
  const m = put(modBytes);
  const s = put(Buffer.from(js));
  const t = performance.now();
  Number(ex.temen_run_onramp(m.p, m.len, s.p, s.len));
  const ms = performance.now() - t;
  const out = readStdout();
  const status = ex.temen_status();
  s.free();
  m.free();
  return { out, ms, status };
}

// WARM: restore the post-warmup snapshot, then `eval_run` only.
function warmEval(js) {
  const s = put(Buffer.from(js));
  const t = performance.now();
  Number(ex.temen_warm_eval(s.p, s.len));
  const ms = performance.now() - t;
  const out = readStdout();
  const status = ex.temen_status();
  s.free();
  return { out, ms, status };
}

const programs = [
  ['trivial', '1;\n'],
  [
    "user's program",
    'function fib(n){return n<2?n:fib(n-1)+fib(n-2);}\n' +
      'console.log("fib", Array.from({length:11},(_,i)=>fib(i)).join(" "));\n' +
      'const xs=[5,3,8,1,9,2,7];\n' +
      'console.log("sorted", xs.slice().sort((a,b)=>a-b).join(","));\n' +
      'console.log("json", JSON.stringify({ok:true,nums:[1,2,3],nested:{pi:Math.PI}}));\n' +
      '"0.1 + 0.2 = " + (0.1 + 0.2);\n',
  ],
  ['100k loop', 'let s=0;for(let i=0;i<100000;i++)s+=i;s;\n'],
];

// Open the warm session once (runs `warmup`, snapshots the post-init image).
const t0 = performance.now();
const m = put(modBytes);
const live = Number(ex.temen_warm_open(m.p, m.len));
const warmupMs = performance.now() - t0;
m.free();
if (live < 0 || ex.temen_status() !== 0) fail(`temen_warm_open: status ${ex.temen_status()}`);
console.error(`warm session opened: warmup ${warmupMs.toFixed(0)} ms, live image ${(live / (1 << 20)).toFixed(1)} MiB`);

console.log(`\n${'program'.padEnd(16)}${'cold ms'.padStart(10)}${'warm ms'.padStart(10)}${'speedup'.padStart(9)}  parity`);
let allOk = true;
for (const [name, js] of programs) {
  const c = cold(js);
  if (c.status !== 0) fail(`cold ${name}: status ${c.status}`);
  const w = warmEval(js);
  if (w.status !== 0) fail(`warm ${name}: status ${w.status}`);
  const parity = c.out === w.out;
  allOk &&= parity;
  console.log(
    `${name.padEnd(16)}${c.ms.toFixed(0).padStart(10)}${w.ms.toFixed(0).padStart(10)}${(c.ms / w.ms).toFixed(1).padStart(8)}x  ${parity ? 'OK' : 'MISMATCH!'}`,
  );
  if (!parity) {
    console.log(`  cold: ${JSON.stringify(c.out)}`);
    console.log(`  warm: ${JSON.stringify(w.out)}`);
  }
}

ex.temen_warm_close();

if (!allOk) fail('cold≡warm parity mismatch');
console.log('\nOK: warm-runtime snapshot — eval_run over the restored snapshot matches cold _start byte-for-byte');
