// **#750 paged-tier tail-cost bench, V8 half.** Times the four memory-hot kernels emitted by
// `cargo run --release -p svm-wasm-jit --example paged_bench_emit` on Node/V8 — the engine the real
// consumer (the browser tier-up) runs on — through both tier-up modules: `plain.wasm` (mask-only)
// and `paged.wasm` (per-access software page check, fully-Rw table). Same kernel, same engine, so
// paged/plain is the **pure page-check tax**; the manifest's native-interp row contextualizes what
// paged mode buys a guest that today runs whole-interpreter. Prints one table; asserts checksum
// parity across interp/plain/paged (zeroed memory everywhere).
//
//   node bench_paged.mjs <dir>       (dir = target/paged_bench from the emit half)
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { join } from 'node:path';

const dir = process.argv[2] ?? 'target/paged_bench';
const manifest = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8'));

const WIN_SIZE = 1 << manifest.win_log2;
const PAGE_LOG2 = manifest.page_log2;
const ENV_PTR = 1024; // env cell (fuel @0 + cross-tier scratch); below the window
const WIN_BASE = 0x1_0000; // window at 64 KiB, wasm-page aligned
const TABLE_BASE = WIN_BASE + WIN_SIZE; // page-state table right after the window
const TABLE_LEN = WIN_SIZE >> PAGE_LOG2;
const FUEL = 2n ** 62n;

const ITERS = 20_000_000; // per timed call: ~20 M accesses
const REPS = 5; // min-of-reps per row

// One instance per variant over a fresh zeroed memory (matching the interp row's zeroed window).
async function instantiate(file, paged) {
  const bytes = readFileSync(join(dir, file));
  const pages = Math.ceil((TABLE_BASE + TABLE_LEN + 65536) / 65536);
  const memory = new WebAssembly.Memory({ initial: pages });
  const { instance } = await WebAssembly.instantiate(bytes, {
    env: {
      memory,
      trap: (code) => { throw new Error(`guest trap ${code}`); },
      call_interp: () => { throw new Error('no cross-tier call expected in these leaves'); },
    },
  });
  const ex = instance.exports;
  // #717/#750 driver contract: the live bound, and (paged) the table base + a fully-Rw table.
  if (paged) {
    new Uint8Array(memory.buffer, TABLE_BASE, TABLE_LEN).fill(1); // 1 = Rw everywhere
    ex.mapped.value = BigInt(WIN_SIZE); // = the table's byte coverage
    ex.pagestate.value = TABLE_BASE;
  } else {
    ex.mapped.value = BigInt(WIN_SIZE);
  }
  const refuel = () => new DataView(memory.buffer).setBigInt64(ENV_PTR, FUEL, true);
  return { ex, refuel };
}

const plain = await instantiate('plain.wasm', false);
const paged = await instantiate('paged.wasm', true);

function time({ ex, refuel }, func, iters, mask) {
  const f = ex[`f${func}`];
  refuel();
  const warm = f(WIN_BASE, ENV_PTR, 1000n, BigInt(mask)); // V8 lazy-compiles on first call — keep it out of the timing
  let best = Infinity;
  let sum = warm;
  for (let r = 0; r < REPS; r++) {
    refuel();
    const t0 = performance.now();
    sum = f(WIN_BASE, ENV_PTR, BigInt(iters), BigInt(mask));
    best = Math.min(best, ((performance.now() - t0) * 1e6) / iters);
  }
  return { ns: best, sum };
}

console.log(`#750 paged-tier tax — V8 ${process.version}, ${ITERS.toLocaleString()} accesses/call, min of ${REPS}`);
console.log(`${'kernel'.padEnd(12)}${'ws'.padStart(7)}${'interp'.padStart(10)}${'plain'.padStart(9)}${'paged'.padStart(9)}${'tax'.padStart(8)}${'vs interp'.padStart(11)}`);
let ok = true;
// Manifest rows run loads before stores, so the load-kernel checksums (window still zeroed) hold;
// store-kernel checksums are sums of indices, independent of window contents.
for (const k of manifest.kernels) {
  const p = time(plain, k.func, ITERS, k.mask);
  const g = time(paged, k.func, ITERS, k.mask);
  // Checksum parity at the interp row's iteration count: the kernels are pure functions of
  // (n, mask, initial-zero window contents for loads) — re-run at the interp n and compare.
  const pc = time(plain, k.func, k.interp_iters, k.mask).sum;
  const gc = time(paged, k.func, k.interp_iters, k.mask).sum;
  const parity = String(pc) === k.interp_checksum && String(gc) === k.interp_checksum;
  ok = ok && parity;
  const tax = g.ns / p.ns;
  console.log(
    `${k.name.padEnd(12)}${(k.ws_mib + ' MiB').padStart(7)}${k.interp_ns_per_op.toFixed(1).padStart(8)}ns${p.ns.toFixed(2).padStart(7)}ns${g.ns.toFixed(2).padStart(7)}ns${(tax.toFixed(2) + 'x').padStart(8)}${((k.interp_ns_per_op / g.ns).toFixed(1) + 'x').padStart(10)}${parity ? '' : '  CHECKSUM MISMATCH'}`,
  );
}
console.log(ok ? 'checksums agree across interp/plain/paged' : 'FAIL: checksum divergence');
process.exit(ok ? 0 : 1);
