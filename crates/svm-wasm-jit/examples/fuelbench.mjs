// V8 (Node) timing for the wasm-JIT fuel A/B. Loads each kernel's three emitted variants
// (mem / global / nofuel), warms V8 to TurboFan, then times f0(win, env, n) with large/small-n
// subtraction (min over reps) so the reported number is per-loop-iteration ns on real V8 — the
// actual target engine, not the wasmi interpreter the differential uses.
//
//   node fuelbench.mjs <dir>
//
// Reads <dir>/manifest.json + <dir>/<kernel>.<variant>.wasm. Prints a CSV: engine,kernel,ns_per_iter.
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';

const dir = process.argv[2];
if (!dir) { console.error('usage: node fuelbench.mjs <dir>'); process.exit(2); }
const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`, 'utf8'));

const ENV_PTR = 1024;          // i64 fuel counter (mem variant) lives here
const WIN_BASE = 0x1_0000;     // guest window base (page 1)
const FUEL = 1n << 61n;
const REPS = 12;
const WARMUP = 30;             // enough calls at LARGE to trip V8 tier-up to TurboFan

function instantiate(bytes) {
  const memory = new WebAssembly.Memory({ initial: 4 });
  let lastTrap = 0;
  const mod = new WebAssembly.Module(bytes);
  const inst = new WebAssembly.Instance(mod, {
    env: {
      memory,
      trap: (code) => { lastTrap = code; },
      call_interp: () => { throw new Error('no cross-tier call in a pure kernel'); },
    },
  });
  return { inst, memory, getTrap: () => lastTrap };
}

// Time one f0(win, env, n) call, resetting fuel first (outside the timed region).
function timeCall(inst, memory, variant, n) {
  const dv = new DataView(memory.buffer);
  if (variant === 'global') inst.exports.fuel.value = FUEL;
  else dv.setBigInt64(ENV_PTR, FUEL, true); // mem + nofuel (harmless for nofuel)
  const t = performance.now();
  inst.exports.f0(WIN_BASE, ENV_PTR, BigInt(n));
  return performance.now() - t;
}

function perIterNs(inst, memory, variant, small, large) {
  for (let i = 0; i < WARMUP; i++) timeCall(inst, memory, variant, large);
  let best = Infinity;
  for (let r = 0; r < REPS; r++) {
    const tl = timeCall(inst, memory, variant, large);
    const ts = timeCall(inst, memory, variant, small);
    const per = ((tl - ts) * 1e6) / (large - small); // ms→ns
    if (per < best) best = per;
  }
  return best;
}

console.log('engine,kernel,ns_per_iter');
for (const k of manifest) {
  for (const variant of ['mem', 'global', 'nofuel']) {
    const bytes = readFileSync(`${dir}/${k.name}.${variant}.wasm`);
    const { inst, memory, getTrap } = instantiate(bytes);
    // Correctness anchor: the emitted result must match the native oracle for this kernel.
    if (variant !== 'nofuel') inst.exports.fuel && (inst.exports.fuel.value = FUEL);
    const dv = new DataView(memory.buffer);
    if (variant === 'global') inst.exports.fuel.value = FUEL;
    else dv.setBigInt64(ENV_PTR, FUEL, true);
    const got = inst.exports.f0(WIN_BASE, ENV_PTR, BigInt(k.large));
    if (getTrap() !== 0) { console.error(`${k.name}.${variant}: trapped ${getTrap()}`); process.exit(1); }
    if (BigInt.asIntN(64, got) !== BigInt(k.result)) {
      console.error(`${k.name}.${variant}: MISCOMPILE got ${BigInt.asIntN(64, got)} want ${k.result}`);
      process.exit(1);
    }
    const ns = perIterNs(inst, memory, variant, k.small, k.large);
    console.log(`wasmjit-${variant},${k.name},${ns.toFixed(3)}`);
  }
}
