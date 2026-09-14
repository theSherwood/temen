// #1233 slice 3 — **is the decline actually costing anything?**
//
// Slice 3's whole effect is "stop declining a session to the interpreter because it *mentions* thread
// words". So the number that decides whether to build it is the card's own program, end to end:
//
//   (a) today      — the card's default program uses `spawn`/`join`, so the WHOLE run declines and
//                    every line of it (sieve included) runs on the bytecode interpreter.
//   (b) slice 3    — the non-thread part stays on the emitted cooperative tier.
//
// (b) is measurable today without building anything: the card program *minus* its thread lines already
// runs coop-emitted — that is exactly what `browser-forth-coop-test.mjs` asserts. So timing the same
// text through `temen_run_onramp` (a) and through `runJitModule` (b) brackets what slice 3 recovers.
//
// The engine has to be a real one. The native wasmi pin cannot answer this: wasmi is itself an
// interpreter, so its "emitted" side would be an interpreter running interpreted wasm. Node's V8 is
// the same class of engine the playground runs on.
//
// Sections are timed separately because the card is two different regimes glued together: the
// definition-heavy prelude (compile-dominated — each word compiled once, called a handful of times)
// and the sieve capstone (compute-dominated). #1233's earlier per-call spike measured a 26-43x
// *dispatch* ceiling but explicitly noted the card is compile-dominated; this measures which regime
// the card actually lives in.
//
// Usage:  node forth-tierup-bench.mjs [module.wasm]   (build the threads cdylib first)

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { runJitModule } from './web/wasmjit-module.js';
import { engineImports } from './engine-imports.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2]
  ?? join(ROOT, 'target/wasm32-unknown-unknown/release/temen_browser.wasm');
const asset = join(ROOT, 'web/assets/forth.temen');
for (const p of [wasmPath, asset]) {
  if (!existsSync(p)) { console.error(`SKIP: ${p} missing`); process.exit(0); }
}

// Verbatim from `browser-forth-coop-test.mjs` / the card's `src` in `web/play.js`, split by regime.
const CARD_DEFS = `\\ Forth on Temen: every word below is JIT-compiled to a verified IR unit.
: sq ( n -- n ) dup * ;
: fact ( n -- n ) dup 1 > if dup 1- recurse * else drop 1 then ;
5 sq . 10 fact . cr

\\ loops: begin/until, begin/while/repeat
: countdown ( n -- ) begin dup . 1- dup 0= until drop cr ;
5 countdown
: sum-to ( n -- s ) 0 swap begin dup 0 > while tuck + swap 1- repeat drop ;
100 sum-to . cr

\\ counted loops: do/loop, i is the index; the accumulator stays on the data stack
: sumsq ( n -- s ) 0 swap 0 do i i * + loop ;
5 sumsq . cr

\\ memory: variables, strings, the heap
variable x   42 x !   x @ 1+ x !   x @ . cr
." hello, forth" cr

\\ fibers: a generator word is a task; resume it from any later line
: counter ( x -- y ) begin 1+ dup yield drop again ;
' counter task
dup 0 resume . . cr
dup 10 resume . . cr
drop
`;
const CARD_SIEVE = `\\ capstone: a whole program — the sieve of Eratosthenes counts primes below N
variable arr   variable lim
: primes ( n -- c )
  lim !  here arr !  lim @ allot  0
  lim @ 2 do
    arr @ i + c@ 0= if
      1+  i dup * lim @ < if
        lim @ i dup * do  1 arr @ i + c!  j +loop
      then
    then
  loop ;
10 primes . 100 primes . 1000 primes . cr
`;
// A deliberately hot program: one word, called a lot. The regime #1233's per-call spike measured.
const HOT = `: sq ( n -- n ) dup * ;
: bench ( -- s ) 0 200000 0 do i sq + loop ;
bench . cr
`;

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));
const u8 = () => new Uint8Array(memory.buffer);
const enc = new TextEncoder();
const guest = readFileSync(asset);

ex.temen_coop_set_tierup_floor(0);

// Validity check: the "coop" column is only meaningful if that run genuinely took the cooperative
// path AND the emitted tier actually drove it. A silent fall-back inside `runJitModule` would make
// this a comparison of the interpreter against itself plus overhead. Count through a Proxy over the
// cdylib exports (the `bench_tierup_cards.mjs` trick, zero engine changes), exactly as
// `browser-forth-coop-test.mjs` does.
const counts = { tierups: 0, bounces: 0, path: 'interp' };
const exCounted = Object.fromEntries(
  Object.entries(Object.getOwnPropertyDescriptors(ex)).map(([k, d]) => {
    const v = d.value;
    if (typeof v !== 'function') return [k, v];
    if (k === 'temen_coop_func') return [k, (...a) => { counts.tierups++; return v(...a); }];
    if (k === 'temen_coop_call_interp') return [k, (...a) => { counts.bounces++; return v(...a); }];
    if (k === 'temen_coop_open') {
      return [k, (...a) => { const r = v(...a); if (r === 0) counts.path = 'coop'; return r; }];
    }
    return [k, v];
  }));

// (a) the declined path: the plain bytecode interpreter, what the card does today.
function timeInterp(stdin) {
  const mp = Number(ex.temen_alloc(guest.length));
  u8().set(guest, mp);
  const sp = Number(ex.temen_alloc(stdin.length));
  u8().set(stdin, sp);
  const t0 = performance.now();
  ex.temen_run_onramp(mp, guest.length, sp, stdin.length);
  const dt = performance.now() - t0;
  const status = ex.temen_status();
  ex.temen_dealloc(mp, guest.length);
  ex.temen_dealloc(sp, stdin.length);
  return { dt, status };
}

// (b) the emitted cooperative tier, through the SHIPPED JS driver (`driveCoopTierupRun`).
async function timeCoop(stdin, label) {
  counts.tierups = 0; counts.bounces = 0; counts.path = 'interp';
  const t0 = performance.now();
  const status = await runJitModule(exCounted, memory, guest, stdin, label);
  return {
    dt: performance.now() - t0,
    status,
    tierups: counts.tierups,
    bounces: counts.bounces,
    path: counts.path,
  };
}

// Min of N samples: the floor is the signal, the tail is scheduler/GC noise.
async function best(stdin, label, reps) {
  let interp = Infinity, coop = Infinity, ok = true, tierups = 0, path = 'interp';
  for (let i = 0; i < reps; i++) {
    const a = timeInterp(stdin);
    if (a.status !== 0) { ok = false; break; }
    interp = Math.min(interp, a.dt);
    try {
      const b = await timeCoop(stdin, `${label}-${i}`);
      if (b.status !== 0) { ok = false; break; }
      coop = Math.min(coop, b.dt);
      tierups = b.tierups; path = b.path;
    } catch (e) { ok = false; console.error(`  (coop run failed: ${e.message})`); break; }
  }
  return { interp, coop, ok, tierups, path };
}

console.log('\n#1233 slice 3 — what the decline costs, on the card\'s own program (V8, min of samples):\n');
console.log('  section                          interp (declined)   coop (emitted)   speedup   tier-ups');
console.log('  ' + '-'.repeat(86));

// What the card's DEFAULT program costs today, the whole path: the playground tries the emitted tier
// first, the thread lines make it decline, and only then does it run the interpreter. So today's cost
// is a wasted emit attempt *plus* the full interpreted run — which is what slice 3 would have to beat.
const CARD_THREADS = `\\ threads: run a word on another vCPU, join its result; atomics on a shared cell
: work ( x -- y ) 1000 * ;
' work 7 spawn join . cr
variable hits
: bump ( n -- y ) begin dup 0 > while 1 hits atomic+! drop 1- repeat ;
' bump 100 spawn ' bump 100 spawn join swap join + . hits @ . cr
`;
async function timeDeclineThenInterp(stdin, label) {
  const t0 = performance.now();
  try { await runJitModule(exCounted, memory, guest, stdin, label); }
  catch { /* the declared decline — the playground logs it and runs the interpreter */ }
  const a = timeInterp(stdin);
  return { dt: performance.now() - t0 + 0, interpOnly: a.dt, status: a.status };
}

const rows = [
  ['card: definitions only', CARD_DEFS, 5],
  ['card: sieve capstone', CARD_SIEVE, 5],
  ['card: both (no thread lines)', CARD_DEFS + CARD_SIEVE, 5],
  ['hot loop (200k calls)', HOT, 3],
];
for (const [label, program, reps] of rows) {
  const { interp, coop, ok, tierups, path } = await best(enc.encode(program), label.replace(/\W+/g, '-'), reps);
  if (!ok) { console.log(`  ${label.padEnd(32)} FAILED (see above)`); continue; }
  const speedup = interp / coop;
  const note = path === 'coop' && tierups > 0 ? `${tierups}` : `NONE (${path}!)`;
  console.log(`  ${label.padEnd(32)} ${interp.toFixed(1).padStart(10)} ms  ${coop.toFixed(1).padStart(13)} ms  ${speedup.toFixed(2).padStart(7)}x   ${note.padStart(8)}`);
}
// The default card, end to end, as the playground actually runs it today.
{
  const stdin = enc.encode(CARD_DEFS + CARD_THREADS + CARD_SIEVE);
  let full = Infinity, pure = Infinity;
  for (let i = 0; i < 3; i++) {
    const r = await timeDeclineThenInterp(stdin, `default-card-${i}`);
    if (r.status !== 0) { full = NaN; break; }
    full = Math.min(full, r.dt);
    pure = Math.min(pure, r.interpOnly);
  }
  console.log('  ' + '-'.repeat(86));
  console.log(`  ${'DEFAULT card (with threads)'.padEnd(32)} ${pure.toFixed(1).padStart(10)} ms  ${'—'.padStart(13)}     ${'—'.padStart(6)}   ${'declines'.padStart(8)}`);
  console.log(`    └ today's real cost = wasted emit attempt + interpreted run: ${full.toFixed(1)} ms`);
}

console.log('\n  interp = the whole run on the bytecode interpreter (what a thread-word session gets today)');
console.log('  coop   = the same text on the emitted cooperative tier (what slice 3 would preserve)\n');
