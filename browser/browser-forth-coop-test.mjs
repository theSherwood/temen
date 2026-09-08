// #1233 — **the Forth kernel on the cooperative wasm-JIT tier**, checked through the real JS driver.
//
// The native `coop_tierup_driver.rs` pin (`coop_forth_kernel_tiers_up_and_matches_the_oracle`) plays
// the browser's role with wasmi; this is the other half: the *actual* `driveCoopTierupRun` in
// `web/wasmjit-module.js`, over the *actual* wasm cdylib, on the path the playground's "wasm-JIT"
// toggle takes for the Forth card (`runJitModule` → whole-program open declines → coop).
//
// The shape it pins is the one the coop driver used to assume never happens: `process`, the kernel's
// outer interpreter, tiers up as an emitted frame and, from *there*, `vm_jit_install`s each colon
// definition (a §22 install inside a bounce, serviced by the engine's nested drive since #1233),
// releases the code handle, and `call.dyn`s the word — so the driver must rebuild its table from
// inside the bounce (`syncTableSync`) and fetch the unit **by slot**, not by the released handle.
//
// Asserts, for the native pin's program and for the card's default program minus its thread lines
// (loops, memory, strings, fibers, the sieve): stdout and status match the plain bytecode path
// byte-for-byte, the coop path was taken, the emitted tier drove it (tierups > 0), and words were
// defined from emitted code (bounces > 0). The card's **thread** lines (`spawn`/`join`) pin the
// declared frontier instead: `thread.spawn`/`join` reached from inside a bounced tier-up region
// cannot be serviced (a nested drive owns no scheduler to create the task in, and a join would park
// the emitted frame that is on the wasm stack), so the run declines to the interpreter — the
// playground logs the fallback and runs the interpreter (`play.js`, "wasm-JIT module unavailable").
//
// Usage:  node browser-forth-coop-test.mjs [module.wasm]   (build the threads cdylib first)

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
  if (!existsSync(p)) {
    console.error(`SKIP: ${p} missing`);
    process.exit(0);
  }
}

// The native pin's program: words defined then called from the same emitted `process` frame, with a
// hot word-to-word loop (`sumsq` over `sq`) so the unit-to-unit `call.dyn` edge dispatches natively.
const PIN_PROGRAM = `: sq ( n -- n ) dup * ;
: fact ( n -- n ) dup 1 > if dup 1- recurse * else drop 1 then ;
5 sq . 10 fact . cr
: sumsq ( n -- s ) 0 swap 0 do i sq + loop ;
100 sumsq . cr
`;

// The playground card's default program (`web/play.js`, the Forth card's `src`; the asset gate
// `tests/forth_asset.rs` pins the same text on the bytecode engine), split at its thread lines.
const CARD_PREFIX = `\\ Forth on Temen: every word below is JIT-compiled to a verified IR unit.
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
const CARD_THREADS = `\\ threads: run a word on another vCPU, join its result; atomics on a shared cell
: work ( x -- y ) 1000 * ;
' work 7 spawn join . cr
variable hits
: bump ( n -- y ) begin dup 0 > while 1 hits atomic+! drop 1- repeat ;
' bump 100 spawn ' bump 100 spawn join swap join + . hits @ . cr
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

// The **threads** cdylib (the build the playground ships): it imports `env.memory`, so supply one
// with a maximum — the emitted wasm's own memory import is shared and needs a bounded one.
const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));
const u8 = () => new Uint8Array(memory.buffer);
const enc = new TextEncoder();
const dec = new TextDecoder();
const readStdout = () => dec.decode(u8().slice(
  Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + Number(ex.temen_stdout_len())));

const guest = readFileSync(asset);

// The counters ride a Proxy over the cdylib exports (the `bench_tierup_cards.mjs` trick, zero engine
// changes): one TIERUP service reads `temen_coop_func` once, every bounce goes through
// `temen_coop_call_interp`, and `temen_coop_open` returning 0 means the coop path was taken.
const counts = { tierups: 0, bounces: 0, path: 'interp' };
const exCounted = Object.fromEntries(
  Object.entries(Object.getOwnPropertyDescriptors(ex)).map(([k, d]) => {
    const v = d.value;
    if (typeof v !== 'function') return [k, v];
    if (k === 'temen_coop_func') return [k, (...a) => { counts.tierups++; return v(...a); }];
    if (k === 'temen_coop_call_interp') return [k, (...a) => { counts.bounces++; return v(...a); }];
    if (k === 'temen_coop_open') {
      return [k, (...a) => {
        const r = v(...a);
        if (r === 0) counts.path = 'coop';
        return r;
      }];
    }
    return [k, v];
  }));

// `process` is small; pin the floor at 0 so the kernel's outer loop is admitted (the native pin runs
// under the same floor), and say so rather than inheriting a default.
ex.temen_coop_set_tierup_floor(0);

const fail = (msg) => { console.error(`FAIL: ${msg}`); process.exitCode = 1; };

const oracle = (stdin) => {
  const mp = Number(ex.temen_alloc(guest.length));
  u8().set(guest, mp);
  const sp = Number(ex.temen_alloc(stdin.length));
  u8().set(stdin, sp);
  ex.temen_run_onramp(mp, guest.length, sp, stdin.length);
  const want = { status: ex.temen_status(), out: readStdout() };
  ex.temen_dealloc(mp, guest.length);
  ex.temen_dealloc(sp, stdin.length);
  return want;
};

for (const [label, program] of [
  ['pin program', PIN_PROGRAM],
  ['card program (thread lines aside)', CARD_PREFIX + CARD_SIEVE],
]) {
  const stdin = enc.encode(program);

  // ---- 1. the oracle: the plain bytecode path ---------------------------------------------------
  const want = oracle(stdin);
  if (want.status !== 0) { fail(`${label}: oracle sanity: status ${want.status}, expected 0`); continue; }

  // ---- 2. the playground's wasm-JIT path ----------------------------------------------------------
  counts.tierups = 0; counts.bounces = 0; counts.path = 'interp';
  let status;
  try {
    status = await runJitModule(exCounted, memory, guest, stdin, `forth-coop-${label}`);
  } catch (e) {
    fail(`${label}: the coop run did not complete: ${e.message}`);
    console.error('       (a trap here is the #1233 symptom: a §22 install from an emitted frame, or a '
      + 'released-after-install unit nulled out of the driver\'s table)');
    continue;
  }
  const got = { status: ex.temen_status(), out: readStdout() };

  // ---- 3. assertions ------------------------------------------------------------------------------
  if (counts.path !== 'coop') fail(`${label}: the coop driver must serve the kernel, took '${counts.path}'`);
  if (counts.tierups < 1) fail(`${label}: the emitted tier must drive: ${counts.tierups} tier-ups`);
  if (counts.bounces < 1) fail(`${label}: words must be defined from emitted code: ${counts.bounces} bounces`);
  if (got.status !== want.status) fail(`${label}: status ${got.status}, oracle ${want.status}`);
  if (got.out !== want.out) fail(`${label}: stdout mismatch\n--- got ---\n${got.out}\n--- oracle ---\n${want.out}`);
  if (status !== 0) fail(`${label}: runJitModule returned status ${status}`);
  if (!process.exitCode) {
    console.log(`ok — ${label}: coop path, ${counts.tierups} tier-up(s), ${counts.bounces} bounce(s), `
      + 'stdout matches the bytecode oracle');
  }
}

// ---- 4. the declared frontier: thread words reached from the emitted outer loop decline ---------
{
  const stdin = enc.encode(CARD_THREADS);
  const want = oracle(stdin);
  if (want.status !== 0) fail(`threads: oracle sanity: status ${want.status}, expected 0`);
  counts.tierups = 0; counts.bounces = 0; counts.path = 'interp';
  let declined = false;
  try {
    await runJitModule(exCounted, memory, guest, stdin, 'forth-coop-threads');
  } catch (e) {
    declined = /trapped \(declined to the interpreter\)/.test(e.message);
    if (!declined) fail(`threads: expected a clean decline, got: ${e.message}`);
  }
  if (counts.path !== 'coop') fail(`threads: the coop driver must open the kernel, took '${counts.path}'`);
  if (!declined) {
    // Servicing `spawn`/`join` inside a bounce landed: move these lines into the parity set above.
    fail('threads: the run completed on the coop tier — the declared frontier moved, update this pin');
  } else if (!process.exitCode) {
    console.log('ok — thread words: declined to the interpreter (the declared frontier: a spawn/join '
      + 'reached from a bounced tier-up region)');
  }
}

if (process.exitCode) process.exit(process.exitCode);
