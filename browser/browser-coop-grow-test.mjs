// #1312 — **the cooperative tier-up window grows**, checked through the real JS driver.
//
// The native `coop_tierup_driver.rs` differential plays the browser's role with wasmi; this is the
// other half: the *actual* `driveCoopTierupRun` in `web/coop-driver.js`, over the *actual* wasm
// cdylib, on the path the playground takes (`runJitModule` → whole-program open declines → coop).
// It is the only place the `"win"` fan-out after a bounce is exercised, because only here is the
// window the engine's own (relocating) backing rather than a mirrored copy.
//
// The guest (`tests/fixtures/coop_grow_past_window.temen`, from `genfixture … grow_past_window`) is
// the reduced synthesized-allocator shape: a hot leaf bounces to a helper that `vm_map`s a page past
// the 32 MiB run window, then stores and loads through it. Before the growable backing that map
// returned `-EINVAL`, the store faulted, and the run trapped with zero tier-ups.
//
// Asserts: stdout and status match the plain bytecode path byte-for-byte, the coop path was taken,
// the emitted tier actually drove it (tierups > 0), and the window really grew past its opening size.
//
// #1954: then a single-threaded grow guest with its root offered as a leaf (`COOP_LEAF_ROOT`): the
// whole program is program 0's leaf — no region emit — and its bounce to the grow helper moves the
// window under the leaf's own frames, which the driver's `"win"` fan-out must follow.
//
// Usage:  node browser-coop-grow-test.mjs [module.wasm]   (build the threads cdylib first)

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { driveCoopTierupRun, runJitModule } from './web/wasmjit-module.js';
import { engineImports } from './engine-imports.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2]
  ?? join(ROOT, 'target/wasm32-unknown-unknown/release/temen_browser.wasm');
const fixture = join(ROOT, 'tests/fixtures/coop_grow_past_window.temen');
for (const p of [wasmPath, fixture]) {
  if (!existsSync(p)) {
    console.error(`SKIP: ${p} missing`);
    process.exit(0);
  }
}

// The **threads** cdylib (the build the playground ships and the one whose emitted modules import a
// shared memory): it imports `env.memory`, so supply one with a maximum — the emitted wasm's own
// memory import is shared and needs a bounded one.
const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));
const u8 = () => new Uint8Array(memory.buffer);
const dec = new TextDecoder();
const readStdout = () => dec.decode(u8().slice(
  Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + Number(ex.temen_stdout_len())));

const guest = readFileSync(fixture);

// ---- 1. the oracle: the plain bytecode path, which reserves 2^40 and grows on demand -------------
const mp = Number(ex.temen_alloc(guest.length));
u8().set(guest, mp);
ex.temen_run_onramp(mp, guest.length, 0, 0);
const want = { status: ex.temen_status(), out: readStdout() };
ex.temen_dealloc(mp, guest.length);
if (want.status !== 0) throw new Error(`oracle sanity: status ${want.status}, expected 0`);
if (want.out.length !== 8) throw new Error(`oracle sanity: ${want.out.length} stdout bytes, expected 8`);

// ---- 2. the playground path, counting what the driver actually did -------------------------------
// The counters ride a Proxy over the cdylib exports (the `bench_tierup_cards.mjs` trick, zero engine
// changes): one TIERUP service reads `temen_coop_func` once, every bounce goes through
// `temen_coop_call_interp`, and `temen_coop_open` returning 0 means the coop path was taken.
const counts = { tierups: 0, bounces: 0, path: 'interp' };
let winAtOpen = 0;
const exCounted = Object.fromEntries(
  Object.entries(Object.getOwnPropertyDescriptors(ex)).map(([k, d]) => {
    const v = d.value;
    if (typeof v !== 'function') return [k, v];
    if (k === 'temen_coop_func') return [k, (...a) => { counts.tierups++; return v(...a); }];
    if (k === 'temen_coop_call_interp') return [k, (...a) => { counts.bounces++; return v(...a); }];
    if (k === 'temen_coop_open') {
      return [k, (...a) => {
        const r = v(...a);
        if (r === 0) { counts.path = 'coop'; winAtOpen = ex.temen_coop_win_len(); }
        return r;
      }];
    }
    return [k, v];
  }));

// The guest's leaf is padded to clear the production tier-up size floor, but pin the floor at its
// default explicitly so this test states what it depends on rather than inheriting it.
ex.temen_coop_set_tierup_floor(4096);

// The window's final size has to be read while the session is still open, so sample it at the last
// bounce (the run closes its session before `runJitModule` returns).
let winAtLastBounce = 0;
const exWatched = { ...exCounted,
  temen_coop_call_interp: (...a) => {
    const r = exCounted.temen_coop_call_interp(...a);
    winAtLastBounce = ex.temen_coop_win_len();
    return r;
  } };

// A trap on the emitted tier surfaces as a throw ("cooperative tier-up run trapped") — that is
// precisely the pre-fix symptom, so name it rather than letting the stack trace speak.
let status;
try {
  status = await runJitModule(exWatched, memory, guest, new Uint8Array(0), 'coop-grow');
} catch (e) {
  console.error(`FAIL: the coop run did not complete: ${e.message}`);
  console.error('       (a trap here is the #1312 symptom: the allocator\'s vm_map past the run '
    + 'window was refused, and the store through it faulted)');
  process.exit(1);
}
const got = { status: ex.temen_status(), out: readStdout() };

// ---- 3. assertions -------------------------------------------------------------------------------
const fail = (msg) => { console.error(`FAIL: ${msg}`); process.exitCode = 1; };

if (counts.path !== 'coop') fail(`the coop driver must serve this guest, took '${counts.path}'`);
if (counts.tierups < 1) fail(`the emitted tier must drive: ${counts.tierups} tier-ups`);
if (counts.bounces < 1) fail(`the leaf must bounce to its grow helper: ${counts.bounces} bounces`);
if (got.status !== want.status) fail(`status ${got.status}, oracle ${want.status}`);
if (got.out !== want.out) {
  const hex = (s) => [...s].map((c) => c.charCodeAt(0).toString(16).padStart(2, '0')).join('');
  fail(`stdout ${hex(got.out)}, oracle ${hex(want.out)}`);
}
if (status !== 0) fail(`runJitModule returned status ${status}`);
if (!(winAtLastBounce > winAtOpen)) {
  fail(`the window must grow past its ${winAtOpen}-byte opening size, saw ${winAtLastBounce}`);
}

// ---- 4. #1954: the root run whole as a leaf --------------------------------------------------------
// The fixture's root spawns a thread, so it can never be a leaf. This guest is its single-threaded
// reduction: the root calls a function that bounces to the `vm_map` helper, stores through the new
// page past the window, and writes the result to stdout.
const WIN = 1 << 25;
const LEAF_SRC = String.raw`memory 16
data ro 40960 "stdout\x00\x00memory"
func () -> (i64) {
block 0 () {
  vseed = i64.const 11
  vr = call 1 (vseed)
  vsl = i64.const 34816
  i64.store vsl vr
  vnp = i64.const 40960
  vnl = i64.const 6
  vout = self.resolve vnp vnl
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vacc: i64) {
  vg = call 2 (vacc)
  i64.store vg vacc
  vld = i64.load vg
  vsum = i64.add vacc vld
  return vsum
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 40968
  vnl = i64.const 6
  vas = self.resolve vnp vnl
  voff = i64.const ${WIN}
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vprobe = i64.const ${WIN + 16}
  return vprobe
  }
}
export 0 func "_start" 0
`;
const src = new TextEncoder().encode(LEAF_SRC);
const sp = Number(ex.temen_alloc(src.length));
u8().set(src, sp);
const parsed = ex.temen_parse(sp, src.length);
ex.temen_dealloc(sp, src.length);
const leafGuest = u8().slice(Number(ex.temen_parse_ptr()), Number(ex.temen_parse_ptr()) + Number(ex.temen_parse_len()));
if (parsed !== 1) throw new Error(`root-leaf guest: ${dec.decode(leafGuest)}`);
const gp = Number(ex.temen_alloc(leafGuest.length));
u8().set(leafGuest, gp);
ex.temen_run_onramp(gp, leafGuest.length, 0, 0);
const leafWant = { status: ex.temen_status(), out: readStdout() };
if (leafWant.status !== 0 || leafWant.out.length !== 8) {
  throw new Error(`root-leaf oracle sanity: status ${leafWant.status}, ${leafWant.out.length} bytes`);
}
const COOP_LEAF_ROOT = 1;
const opened = ex.temen_coop_open(gp, leafGuest.length, 0, 0, 1, 0, 0, COOP_LEAF_ROOT);
ex.temen_dealloc(gp, leafGuest.length);
if (opened !== 0) {
  fail(`the root-leaf open refused (status ${ex.temen_status()})`);
} else {
  const leafOpen = {
    region: ex.temen_coop_wasm_len(),
    leaf: ex.temen_coop_leaf_wasm_len(0),
    win: ex.temen_coop_win_len(),
  };
  let leafWin = 0;
  const exLeaf = { ...ex,
    temen_coop_call_interp: (...a) => {
      const r = ex.temen_coop_call_interp(...a);
      leafWin = ex.temen_coop_win_len();
      return r;
    } };
  const leafCounts = {};
  const leafStatus = await driveCoopTierupRun(exLeaf, memory, { counts: leafCounts });
  const leafGot = { status: ex.temen_status(), out: readStdout() };
  if (leafOpen.region !== 0 || leafOpen.leaf === 0) {
    fail(`program 0 must be the root's leaf with no region emit: ${JSON.stringify(leafOpen)}`);
  }
  if (leafCounts.leaves !== 1) fail(`the root must run as one leaf: ${leafCounts.leaves}`);
  if (leafStatus !== 0 || leafGot.status !== leafWant.status) {
    fail(`root leaf: status ${leafGot.status} (returned ${leafStatus}), oracle ${leafWant.status}`);
  }
  if (leafGot.out !== leafWant.out) fail('root leaf: stdout differs from the oracle');
  if (!(leafWin > leafOpen.win)) {
    fail(`root leaf: the window must grow past ${leafOpen.win} bytes, saw ${leafWin}`);
  }
}

if (process.exitCode) process.exit(process.exitCode);
console.log(`ok — coop path, ${counts.tierups} tier-up(s), ${counts.bounces} bounce(s), `
  + `window ${winAtOpen} → ${winAtLastBounce} bytes, stdout matches the bytecode oracle; `
  + 'as a root leaf, the same over a growing window');
