// #1312 — **the cooperative tier-up window grows**, checked through the real JS driver.
//
// The native `coop_tierup_driver.rs` differential plays the browser's role with wasmi; this is the
// other half: the *actual* `driveCoopTierupRun` in `web/coop-driver.js`, over the *actual* wasm
// cdylib, on the path the playground takes (`runJitModule` → whole-program open declines → coop).
// On the threads build a pump root's window is a memory of its own (#1414 5d), which grows in place;
// the plain build's grows in the engine's memory, relocating, and the driver's `"win"` fan-out after a
// bounce follows it there.
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
// #2248: then a child that runs its program's child image, a granted module, tiers up through the
// image's own emit, a program of the run's besides its own, over its own window: a memory of its own
// (#1414 5d), which the engine gives back once the child has ended, and the page keeps, zeroed, for
// the next child.
//
// #1414 5d: then a root's memory of its own, against the oracle each time: a root leaf grown past
// 256 MiB, which leaves the engine's memory as it was; a map past a smaller grant, refused as the
// oracle refuses it (`temen_set_mem_grant`), and allowed again under the default; a window no memory
// can hold, whose open fails cleanly; and a collecting guest whose emitted frames spill its roots,
// which the bounce copies to where `gc.roots` scans them.
//
// Usage:  node browser-coop-grow-test.mjs [module.wasm]   (build the threads cdylib first)

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { driveCoopTierupRun, runJitModule } from './web/wasmjit-module.js';
import { instantiateEngine } from './engine-imports.mjs';
import { foreignMemory, headerOf } from './web/foreign-mem.js';

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
const ex = await instantiateEngine(mod, memory);
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

// The guest calls its leaf once: tier up every eligible call at its start (#2474), so the leaf runs
// emitted, and say so rather than inheriting the default.
ex.temen_coop_set_hotness(0, 0);

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
const leafSrc = (WIN) => String.raw`memory 16
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
const LEAF_SRC = leafSrc(WIN);
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
    leaf: ex.temen_coop_program_wasm_len(0),
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

// ---- 5. #2248: a granted child's program tiers up -------------------------------------------------
// The root spawns its child image (granted as `"child"` from its `_child` export) detached, joins
// it and writes its result. The image's bootstrap calls f1, which the image's emit tiers up whole:
// f1 calls f2 natively, and f2 bounces to f3, which stays interpreted (it resolves a name), for the
// address it then stores through and loads back — in the CHILD's window, the pending task's.
const CHILD_SRC = String.raw`memory 16
data ro 40960 "instantiator"
data ro 40976 "budget"
data ro 40984 "child"
data ro 40992 "stdout"
func () -> (i64) {
block 0 () {
  vip = i64.const 40960
  vil = i64.const 12
  vinst = self.resolve vip vil
  vbp = i64.const 40976
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vcp = i64.const 40984
  vcl = i64.const 5
  vc = self.resolve vcp vcl
  vrec = i64.const 34816
  vw0 = i64.const -4294967296
  i64.store vrec vw0
  i32.store vrec vc offset=8
  i32.store vrec vb offset=12
  vw6 = i64.const 4294967295
  i64.store vrec vw6 offset=48
  vh = call.cap 6 17 (i64) -> (i32) vinst (vrec)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vinst (vh)
  vsl = i64.const 35840
  i64.store vsl vj
  vop = i64.const 40992
  vol = i64.const 6
  vout = self.resolve vop vol
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vj
  }
}
func (i64) -> (i64) {
block 0 (vinst: i64) {
  vseed = i64.const 11
  vr = call 2 (vseed)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vacc: i64) {
  vg = call 3 (vacc)
  i64.store vg vacc
  vld = i64.load vg
  vsum = i64.add vacc vld
  return vsum
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 40992
  vnl = i64.const 6
  vh = self.resolve vnp vnl
  vaddr = i64.const 32768
  return vaddr
  }
}
export 0 func "_start" 0
export 1 func "_child" 1
`;
const csrc = new TextEncoder().encode(CHILD_SRC);
const cs = Number(ex.temen_alloc(csrc.length));
u8().set(csrc, cs);
const cparsed = ex.temen_parse(cs, csrc.length);
ex.temen_dealloc(cs, csrc.length);
const childGuest = u8().slice(Number(ex.temen_parse_ptr()), Number(ex.temen_parse_ptr()) + Number(ex.temen_parse_len()));
if (cparsed !== 1) throw new Error(`child guest: ${dec.decode(childGuest)}`);
const cp = Number(ex.temen_alloc(childGuest.length));
u8().set(childGuest, cp);
ex.temen_run_onramp(cp, childGuest.length, 0, 0);
const childWant = { status: ex.temen_status(), out: readStdout() };
if (childWant.status !== 0 || childWant.out.length !== 8) {
  throw new Error(`child oracle sanity: status ${childWant.status}, ${childWant.out.length} bytes`);
}
const childOpened = ex.temen_coop_open(cp, childGuest.length, 0, 0, 1, 0, 0, 0);
ex.temen_dealloc(cp, childGuest.length);
// The program each tier-up names, the memory it names, each bounce's pending program, and each
// program whose own emit the driver loaded.
const childPrograms = [];
const childMems = [];
const childBounces = [];
const childLoaded = [];
// What the child's own memory holds where f2 stores, as each of its emitted calls returns.
const childStored = [];
if (childOpened !== 0) {
  fail(`the child guest's open refused (status ${ex.temen_status()})`);
} else {
  const exChild = { ...ex,
    temen_coop_module: () => {
      const m = ex.temen_coop_module();
      childPrograms.push(m);
      return m;
    },
    temen_coop_tierup_mem: () => {
      const mem = ex.temen_coop_tierup_mem();
      childMems.push(mem);
      return mem;
    },
    temen_coop_deliver: (...a) => {
      const mem = childMems.at(-1);
      if (mem >= 0) {
        const own = foreignMemory(memory, mem);
        childStored.push(new BigInt64Array(own.buffer)[(headerOf(own).base + 32768) / 8]);
      }
      return ex.temen_coop_deliver(...a);
    },
    temen_coop_call_interp: (...a) => {
      childBounces.push(ex.temen_coop_module());
      return ex.temen_coop_call_interp(...a);
    },
    temen_coop_program_wasm_ptr: (m) => {
      childLoaded.push(m);
      return ex.temen_coop_program_wasm_ptr(m);
    } };
  const childStatus = await driveCoopTierupRun(exChild, memory, {});
  const childGot = { status: ex.temen_status(), out: readStdout() };
  if (!childPrograms.includes(1)) {
    fail(`the child must tier up in its image, program 1: tier-ups named ${JSON.stringify(childPrograms)}`);
  }
  // An image keeps the program's functions where they are, so the run's own emit would compute the
  // same: the driver must have run the image's own.
  if (JSON.stringify(childLoaded) !== '[1]') {
    fail(`the driver must load program 1's own emit once: loaded ${JSON.stringify(childLoaded)}`);
  }
  if (!childBounces.includes(1)) {
    fail(`the child's emitted frames must bounce to f3: bounces from ${JSON.stringify(childBounces)}`);
  }
  // #1414 5d: the child's window is a memory of its own, which each of its tier-ups names and the
  // root's do not, and which the engine gave back when the child ended: the page keeps it, its
  // window zeroed.
  const named = childPrograms.map((m, i) => [m, childMems[i]]);
  const own = [...new Set(named.filter(([m]) => m === 1).map(([, mem]) => mem))];
  const kept = own.length === 1 ? foreignMemory(memory, own[0]) : undefined;
  if (own.length !== 1 || !(own[0] >= 0) || named.some(([m, mem]) => m !== 1 && mem === own[0])) {
    fail(`program 1's tier-ups must name one memory of the child's own: ${JSON.stringify(named)}`);
  } else if (!childStored.includes(11n)) {
    fail(`f2's store must land in the child's own memory: it held ${childStored}`);
  } else if (!kept || !new Uint8Array(kept.buffer, headerOf(kept).base).every((b) => b === 0)) {
    fail(`the child's memory ${own[0]} must come back zeroed once the child has ended`);
  }
  if (childStatus !== 0 || childGot.status !== childWant.status) {
    fail(`child: status ${childGot.status} (returned ${childStatus}), oracle ${childWant.status}`);
  }
  if (childGot.out !== childWant.out) fail('child: stdout differs from the oracle');
}

// ---- 6. #1414 5d: a root's window is a memory of its own --------------------------------------------
const show = (v) => JSON.stringify(v, (_, x) => (typeof x === 'bigint' ? `${x}n` : x));
const parse = (text) => {
  const bytes = new TextEncoder().encode(text);
  const p = Number(ex.temen_alloc(bytes.length));
  u8().set(bytes, p);
  const ok = ex.temen_parse(p, bytes.length);
  ex.temen_dealloc(p, bytes.length);
  const out = u8().slice(Number(ex.temen_parse_ptr()), Number(ex.temen_parse_ptr()) + Number(ex.temen_parse_len()));
  if (ok !== 1) throw new Error(`parse: ${dec.decode(out)}`);
  return out;
};
// `f(ptr, len)` over a copy of `guest` in the engine's memory.
const withGuest = (guest, f) => {
  const p = Number(ex.temen_alloc(guest.length));
  u8().set(guest, p);
  try {
    return f(p, guest.length);
  } finally {
    ex.temen_dealloc(p, guest.length);
  }
};
// The 8 bytes a guest wrote to stdout, as the i64 it wrote.
const outWord = () => {
  const p = Number(ex.temen_stdout_ptr());
  const n = Number(ex.temen_stdout_len());
  return n === 8 ? new DataView(u8().slice(p, p + 8).buffer).getBigInt64(0, true) : `${n} bytes`;
};
const oracle = (guest) => withGuest(guest, (p, n) => {
  ex.temen_run_onramp(p, n, 0, 0);
  return { status: ex.temen_status(), out: outWord() };
});
// Open `guest` on the pump in `mode` and drive it, noting the memories its events name, its tier-ups,
// the most words a bounce carried spilled, and how far the engine's memory grew.
const coop = async (guest, mode) => {
  const r = { mems: new Set(), tierups: 0, spill: 0 };
  const before = memory.buffer.byteLength;
  r.opened = withGuest(guest, (p, n) => ex.temen_coop_open(p, n, 0, 0, 1, 0, 0, mode));
  if (r.opened !== 0) return r;
  const exSeen = { ...ex,
    temen_coop_tierup_mem: () => {
      const mem = ex.temen_coop_tierup_mem();
      r.mems.add(mem);
      return mem;
    },
    temen_coop_func: () => {
      r.tierups++;
      return ex.temen_coop_func();
    },
    temen_coop_call_interp: (target, argsPtr, spillLen) => {
      r.spill = Math.max(r.spill, spillLen);
      return ex.temen_coop_call_interp(target, argsPtr, spillLen);
    } };
  r.returned = await driveCoopTierupRun(exSeen, memory, {});
  r.got = { status: ex.temen_status(), out: outWord() };
  r.grew = memory.buffer.byteLength - before;
  return r;
};
const own = (r) => r.mems.size === 1 && [...r.mems][0] >= 0;
const same = (r, want) => r.returned === 0 && r.got.status === want.status && r.got.out === want.out;

// 6a. The root leaf of section 4, its map moved past 256 MiB: its window grows there in its own memory.
const BIG = 300 << 20;
const bigGuest = parse(leafSrc(BIG));
const bigWant = oracle(bigGuest);
const big = await coop(bigGuest, COOP_LEAF_ROOT);
if (big.opened !== 0) {
  fail(`a root that grows past 256 MiB must open: ${big.opened}`);
} else {
  if (!same(big, bigWant)) fail(`a root grown past 256 MiB: ${show(big.got)}, oracle ${show(bigWant)}`);
  if (!own(big)) fail(`a root's events must name one memory of its own: ${show([...big.mems])}`);
  if (big.grew > 16 << 20) {
    fail(`the engine's memory grew ${big.grew} bytes under a root grown past 256 MiB: the window is not its own`);
  }
}

// 6b. A root that maps 16 KiB at `at` and writes what the map returned.
const mapSrc = (at) => String.raw`memory 16
data ro 40960 "stdout\x00\x00memory"
func () -> (i64) {
block 0 () {
  vr = call 1 ()
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
func () -> (i64) {
block 0 () {
  vnp = i64.const 40968
  vnl = i64.const 6
  vas = self.resolve vnp vnl
  voff = i64.const ${at}
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  return vr
  }
}
export 0 func "_start" 0
`;
const mapGuest = parse(mapSrc(100 << 20));
ex.temen_set_mem_grant(BigInt(64 << 20));
const cappedWant = oracle(mapGuest);
const capped = await coop(mapGuest, COOP_LEAF_ROOT);
ex.temen_set_mem_grant(0n);
const freed = await coop(mapGuest, COOP_LEAF_ROOT);
if (cappedWant.out !== -12n) fail(`the oracle must refuse a map past a 64 MiB grant with -ENOMEM: ${cappedWant.out}`);
if (capped.opened !== 0 || !same(capped, cappedWant)) {
  fail(`a map past a 64 MiB grant: ${show(capped.got)}, oracle ${show(cappedWant)}`);
}
if (freed.opened !== 0 || freed.got?.out !== 0n) {
  fail(`under the default grant the same map must succeed: ${show(freed.got)}`);
}

// 6c. A window no wasm32 memory can hold: the open fails cleanly.
const huge = withGuest(parse(`memory 32
func () -> (i64) {
block 0 () {
  v = i64.const 0
  return v
  }
}
export 0 func "_start" 0
`), (p, n) => ex.temen_coop_open(p, n, 0, 0, 1, 0, 0, COOP_LEAF_ROOT));
if (huge !== -2) fail(`a 4 GiB window's open must fail with STATUS_UNSUPPORTED (-2): ${huge}`);

// 6d. A collector (#1627). f1 tiers up and holds `vroot`, in the scanned range, across its call to the
// collector, which stays interpreted: the emitted frame spills `vroot` before it bounces, and the
// bounce copies the spilled words out of the root's memory to where `gc.roots` scans them. The scan
// counts the collector's own `vlo` and `vroot`, as it does on the oracle, where f1's frame holds it.
const GC_SRC = String.raw`memory 16
data ro 40960 "stdout"
func () -> (i64) {
block 0 () {
  vp = i64.const 137120
  vr = call 1 (vp)
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
block 0 (vp: i64) {
  vk = i64.const 100000
  vroot = i64.sub vp vk
  vres = call 2 ()
  vd = i64.sub vroot vroot
  vnine = i64.const 9
  vs = i64.add vres vnine
  vt = i64.add vs vd
  return vt
  }
}
func () -> (i64) {
block 0 () {
  vlo = i64.const 36864
  vhi = i64.const 40960
  vmask = i64.const -1
  vbuf = i64.const 45056
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
  }
}
export 0 func "_start" 0
`;
const gcGuest = parse(GC_SRC);
const gcWant = oracle(gcGuest);
const gc = await coop(gcGuest, 0);
if (gcWant.status !== 0 || gcWant.out !== 11n) fail(`gc oracle sanity: ${show(gcWant)}`);
if (gc.opened !== 0) {
  fail(`the collecting root's open refused: ${gc.opened}`);
} else {
  if (gc.tierups < 1 || gc.spill < 1) {
    fail(`f1 must run emitted and spill across its bounce: ${gc.tierups} tier-up(s), ${gc.spill} word(s)`);
  }
  if (!own(gc)) fail(`the collecting root's events must name one memory of its own: ${show([...gc.mems])}`);
  if (!same(gc, gcWant)) fail(`the collector's roots: ${show(gc.got)}, oracle ${show(gcWant)}`);
}
ex.temen_coop_reset_hotness();

if (process.exitCode) process.exit(process.exitCode);
console.log(`ok — coop path, ${counts.tierups} tier-up(s), ${counts.bounces} bounce(s), `
  + `window ${winAtOpen} → ${winAtLastBounce} bytes, stdout matches the bytecode oracle; `
  + 'as a root leaf, the same over a growing window; a child image tiers up as program '
  + `${show([...new Set(childPrograms)])} over memories ${show([...new Set(childMems)])}; `
  + `a root grown past 256 MiB in a memory of its own (the engine's grew ${big.grew} bytes), a map past a `
  + `64 MiB grant refused (${capped.got?.out}), a 4 GiB window refused, and a collector's spilled roots `
  + `found (${gc.got?.out})`);
