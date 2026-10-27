// #1954 — the embedder cases `browser-coop-embedder-test.mjs` runs, in any host that has the engine:
// Node, and Chromium, whose JSPI suspends a root leaf's emitted frames where a call parks. Each case
// opens a program as a tier-up session (`temen_coop_open`) the way an embedder's Release button does
// — with declared host-completed caps, a root-leaf mode, and no regions — and runs it through
// `driveCoopTierupRun`'s embedder hooks: slices (`budget`/`onSlice`), streamed output (`onOutput`),
// answered cap parks (`onCapPark`) and reported traps (`trapDeclines: false`). `runCases` reports what
// each run did; the caller checks it.

import { onCap as jsCaps } from './coop-embedder-jscaps.mjs';

// Where executor 2's Workers import the same capabilities from (`capModule`, #2645).
const JS_CAPS_MODULE = new URL('./coop-embedder-jscaps.mjs', import.meta.url).href;

const COOP_LEAF_ROOT = 1;
const COOP_LEAF_SUSPENDS = 2;
const COOP_NO_REGIONS = 4;

// The ping guest: `_start` asks the embedder `ping(i)` for each `i` in `0..n` through the declared cap
// `ping`, sums the answers, writes the sum to stdout and returns it.
const pingGuest = (n) => `memory 16
import 0 "ping" (i64) -> (i64)
import 1 "write" (i64, i64) -> (i64)
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz, vz)
}
block 1 (vi: i64, vs: i64) {
  vp = call.import 0 (vi)
  vs2 = i64.add vs vp
  vone = i64.const 1
  vi2 = i64.add vi vone
  vn = i64.const ${n}
  vgo = i64.ne vi2 vn
  br_if vgo 1(vi2, vs2) 2(vs2)
}
block 2 (vr: i64) {
  vsl = i64.const 34816
  i64.store vsl vr
  vlen8 = i64.const 8
  vw = call.import 1 (vsl, vlen8)
  return vr
  }
}
export 0 func "_start" 0
`;

// A guest that writes "hi\n" and then never ends.
const SPIN = `memory 16
import 0 "write" (i64, i64) -> (i64)
data 34816 "hi\\n"
func () -> (i64) {
block 0 () {
  vsl = i64.const 34816
  vlen = i64.const 3
  vw = call.import 0 (vsl, vlen)
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  br 1(vi2)
  }
}
export 0 func "_start" 0
`;

// A guest that writes "before\n" and then loads from address 8, inside the NULL guard.
const FAULTS = `memory 16
import 0 "write" (i64, i64) -> (i64)
data 34816 "before\\n"
func () -> (i64) {
block 0 () {
  vsl = i64.const 34816
  vlen = i64.const 7
  vw = call.import 0 (vsl, vlen)
  va = i64.const 8
  vv = i64.load va
  return vv
  }
}
export 0 func "_start" 0
`;

// #2126 — a guest that writes "before\n" and then divides by a zero it loads, a trap the wasm engine
// raises itself (no `env.trap`) in emitted code.
const DIVIDES = `memory 16
import 0 "write" (i64, i64) -> (i64)
data 34816 "before\\n"
func () -> (i64) {
block 0 () {
  vsl = i64.const 34816
  vlen = i64.const 7
  vw = call.import 0 (vsl, vlen)
  vza = i64.const 34824
  vz = i64.load vza
  vn = i64.const 42
  vq = i64.div_s vn vz
  return vq
  }
}
export 0 func "_start" 0
`;

// A guest that writes "tick\n" 50 times, spinning 2000 back-edges between writes, and returns 50.
const TICKS = `memory 16
import 0 "write" (i64, i64) -> (i64)
data 34816 "tick\n"
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vsl = i64.const 34816
  vlen = i64.const 5
  vw = call.import 0 (vsl, vlen)
  vz = i64.const 0
  br 2(vi, vz)
}
block 2 (vi2: i64, vj: i64) {
  vone = i64.const 1
  vj2 = i64.add vj vone
  vn = i64.const 2000
  vgo = i64.ne vj2 vn
  br_if vgo 2(vi2, vj2) 3(vi2)
}
block 3 (vi3: i64) {
  vone3 = i64.const 1
  vi4 = i64.add vi3 vone3
  vm = i64.const 50
  vmore = i64.ne vi4 vm
  br_if vmore 1(vi4) 4(vi4)
}
block 4 (vr: i64) {
  return vr
  }
}
export 0 func "_start" 0
`;

// #2557 — the root waits 1 ms on a word nobody notifies while its thread asks the embedder `ping(7)`,
// reads the flag the thread raises once answered, and joins it: `2` (timed out) * 10000 + the flag
// * 1000 + the answer.
const TIMER_WHILE_OWED = `memory 16
import 0 "ping" (i64) -> (i64)
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vaddr = i64.const 36864
  vexp = i32.const 0
  vto = i64.const 1000000
  vst = i32.atomic.wait vaddr vexp vto
  vflagat = i64.const 40960
  vflag = i64.load vflagat
  vr_st, vr = thread.join vt
  vst64 = i64.extend_i32_s vst
  vk = i64.const 10000
  vs = i64.mul vst64 vk
  vk2 = i64.const 1000
  vf = i64.mul vflag vk2
  vsf = i64.add vs vf
  vsum = i64.add vsf vr
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vseven = i64.const 7
  vr = call.import 0 (vseven)
  vaddr = i64.const 40960
  vone = i64.const 1
  i64.store vaddr vone
  return vr
  }
}
export 0 func "_start" 0
`;

// #2439 — a guest that waits `ms` on a word nobody notifies, and returns what the wait answered.
const loneWait = (ms) => `memory 16
func () -> (i64) {
block 0 () {
  vaddr = i64.const 36864
  vexp = i32.const 0
  vto = i64.const ${ms * 1000000}
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
export 0 func "_start" 0
`;
export const LONE_WAIT_MS = 30;

// #2579 — a guest of the JS capabilities its embedder staged, each answered inside the call
// (`onCap`): `js.upper(ptr, len)` uppercases "shout\n" in the guest's window and returns the length,
// and `js.add` doubles it. The guest writes the line and returns 12.
const JS_CAPS = `memory 16
import 0 "js.upper" (i64, i64) -> (i64)
import 1 "js.add" (i64, i64) -> (i64)
import 2 "write" (i64, i64) -> (i64)
data 34816 "shout\\n"
func () -> (i64) {
block 0 () {
  vp = i64.const 34816
  vl = i64.const 6
  vn = call.import 0 (vp, vl)
  vs = call.import 1 (vn, vn)
  vw = call.import 2 (vp, vl)
  return vs
  }
}
export 0 func "_start" 0
`;
export const JS_CAPS_OUT = 'SHOUT\n';

// #2579 — on executor 2, `JS_CAPS`'s calls made by a thread of the root, on a Worker of its own: it
// uppercases "shout\n" and doubles its length, and returns 1000 * the first byte of the window it
// wrote + 12: 83012.
const JS_CAPS_THREAD = `memory 16
import 0 "js.upper" (i64, i64) -> (i64)
import 1 "js.add" (i64, i64) -> (i64)
data 34816 "shout\\n"
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vr_st, vr = thread.join vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vp = i64.const 34816
  vl = i64.const 6
  vn = call.import 0 (vp, vl)
  vs = call.import 1 (vn, vn)
  vb = i64.load8_u vp
  vk = i64.const 1000
  vm = i64.mul vb vk
  vr = i64.add vm vs
  return vr
  }
}
export 0 func "_start" 0
`;
export const JS_CAPS_THREAD_VALUE = 83012n;

// #2579 — a detached child's call to a JS capability its root re-granted it by name: on executor 2
// the child's window is a memory of its own, which the page reads and writes through the call's
// view. The root spawns the staged module `kid` (an op-17 record at 35328 granting `js.upper` from
// the record at 35072), joins it and returns what it returns: 1000 * the first byte of its window
// after the call ('C') + the length the call returned, 67006.
const JS_CAPS_KID = `memory 16
import 0 "js.upper" (i64, i64) -> (i64)
data 34816 "child\\n"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vp = i64.const 34816
  vl = i64.const 6
  vn = call.import 0 (vp, vl)
  vb = i64.load8_u vp
  vk = i64.const 1000
  vm = i64.mul vb vk
  vr = i64.add vm vn
  return vr
  }
}
`;
const le = (v, n) => Array.from({ length: n }, (_, i) => `\\x${((v >>> (8 * i)) & 0xff).toString(16).padStart(2, '0')}`).join('');
// The 64-byte spawn record: a 64 KiB window, no pager, the module and budget stored at run time,
// one named grant at 35072, no args, no region.
const KID_REC = le(16, 4) + le(-1, 4) + le(0, 8) + le(35072, 4) + le(0, 4) + le(1, 4) + le(0, 4)
  + le(0, 16) + le(-1, 4) + le(0, 4) + le(0, 8);
const JS_CAPS_CHILD = `memory 16
data 34816 "instantiator"
data 34832 "kid"
data 34840 "budget"
data 34848 "js.upper"
data 35328 "${KID_REC}"
func () -> (i64) {
block 0 () {
  vip = i64.const 34816
  vil = i64.const 12
  vinst = self.resolve vip vil
  vkp = i64.const 34832
  vkl = i64.const 3
  vkid = self.resolve vkp vkl
  vbp = i64.const 34840
  vbl = i64.const 6
  vbud = self.resolve vbp vbl
  vjp = i64.const 34848
  vjl = i64.const 8
  vjs = self.resolve vjp vjl
  vg = i64.const 35072
  vgn = i32.const 34848
  i32.store vg vgn
  vgl = i32.const 8
  i32.store vg vgl offset=4
  i32.store vg vjs offset=8
  vgf = i32.const 0
  i32.store vg vgf offset=12
  vrp = i64.const 35328
  i32.store vrp vkid offset=8
  i32.store vrp vbud offset=12
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vst, vj = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
  return vj
  }
}
export 0 func "_start" 0
`;
export const JS_CAPS_CHILD_VALUE = 67006n;

// #2645 — a thread's JS call that waits for another thread's host calls: thread B writes a byte to
// stdout, then counts it at 40960, until thread A is done (the word at 40968); A waits until B has
// counted one, then calls `js.waitFor` (slot 2) on the counter. A call that held the domain's host
// kept B's writes out until it gave up (0); one that lets it go sees them (1). A waits for B first
// because a call answered on the page holds the page, which starts the run's Workers. The root
// returns A's answer.
const JS_CAPS_WAITS = `memory 16
import 0 "js.waitFor" (i64, i64) -> (i64)
import 1 "write" (i64, i64) -> (i64)
data 34816 "."
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  va = thread.spawn 1 vz vz
  vb = thread.spawn 2 vz vz
  vra_st, vra = thread.join va
  vrb_st, vrb = thread.join vb
  return vra
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1(vsp)
}
block 1 (vs1: i64) {
  vc = i64.const 40960
  vn = i64.load vc
  vz = i64.const 0
  vnone = i64.eq vn vz
  br_if vnone 1(vs1) 2(vs1)
}
block 2 (vs2: i64) {
  vc2 = i64.const 40960
  vmore = i64.const 100
  vr = call.import 0 (vc2, vmore)
  vd = i64.const 40968
  vone = i64.const 1
  i64.store vd vone
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz0 = i64.const 0
  br 1(vz0)
}
block 1 (vn: i64) {
  vp = i64.const 34816
  vl = i64.const 1
  vw = call.import 1 (vp, vl)
  vone = i64.const 1
  vn2 = i64.add vn vone
  vc = i64.const 40960
  i64.store vc vn2
  vd = i64.const 40968
  vdone = i64.load vd
  vz1 = i64.const 0
  vgo = i64.eq vdone vz1
  br_if vgo 1(vn2) 2(vn2)
}
block 2 (vr: i64) {
  return vr
  }
}
export 0 func "_start" 0
`;

export const pong = (x) => 2n * x + 1n;
export const PINGS = 30;

export async function runCases({ ex, memory, drive, suspends }) {
  const u8 = () => new Uint8Array(memory.buffer);
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  const put = (bytes) => {
    const p = Number(ex.temen_alloc(bytes.length || 1));
    u8().set(bytes, p);
    return p;
  };
  const shared = typeof SharedArrayBuffer !== 'undefined' && memory.buffer instanceof SharedArrayBuffer;
  // Open `src` with `caps` declared in `mode`: the engine's status (0 = open).
  const encode = (src) => {
    const text = enc.encode(src);
    const sp = put(text);
    const ok = ex.temen_parse(sp, text.length);
    ex.temen_dealloc(sp, text.length || 1);
    const bytes = u8().slice(Number(ex.temen_parse_ptr()), Number(ex.temen_parse_ptr()) + Number(ex.temen_parse_len()));
    if (ok !== 1) throw new Error(`parse: ${dec.decode(bytes)}`);
    return bytes;
  };
  // #2579: stage JS capabilities for the next open: their slots.
  const stageJsCaps = (names) => names.map((name) => {
    const b = enc.encode(name);
    const p = put(b);
    const slot = ex.temen_onramp_stage_js_cap(p, b.length);
    ex.temen_dealloc(p, b.length || 1);
    return slot;
  });
  const open = (src, caps, mode) => {
    const bytes = encode(src);
    const mp = put(bytes);
    const names = enc.encode(caps.join('\n'));
    const cp = put(names);
    const r = ex.temen_coop_open(mp, bytes.length, 0, 0, shared ? 1 : 0, cp, names.length, mode);
    ex.temen_dealloc(mp, bytes.length);
    ex.temen_dealloc(cp, names.length || 1);
    return r;
  };
  // Drive the open session with the embedder hooks, recording what they saw.
  const run = async (opts = {}) => {
    const seen = { out: '', err: '', chunks: 0, slices: 0, parks: 0, capCalls: 0, counts: {} };
    const status = await drive(ex, memory, {
      counts: seen.counts,
      budget: 1000,
      onOutput: (o, e) => { seen.chunks++; seen.out += dec.decode(o); seen.err += dec.decode(e); },
      onSlice: async () => {
        seen.slices++;
        await new Promise((r) => setTimeout(r, 0)); // the embedder's event loop gets a turn
        return opts.stopAfter === undefined || seen.slices < opts.stopAfter;
      },
      onCapPark: async ({ index, args }) => {
        seen.parks++;
        if (opts.stopAtPark === seen.parks) return null;
        if (index !== 0) throw new Error(`unexpected cap ${index}`);
        // Answered asynchronously, as a UI would.
        await new Promise((r) => setTimeout(r, opts.answerAfter ?? 0));
        return pong(args[0]);
      },
      onCap: opts.onCap && ((...call) => {
        seen.capCalls++;
        return opts.onCap(...call);
      }),
      trapDeclines: opts.trapDeclines,
    });
    return { status, value: status === 0 ? ex.temen_run_value() : null, ...seen };
  };
  const trapName = () => dec.decode(u8().slice(Number(ex.temen_trap_ptr()), Number(ex.temen_trap_ptr()) + Number(ex.temen_trap_len())));

  // Where the host suspends, the root may park and still run as a leaf; else it runs interpreted.
  const leafMode = (suspends ? COOP_LEAF_SUSPENDS : COOP_LEAF_ROOT) | COOP_NO_REGIONS;
  const results = {};
  if (open(pingGuest(PINGS), ['ping'], leafMode) !== 0) throw new Error(`open ping: ${ex.temen_status()}`);
  results.pings = await run();
  if (open(pingGuest(PINGS), ['ping'], COOP_NO_REGIONS) !== 0) throw new Error('open ping, interpreted');
  results.pingsInterpreted = await run();
  // #2557: the answer comes 50 ms after its call, and the root's 1 ms timer fires meanwhile.
  if (open(TIMER_WHILE_OWED, ['ping'], COOP_NO_REGIONS) !== 0) throw new Error('open timer while owed');
  results.timerWhileOwed = await run({ answerAfter: 50 });
  // #2439: a lone wait takes its time: the driver waits the run's timer out.
  if (open(loneWait(LONE_WAIT_MS), [], COOP_NO_REGIONS) !== 0) throw new Error('open lone wait');
  const waitFrom = performance.now();
  results.loneWait = { ...(await run()), ms: performance.now() - waitFrom };
  // #2579: JS capabilities answered inside the call, interpreted and from a root leaf, whose calls
  // bounce out of its emitted code (under JSPI, out of frames the host could suspend).
  for (const [key, mode] of [['jsCaps', COOP_NO_REGIONS], ['jsCapsLeaf', leafMode]]) {
    if (stageJsCaps(['js.upper', 'js.add']).join() !== '0,1') throw new Error('stage the JS caps');
    if (open(JS_CAPS, [], mode) !== 0) throw new Error(`open ${key}: ${ex.temen_status()}`);
    results[key] = await run({ onCap: jsCaps });
  }
  if (open(pingGuest(PINGS), ['ping'], leafMode) !== 0) throw new Error('open ping to stop');
  results.stopAtPark = await run({ stopAtPark: 3 });
  // Interpreted: emitted code is not sliced, so a root leaf that never parks would never come back
  // to `onSlice` until emitted code checks a budget (#1954 step 4).
  if (open(SPIN, [], COOP_NO_REGIONS) !== 0) throw new Error('open spin');
  results.spin = await run({ stopAfter: 20 });
  // #1954 step 4 — where the host suspends, a leaf is sliced at its own safepoints: an endless leaf
  // stops at a checkpoint, a long one ends with its output streamed from inside it. (Where it cannot,
  // a leaf that never parks runs to its end without a checkpoint, so these run only under JSPI.)
  if (suspends) {
    if (open(SPIN, [], leafMode) !== 0) throw new Error('open spin leaf');
    results.spinLeaf = await run({ stopAfter: 20 });
    if (open(TICKS, [], leafMode) !== 0) throw new Error('open ticks leaf');
    results.ticksLeaf = await run();
    // #1822: a fault in the leaf's emitted code reaches the guest as the trap its `env.trap` named.
    if (open(FAULTS, [], leafMode) !== 0) throw new Error('open faults leaf');
    results.faultsLeaf = { ...(await run({ trapDeclines: false })), trap: trapName(), addr: ex.temen_fault_addr() };
    // #2126: a trap the engine raises itself is named from its message, as the interpreter names it.
    if (open(DIVIDES, [], leafMode) !== 0) throw new Error('open divides leaf');
    results.dividesLeaf = { ...(await run({ trapDeclines: false })), trap: trapName() };
  }
  if (open(TICKS, [], COOP_NO_REGIONS) !== 0) throw new Error('open ticks');
  results.ticks = await run();
  if (open(FAULTS, [], COOP_NO_REGIONS) !== 0) throw new Error('open faults');
  results.faults = { ...(await run({ trapDeclines: false })), trap: trapName(), addr: ex.temen_fault_addr() };
  if (open(DIVIDES, [], COOP_NO_REGIONS) !== 0) throw new Error('open divides');
  results.divides = { ...(await run({ trapDeclines: false })), trap: trapName() };
  if (open(FAULTS, [], COOP_NO_REGIONS) !== 0) throw new Error('open faults again');
  results.faultsDecline = await run().then(() => 'returned', (e) => String(e.message));
  return results;
}

// #2531's shape on an on-ramp root: the root spins on a flag that its thread sets once its `ping(7)`
// returns, then joins the thread and returns its answer, `pong(7)`.
const THREAD_PING = `memory 16
import 0 "ping" (i64) -> (i64)
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  br 1(vt)
}
block 1 (vt1: i32) {
  vaddr = i64.const 40960
  vflag = i64.load vaddr
  vzero = i64.const 0
  vwait = i64.eq vflag vzero
  br_if vwait 1(vt1) 2(vt1)
}
block 2 (vt2: i32) {
  vr_st, vr = thread.join vt2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vseven = i64.const 7
  vr = call.import 0 (vseven)
  vaddr = i64.const 40960
  vone = i64.const 1
  i64.store vaddr vone
  return vr
  }
}
export 0 func "_start" 0
`;

// A guest that copies stdin to stdout until stdin ends, and returns how many bytes it copied.
const ECHO = `memory 16
import 0 "read" (i64, i64) -> (i64)
import 1 "write" (i64, i64) -> (i64)
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vtot: i64) {
  vbuf = i64.const 40960
  vcap = i64.const 256
  vn = call.import 0 (vbuf, vcap)
  vzero = i64.const 0
  vmore = i64.gt_s vn vzero
  br_if vmore 2(vtot, vn) 3(vtot)
}
block 2 (vt2: i64, vn2: i64) {
  vbuf2 = i64.const 40960
  vw = call.import 1 (vbuf2, vn2)
  vt3 = i64.add vt2 vn2
  br 1(vt3)
}
block 3 (vr: i64) {
  return vr
  }
}
export 0 func "_start" 0
`;

// #1414 5b-2 — the x2 row: the same embedder's runs on executor 2, each task on a Worker of its own
// (`par.js`'s `makeRunner`, so Chromium only). The page answers each call it is told of, and feeds
// input, while the run's other tasks go on: it leaves both in the engine's memory and wakes the
// Worker that pumps the run. `runX2Cases` reports what each run did; the caller checks it.
export async function runX2Cases({ par, eng }) {
  const { ex, memory } = eng;
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  const run = par.makeRunner(eng);
  const parsed = (src) => {
    const text = enc.encode(src);
    const sp = Number(ex.temen_alloc(text.length));
    new Uint8Array(memory.buffer).set(text, sp);
    const ok = ex.temen_parse(sp, text.length);
    ex.temen_dealloc(sp, text.length);
    const p = Number(ex.temen_parse_ptr());
    const bytes = new Uint8Array(memory.buffer).slice(p, p + Number(ex.temen_parse_len()));
    if (ok !== 1) throw new Error(`parse: ${dec.decode(bytes)}`);
    return bytes;
  };
  // Run `src` on executor 2 under the on-ramp powerbox with `caps` declared, recording what the
  // page saw. `fed` makes its stdin fed later: `onIdle(r)` gets each idle, and `start(r)` the run
  // as it starts. With `tierup`, the run has its emitted tier, and the case records how many calls
  // ran there and whether the run took its root whole as a leaf (#1414 5c). A run still going after
  // `X2_HANG_MS` is stopped, and the cases end there with what the page saw of it: the harness
  // fails rather than hangs.
  // #2579: what the page stages for the next open, as the pump's cases stage it: JS capabilities
  // (each takes the next slot) and child modules by name.
  const put = (bytes) => {
    const p = Number(ex.temen_alloc(bytes.length || 1));
    new Uint8Array(memory.buffer).set(bytes, p);
    return p;
  };
  const stage = (js, kids) => {
    js.forEach((name, slot) => {
      const b = enc.encode(name);
      const p = put(b);
      if (ex.temen_onramp_stage_js_cap(p, b.length) !== slot) throw new Error(`stage ${name}`);
      ex.temen_dealloc(p, b.length || 1);
    });
    for (const [name, src] of kids) {
      const n = enc.encode(name), m = parsed(src);
      const np = put(n), mp = put(m);
      if (ex.temen_onramp_stage_module(np, n.length, mp, m.length) !== 0) throw new Error(`stage ${name}`);
      ex.temen_dealloc(np, n.length || 1);
      ex.temen_dealloc(mp, m.length || 1);
    }
  };
  const go = async (src, { caps = [], stopAtPark, answerAfter = 0, fed = false, onIdle, start, tierup = false, js = [], kids = [], onCap, capModule } = {}) => {
    const seen = { out: '', parks: 0, idles: 0, capCalls: 0 };
    const abort = new AbortController();
    stage(js, kids);
    const r = run(parsed(src), {
      onramp: true,
      caps,
      tierup,
      stdinLater: fed,
      signal: abort.signal,
      onCap: onCap && ((...call) => {
        seen.capCalls++;
        return onCap(...call);
      }),
      capModule,
      onOutput: (o) => { seen.out += dec.decode(o); },
      onCapPark: async ({ index, args }) => {
        seen.parks++;
        if (stopAtPark === seen.parks) return null;
        if (index !== 0) throw new Error(`unexpected cap ${index}`);
        await new Promise((res) => setTimeout(res, answerAfter)); // answered asynchronously, as a UI would
        return pong(args[0]);
      },
      onIdle: () => {
        seen.idles++;
        if (onIdle) onIdle(r, seen.idles);
      },
    });
    if (start) start(r);
    let timer;
    const hung = new Promise((_, reject) => {
      timer = setTimeout(() => {
        abort.abort();
        const saw = `${seen.parks} parks, ${seen.idles} idles, ${seen.out.length} bytes out`;
        reject(Object.assign(new Error(`no end after ${X2_HANG_MS} ms (${saw})`), { hung: true }));
      }, X2_HANG_MS);
    });
    try {
      const done = await Promise.race([r, hung]);
      const leaf = ex.temen_x2_leaf_len(0) > 0;
      return { value: done.value, tierups: done.tierups, childMems: done.childMems, leaf, ...seen };
    } catch (e) {
      if (e.hung) throw e;
      return { error: e.message, faultAddr: e.faultAddr, ...seen };
    } finally {
      clearTimeout(timer);
    }
  };
  const results = {};
  results.pings = await go(pingGuest(PINGS), { caps: ['ping'] });
  results.stopAtPark = await go(pingGuest(PINGS), { caps: ['ping'], stopAtPark: 3 });
  results.threadPing = await go(THREAD_PING, { caps: ['ping'] });
  // #2557's case: the root's 1 ms timer fires while its thread's answer is owed, 50 ms away.
  results.timerWhileOwed = await go(TIMER_WHILE_OWED, { caps: ['ping'], answerAfter: 50 });
  // Fed as the run idles on its reads: a part at the first, the rest and the end at the second.
  results.echoLater = await go(ECHO, {
    fed: true,
    onIdle: (r, n) => {
      if (n === 1) r.feed(enc.encode('hello '));
      if (n === 2) {
        r.feed(enc.encode('world\n'));
        r.end();
      }
    },
  });
  // Fed as the run starts, before its Worker opens it.
  results.echoEarly = await go(ECHO, {
    fed: true,
    start: (r) => {
      r.feed(enc.encode('early\n'));
      r.end();
    },
  });
  // Fed more at once than the engine's inbox holds, as the run idles: the page wakes the Worker to
  // take what fits, then leaves the rest.
  results.echoBig = await go(ECHO, {
    fed: true,
    onIdle: (r, n) => {
      if (n === 1) {
        r.feed(enc.encode(bigFeed()));
        r.end();
      }
    },
  });
  results.faults = await go(FAULTS);
  // #2579: JS capabilities answered inside each call, each task's Worker waiting inside it while the
  // page answers: the root's (interpreted, and run whole as a leaf), a thread's, and a detached
  // child's over a memory of its own.
  // #2645: the same, answered by the capabilities' module in each calling task's own Worker, and a
  // thread's call that waits for another thread's host calls, which go on while it waits.
  const caps3 = ['js.upper', 'js.add', 'js.waitFor'];
  for (const [at, answered] of [['', { onCap: jsCaps }], ['InWorker', { capModule: JS_CAPS_MODULE }]]) {
    const upperAdd = { js: ['js.upper', 'js.add'], ...answered };
    results[`jsCaps${at}`] = await go(JS_CAPS, upperAdd);
    results[`jsCapsLeaf${at}`] = await go(JS_CAPS, { ...upperAdd, tierup: true });
    results[`jsCapsThread${at}`] = await go(JS_CAPS_THREAD, upperAdd);
    results[`jsCapsChild${at}`] = await go(JS_CAPS_CHILD, { js: ['js.upper'], kids: [['kid', JS_CAPS_KID]], ...answered });
    results[`jsCapsWaits${at}`] = await go(JS_CAPS_WAITS, { js: caps3, ...answered });
  }
  // #1414 5c: the root as a leaf, which parks where the page answers its calls, and where it reads
  // stdin the page feeds as it idles.
  results.leafPings = await go(pingGuest(PINGS), { caps: ['ping'], tierup: true });
  results.leafEcho = await go(ECHO, {
    fed: true,
    tierup: true,
    onIdle: (r, n) => {
      if (n === 1) r.feed(enc.encode('hello '));
      if (n === 2) {
        r.feed(enc.encode('world\n'));
        r.end();
      }
    },
  });
  return results;
}

// How long an x2 case may run before the harness stops it and fails.
const X2_HANG_MS = 60_000;

// Input fed at once past what the engine's inbox holds (64 KiB), for `echoBig`: enough that the
// page's retries fill the inbox again after the Worker has taken the first of it.
export const bigFeed = () => Array.from({ length: 256 * 1024 }, (_, i) => String.fromCharCode(97 + (i % 26))).join('');
