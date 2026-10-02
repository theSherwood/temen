// #1954 — the embedder cases `browser-coop-embedder-test.mjs` runs, in any host that has the engine:
// Node, and Chromium, whose JSPI suspends a root leaf's emitted frames where a call parks. Each case
// opens a program as a tier-up session (`temen_coop_open`) the way an embedder's Release button does
// — with declared host-completed caps, a root-leaf mode, and no regions — and runs it through
// `driveCoopTierupRun`'s embedder hooks: slices (`budget`/`onSlice`), streamed output (`onOutput`),
// answered cap parks (`onCapPark`) and reported traps (`trapDeclines: false`). `runCases` reports what
// each run did; the caller checks it.

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
  const open = (src, caps, mode) => {
    const text = enc.encode(src);
    const sp = put(text);
    const ok = ex.temen_parse(sp, text.length);
    ex.temen_dealloc(sp, text.length || 1);
    const bytes = u8().slice(Number(ex.temen_parse_ptr()), Number(ex.temen_parse_ptr()) + Number(ex.temen_parse_len()));
    if (ok !== 1) throw new Error(`parse: ${dec.decode(bytes)}`);
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
    const seen = { out: '', err: '', chunks: 0, slices: 0, parks: 0, counts: {} };
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
        await new Promise((r) => setTimeout(r, 0)); // answered asynchronously, as a UI would
        return pong(args[0]);
      },
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
  }
  if (open(TICKS, [], COOP_NO_REGIONS) !== 0) throw new Error('open ticks');
  results.ticks = await run();
  if (open(FAULTS, [], COOP_NO_REGIONS) !== 0) throw new Error('open faults');
  results.faults = { ...(await run({ trapDeclines: false })), trap: trapName(), addr: ex.temen_fault_addr() };
  if (open(FAULTS, [], COOP_NO_REGIONS) !== 0) throw new Error('open faults again');
  results.faultsDecline = await run().then(() => 'returned', (e) => String(e.message));
  return results;
}
