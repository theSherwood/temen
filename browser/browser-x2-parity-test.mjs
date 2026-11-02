// Real-browser (V8) test for #1414 slice 5a: what a run on the parallel driver (executor 2) gives its
// page that a run on the pump does, on the threads build.
//
// - **Output as it is written.** Eight threads, each on its own Worker, write their digit ten times
//   to one stdout. `onOutput` hands it on as the Workers post it, in the order the run's buffer holds
//   it: the bytes handed on are the run's stdout exactly, and some arrive while the run still runs.
//   The pump's whole-program wasm-JIT run (`runJitModule`, the root leaf) and its interpreted
//   one-shot stream too (#2523: the root leaf's open streamed nothing).
// - **The root as a leaf (#1414 5c).** With its emitted tier, a run on executor 2 takes an on-ramp
//   program's root whole, as the pump's root leaf does: its entry is the run's one tier-up, and it
//   prints what the pump's prints.
// - **A stop that needs no reload.** A run spinning forever, and one asleep in a 30 s wait, each end
//   promptly on their `AbortSignal` (`stopped`, not `broken`), and the same engine then runs the
//   output program again. A finished run's signal, aborted during a later run, leaves that run be.
// - **The run's files and frame.** A program's `vm_fs` writes are in `temen_coop_fs_image` after the
//   run, as the pump leaves them, and a `display` guest's frame is the pump's.
//
// `outputInOrder`, which orders the Workers' chunks, is checked on chunks that arrive out of order and
// one that arrives after the run's end.
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
const ROOT = dirname(fileURLToPath(import.meta.url));

async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

// The I/O recipe hands the root its stdout handle. Thread i (packed into its arg above the handle)
// writes digit i ten times, a byte a call; the root joins all eight, then waits 200 ms, so the page
// hears the chunks before the run ends, and returns 80.
const OUTPUT_SRC = `memory 16
data 16384 "01234567"
func (i32) -> (i64) {
block 0 (vh: i32) {
  vhh = i64.extend_i32_u vh
  vz = i64.const 0
  br 1(vz, vhh)
}
block 1 (vi: i64, vh1: i64) {
  v8 = i64.const 8
  vmore = i64.lt_u vi v8
  br_if vmore 2(vi, vh1) 3()
}
block 2 (vi2: i64, vh2: i64) {
  vsh = i64.const 32
  vhi = i64.shl vi2 vsh
  varg = i64.or vhi vh2
  vsp = i64.const 0
  vt = thread.spawn 1 vsp varg
  v4 = i64.const 4
  voff = i64.mul vi2 v4
  vbase = i64.const 16400
  vslot = i64.add vbase voff
  i32.store vslot vt
  vone = i64.const 1
  vnext = i64.add vi2 vone
  br 1(vnext, vh2)
}
block 3 () {
  vz3 = i64.const 0
  br 4(vz3)
}
block 4 (vj: i64) {
  v8b = i64.const 8
  vmorej = i64.lt_u vj v8b
  br_if vmorej 5(vj) 6()
}
block 5 (vj2: i64) {
  v4b = i64.const 4
  voffj = i64.mul vj2 v4b
  vbasej = i64.const 16400
  vslotj = i64.add vbasej voffj
  vtj = i32.load vslotj
  vjoined_st, vjoined = thread.join vtj
  vonej = i64.const 1
  vnextj = i64.add vj2 vonej
  br 4(vnextj)
}
block 6 () {
  va = i64.const 16448
  vexp = i32.const 0
  vto = i64.const 200000000
  vst = i32.atomic.wait va vexp vto
  vr = i64.const 80
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vmask = i64.const 4294967295
  vh64 = i64.and varg vmask
  vh = i32.wrap_i64 vh64
  vsh = i64.const 32
  vidx = i64.shr_u varg vsh
  vbase = i64.const 16384
  vptr = i64.add vbase vidx
  vz = i64.const 0
  br 1(vz, vptr, vh)
}
block 1 (vk: i64, vp: i64, vhd: i32) {
  vten = i64.const 10
  vmore = i64.lt_u vk vten
  br_if vmore 2(vk, vp, vhd) 3()
}
block 2 (vk2: i64, vp2: i64, vhd2: i32) {
  vlen = i64.const 1
  vw = call.cap 0 1 (i64, i64) -> (i64) vhd2(vp2, vlen)
  vone = i64.const 1
  vk3 = i64.add vk2 vone
  br 1(vk3, vp2, vhd2)
}
block 3 () {
  vz3 = i64.const 0
  return vz3
  }
}
`;

// A root that never ends.
const SPIN_SRC = `memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  br 1(vi2)
  }
}
`;

// A root asleep in a 30 s futex wait nothing notifies.
const SLEEP_SRC = `memory 16
func () -> (i64) {
block 0 () {
  va = i64.const 16384
  vexp = i32.const 0
  vto = i64.const 30000000000
  vst = i32.atomic.wait va vexp vto
  vr = i64.extend_i32_u vst
  return vr
  }
}
`;

// An on-ramp program that writes "parallel\\n" to out.txt through \`vm_fs\` (open with
// write|create|truncate, write, close) and returns the bytes written.
const FS_SRC = `memory 16
import 0 "vm_fs" (i64, i64, i64, i64) -> (i64)
data 34816 "out.txt"
data 34832 "parallel\\n"
func () -> (i64) {
block 0 () {
  vop = i64.const 0
  vpath = i64.const 34816
  vplen = i64.const 7
  vflags = i64.const 26
  vfd = call.import 0 (vop, vpath, vplen, vflags)
  vopw = i64.const 2
  vbuf = i64.const 34832
  vn = i64.const 9
  vw = call.import 0 (vopw, vfd, vbuf, vn)
  vopc = i64.const 4
  vz = i64.const 0
  vc = call.import 0 (vopc, vfd, vz, vz)
  return vw
  }
}
export 0 func "_start" 0
`;

const res = await page.evaluate(async ({ outputSrc, spinSrc, sleepSrc, fsSrc }) => {
  const { loadEngine, makeRunner, readParStdoutBytes, outputInOrder } = await import('./par.js');
  const { runJitModule } = await import('./wasmjit-module.js');
  const { driveCoopTierupRun } = await import('./coop-driver.js');
  const eng = await loadEngine();
  const { ex, memory } = eng;
  const u8 = () => new Uint8Array(memory.buffer);
  const dec = (b) => new TextDecoder().decode(b);
  const parse = (src) => {
    const b = new TextEncoder().encode(src);
    const p = ex.temen_alloc(b.length);
    u8().set(b, p);
    const ok = ex.temen_parse(p, b.length);
    const out = u8().slice(ex.temen_parse_ptr(), ex.temen_parse_ptr() + ex.temen_parse_len());
    ex.temen_dealloc(p, b.length);
    if (ok !== 1) throw new Error('parse: ' + dec(out));
    return out;
  };
  const asset = async (name) => new Uint8Array(await (await fetch(`./assets/${name}.temen`)).arrayBuffer());
  const stdoutSlot = () => dec(u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len()));
  const frame = () => {
    const w = Number(ex.temen_framebuffer_width()), h = Number(ex.temen_framebuffer_height());
    const p = Number(ex.temen_framebuffer_ptr()), n = Number(ex.temen_framebuffer_len());
    return { w, h, px: Array.from(u8().slice(p, p + n)).join(',') };
  };
  const fsImage = () => {
    const n = Number(ex.temen_coop_fs_image());
    const p = Number(ex.temen_coop_fs_ptr());
    return u8().slice(p, p + n);
  };
  const run = makeRunner(eng);
  const out = {};

  // Chunks 1..5 of stdout ("a", "c", "e") and stderr ("B", "D") arrive as 3, 1, 5, 2; 4 is still on
  // its way when the run ends, and arrives after.
  {
    const got = [];
    const o = outputInOrder((a, b) => got.push(`${dec(a)}|${dec(b)}`));
    const enc = (s) => new TextEncoder().encode(s);
    for (const [seq, fd, s] of [[3, 1, 'c'], [1, 1, 'a'], [5, 1, 'e'], [2, 2, 'B']]) o.chunk(fd, seq, enc(s));
    o.end(enc('ace'), enc('BD'));
    o.chunk(2, 4, enc('D'));
    out.order = got;
  }

  // x2: output as it is written, in the run's order.
  const x2Output = async () => {
    let live = true, streamed = 0, text = '';
    const r = await run(parse(outputSrc), {
      io: true,
      onOutput: (o, e) => { text += dec(o) + dec(e); if (live) streamed++; },
    });
    live = false;
    return { value: r.value.toString(), text, final: dec(readParStdoutBytes(eng)), streamed };
  };
  out.output = await x2Output();

  // The pump: the root leaf (#2523) and the interpreted one-shot stream their stdout.
  const hello = await asset('hello_c');
  const sink = (chunks) => (fd, seq, bytes) => { if (fd === 1) chunks.push(dec(bytes)); };
  {
    const chunks = [];
    globalThis.__temen_stream_chunk = sink(chunks);
    const status = await runJitModule(ex, memory, hello, new Uint8Array(0));
    globalThis.__temen_stream_chunk = null;
    out.rootLeaf = { status, streamed: chunks.join(''), chunks: chunks.length, final: stdoutSlot() };
  }
  {
    const chunks = [];
    globalThis.__temen_stream_chunk = sink(chunks);
    const mp = Number(ex.temen_alloc(hello.length));
    u8().set(hello, mp);
    ex.temen_run_onramp(mp, hello.length, 0, 0);
    ex.temen_dealloc(mp, hello.length);
    globalThis.__temen_stream_chunk = null;
    out.oneShot = { streamed: chunks.join(''), chunks: chunks.length, final: stdoutSlot() };
  }

  // x2: the root as a leaf (#1414 5c). A run that fails says why, and fails only its own check.
  try {
    const r = await run(hello, { onramp: true, tierup: true });
    const leaf = ex.temen_x2_leaf_len(0) > 0;
    out.x2Leaf = { value: r.value.toString(), tierups: r.tierups, leaf, final: dec(readParStdoutBytes(eng)) };
  } catch (e) {
    out.x2Leaf = { err: e.message };
  }

  // x2: a stop ends a spinning run and a sleeping one, and the engine runs on.
  const stopped = async (src, ms) => {
    const ac = new AbortController();
    const t0 = performance.now();
    setTimeout(() => ac.abort(), ms);
    try {
      const r = await run(parse(src), { signal: ac.signal });
      return { ran: r.value.toString() };
    } catch (e) {
      return { err: e.message, broken: !!e.broken, ms: Math.round(performance.now() - t0) };
    }
  };
  out.stopSpin = await stopped(spinSrc, 100);
  out.stopSleep = await stopped(sleepSrc, 100);
  out.afterStop = await x2Output();
  // A finished run's signal reaches no later run: aborted while the next run waits out its end, it
  // leaves that run to end as it would.
  {
    const old = new AbortController();
    await run(parse(outputSrc), { io: true, signal: old.signal });
    const t = setTimeout(() => old.abort(), 50);
    try {
      const r = await run(parse(outputSrc), { io: true });
      out.lateAbort = { value: r.value.toString() };
    } catch (e) {
      out.lateAbort = { err: e.message };
    } finally {
      clearTimeout(t);
    }
  }

  // x2: the run's files and its frame, as the pump's.
  {
    const fs = parse(fsSrc);
    const r = await run(fs, { onramp: true });
    const x2 = fsImage();
    const COOP_NO_REGIONS = 4;
    const mp = Number(ex.temen_alloc(fs.length));
    u8().set(fs, mp);
    const opened = ex.temen_coop_open(mp, fs.length, 0, 0, 1, 0, 0, COOP_NO_REGIONS);
    ex.temen_dealloc(mp, fs.length);
    const status = opened === 0 ? await driveCoopTierupRun(ex, memory, { trapDeclines: false }) : null;
    const pump = fsImage();
    out.fs = {
      value: r.value.toString(), opened, status, x2Len: x2.length, pumpLen: pump.length,
      same: x2.length === pump.length && x2.every((b, i) => b === pump[i]),
      hasFile: dec(x2).includes('out.txt') && dec(x2).includes('parallel'),
    };
  }
  {
    const gradient = await asset('gradient');
    await run(gradient, { onramp: true });
    const x2 = frame();
    const mp = Number(ex.temen_alloc(gradient.length));
    u8().set(gradient, mp);
    ex.temen_run_onramp(mp, gradient.length, 0, 0);
    ex.temen_dealloc(mp, gradient.length);
    const pump = frame();
    out.frame = { w: x2.w, h: x2.h, same: x2.w === pump.w && x2.h === pump.h && x2.px === pump.px, pumpW: pump.w };
  }
  return out;
}, { outputSrc: OUTPUT_SRC, spinSrc: SPIN_SRC, sleepSrc: SLEEP_SRC, fsSrc: FS_SRC });

await browser.close();
await new Promise((r) => server.close(r));
console.log('RESULT', JSON.stringify(res));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));

const digits = (s) => [...'01234567'].every((d) => s.split(d).length - 1 === 10);
const outputOk = (o) => o && o.value === '80' && o.final.length === 80 && digits(o.final) && o.text === o.final && o.streamed > 0;
const checks = {
  'chunks are handed on in order, the rest at the end': JSON.stringify(res.order) === JSON.stringify(['a|', '|B', 'c|', 'e|D']),
  'x2 output streams in the run\'s order': outputOk(res.output),
  'the pump\'s root leaf streams (#2523)': res.rootLeaf.status === 0 && res.rootLeaf.chunks > 0 && res.rootLeaf.streamed === res.rootLeaf.final,
  'the pump\'s one-shot streams': res.oneShot.chunks > 0 && res.oneShot.streamed === res.oneShot.final,
  'x2 runs the root whole as a leaf, as the pump does': res.x2Leaf.leaf && res.x2Leaf.tierups === 1 && res.x2Leaf.final === res.rootLeaf.final,
  'a stop ends a spinning run': res.stopSpin.err === 'stopped' && !res.stopSpin.broken && res.stopSpin.ms < 3000,
  'a stop ends a sleeping run': res.stopSleep.err === 'stopped' && !res.stopSleep.broken && res.stopSleep.ms < 3000,
  'the engine runs on after a stop': outputOk(res.afterStop),
  'a finished run\'s signal reaches no later run': res.lateAbort.value === '80',
  'x2 leaves the run\'s files as the pump does': res.fs.value === '9' && res.fs.status === 0 && res.fs.hasFile && res.fs.same,
  'x2 reports the frame the pump does': res.frame.w > 0 && res.frame.h > 0 && res.frame.same,
};
for (const [name, ok] of Object.entries(checks)) console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}`);
const ok = errors.length === 0 && Object.values(checks).every(Boolean);
console.log(ok ? 'PASS — executor 2 streams its output, stops without a reload, runs its root whole as a leaf, and reports its files and frame as the pump does' : 'FAIL');
process.exit(ok ? 0 : 1);
