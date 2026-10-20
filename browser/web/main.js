// THREADS/BROWSER step 4c-wasm in a REAL browser — the page-side orchestrator. Proves two things run
// in an actual browser (Chromium via Playwright), not just Node:
//   1. the **powerbox** (`temen_run_pb`) — a guest writes to stdout, single-threaded on the page;
//   2. genuine **parallelism** — one guest's `thread.spawn`ed threads run on separate Web Workers over
//      a shared `WebAssembly.Memory` (only available because the server sent COOP/COEP), synchronising
//      via `Atomics` → the 8-thread counter kernel returns 4000.
// The Worker orchestration itself lives in `par.js` (shared with the playground, `play.js`).

import { fetchBytes, loadEngine, makeRunner, readParStdout, readParStdoutBytes } from '/web/par.js';
import { compileJit } from '/web/wasmjit.js';

const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (m) => { logEl.textContent += m + '\n'; };
const set = (id, status, text) => { const e = $(id); e.dataset.status = status; e.textContent = text; };

async function main() {
  set('isolated', String(self.crossOriginIsolated), `crossOriginIsolated: ${self.crossOriginIsolated}`);
  if (!self.crossOriginIsolated) {
    set('powerbox', 'fail', 'powerbox: skipped (no cross-origin isolation)');
    set('x2', 'fail', 'x2: skipped (no SharedArrayBuffer)');
    return;
  }

  let eng;
  try {
    eng = await loadEngine();
  } catch (e) {
    set('x2', 'fail', `x2: ${e.message}`);
    return;
  }
  const { memory, ex } = eng;
  const u8 = () => new Uint8Array(memory.buffer);
  log(`module loaded; shared=${memory.buffer instanceof SharedArrayBuffer}; TLS ${ex.__tls_size.value}B`);

  // --- 1) powerbox smoke (single-threaded, on the page) -------------------------------------------
  try {
    const pb = await fetchBytes('/corpus/pb_hello.temenc');
    const p = ex.temen_alloc(pb.length);
    u8().set(pb, p);
    ex.temen_run_pb(p, pb.length, 0, 0);
    const status = ex.temen_status();
    // `slice` (not `subarray`) copies out of the SharedArrayBuffer — TextDecoder rejects shared views.
    const out = new TextDecoder().decode(u8().slice(ex.temen_stdout_ptr(), ex.temen_stdout_ptr() + ex.temen_stdout_len()));
    const ok = status === 0 && out === 'hello, powerbox!\n';
    set('powerbox', ok ? 'pass' : 'fail', `powerbox: status=${status} stdout=${JSON.stringify(out)} ${ok ? 'PASS' : 'FAIL'}`);
    log(`powerbox → ${JSON.stringify(out)}`);
  } catch (e) {
    set('powerbox', 'fail', `powerbox: error ${e}`);
  }

  // Runs share an engine until one ends other than by its root returning: a trap, a failure or an
  // `exit`. Such a run can end with Workers still running in the engine (a thread's trap ends a run
  // while the root still runs), and the page terminates them. Chromium stops a terminated Worker that
  // is still running about two seconds later, wherever it is, and one stopped inside the engine's
  // allocator never releases the allocator's lock: every later allocation in that memory spins
  // forever. So the run after one takes a fresh engine (`loadEngine(prev)`: a new memory over the
  // compiled module). Not every run: each engine's memory holds a large address-space reservation
  // until the page collects it, and a page of quick runs runs out of them. A run returns its engine,
  // for reading its output back. `hotness` sets when calls tier up (#2474), which is the engine's,
  // so a run that sets it takes an engine of its own.
  let cur = eng;
  const run = async (guest, { hotness, ...opts } = {}) => {
    const own = hotness === undefined ? cur : await loadEngine(eng);
    if (hotness !== undefined) own.ex.temen_coop_set_hotness(...hotness);
    let returned = false;
    try {
      const r = await makeRunner(own)(guest, opts);
      returned = r.value !== null;
      return { ...r, eng: own };
    } finally {
      if (!returned && own === cur) cur = await loadEngine(eng);
    }
  };
  const runPath = async (guestPath, opts = {}) => {
    const o = { ...opts };
    if (o.unitPath) {
      o.unit = await fetchBytes(o.unitPath);
      delete o.unitPath;
    }
    return run(await fetchBytes(guestPath), o);
  };
  // The emitted tier: each thread runs its task's regions on its Worker. The guests' calls are
  // short, so these runs' engines tier up every eligible call (#2474), as coop's own tests do.
  const emit = { tierup: true, hotness: [0, 0] };

  // --- 2) one guest's threads across real Web Workers (executor 2, #1414 B6) ----------------------
  // The run is one in-Rust call on the root's Worker, and each of its threads a Worker the page
  // starts. First the 8-thread counter kernel (4000), and each thread `install`ing a host-compiled §22
  // unit into the shared Domain and `call_indirect`ing its own raced slot (8 × 142 = 1136). Then host
  // I/O (THREADS.md 4d): 8 threads each write "tick\n" through the run's ONE shared powerbox (a
  // Mutex<Host> in shared memory, dispatched in-Rust under the lock) and bump a shared counter → 8.
  // #152: the same with the **on-ramp** grants — a manifest-`_start` guest whose 4 threads each
  // `write` a letter through the one bound `write` import, then the root writes the 8-byte total
  // (8060) and `exit`s 7. #1761: a fiber created on the root's Worker and resumed on a spawned one,
  // reading `vcpu.tls` on each (→ 13). DESIGN.md §12 / I37: a thread's trap ends the run. Then the M:N
  // guests: a fiber parked on a futex and woken by its own thread, resumed on another, woken by the
  // park-time recheck, timed out while the root sleeps on the page's clock, and polled past its
  // deadline. Then a deadlock and a live call to a serving child. The native suites run the same
  // fixtures.
  try {
    const t0 = performance.now();
    const th = await runPath('/corpus/threads.temenc');
    const jt = await runPath('/corpus/threads_jit_install.temenc', { jit: true });
    const io = await runPath('/corpus/threads_io.temenc', { io: true });
    const out = readParStdout(io.eng);
    const r = await runPath('/corpus/threads_onramp.temenc', { onramp: true });
    const ob = readParStdoutBytes(r.eng);
    const fb = await runPath('/corpus/threads_fibers.temenc');
    const ct = await runPath('/corpus/threads_child_trap.temenc').then(() => 'no trap', (e) => e.message);
    const letters = new TextDecoder().decode(ob.slice(0, 4)).split('').sort().join('');
    const total = ob.length === 12 ? new DataView(ob.buffer).getBigInt64(4, true) : null;
    const fibers = [];
    for (const [name, want] of [['fiber_park_futex', 331100n], ['fiber_park_then_migrate', 3110n],
      ['fiber_park_not_equal', 30101n], ['fiber_park_timed_wait', 30102n], ['fiber_park_poll_loop', 2n]]) {
      const got = await runPath(`/corpus/${name}.temenc`).then((v) => v.value, (e) => e.message);
      fibers.push({ name, got, ok: got === want, want });
    }
    const dl = await runPath('/corpus/join_a_forever_waiter.temenc').then(() => 'no trap', (e) => e.message);
    // #2219: the live caller spawns func 1's child image, granted as the run's unit.
    const sv = await runPath('/corpus/live_caller.temenc', { inst: true, unitPath: '/corpus/live_caller_child.temenc', winSize: 1 << 17, minter: 1 << 20 });
    // #1414 B6-3: each thread runs its task's tier-ups on its own Worker — the `tierup` kernel's 8
    // workers each call the emitted leaf once. Then a root that returns while its thread counts
    // forever in an emitted leaf: the run's end stops the emitted code, so the run returns. The
    // regions are the coop driver's emit (B6-3b-4).
    const tu = await runPath('/corpus/threads_tierup.temenc', emit);
    const sp = await runPath('/corpus/root_leaves_a_spinner.temenc', emit);
    // #1414 ED-1: a thread whose entry is eligible runs it whole on its Worker's emitted tier, and
    // its results end the thread: each of the 8 counters' entries is one region.
    const te = await runPath('/corpus/threads_entry_tierup.temenc', emit);
    // #2443: a fork-join whose spawner runs emitted. Its spawns, its wait for the first thread's
    // done flag and its joins each stop its task from inside a bounce, as each thread's notify
    // does: the spawner's call and the 4 threads' entries are 5 regions.
    const fj = await runPath('/corpus/fork_join_in_a_call.temenc', emit);
    // #1414 B6-3b-4b: paged regions. An on-ramp guest protects a page read-only, and each of its 5
    // tasks reads it through the emitted leaf, checked against its thread's page-state table; the
    // twin's leaf stores there and traps. The run reports where, as it does interpreted (#2275).
    const paged = { onramp: true, ...emit };
    const pg = await runPath('/corpus/threads_paged.temenc', paged);
    const faulted = (e) => ({ why: e.message, at: e.faultAddr });
    const pgt = await runPath('/corpus/threads_paged_trap.temenc', paged).then(() => ({ why: 'no trap' }), faulted);
    const pgi = await runPath('/corpus/threads_paged_trap.temenc', { onramp: true }).then(() => ({ why: 'no trap' }), faulted);
    // #1414 B6-3b-4c: a collecting guest. Each of its 5 tasks' emitted leaf pushes the words it holds
    // across its call to the collector to its thread's spill stack, and the collector counts them.
    const gs = await runPath('/corpus/threads_gc_spill.temenc', emit);
    // #1414 B6-3b: §22 units on emitted wasm, each invoke on its own thread's Worker — the host's
    // unit (i32 and f64 signatures: the Worker marshals the args' slot bits to JS Numbers), units the
    // guest compiles itself through the shared Mutex<Host> powerbox (concurrent compiles serialize on
    // the lock), and a B2 dispatcher whose `call.dyn` lands in the unit its thread installed, through
    // its Worker's copy of the run's dispatch table.
    const jc32 = await runPath('/corpus/threads_jit_invoke.temenc', { jitCodegen: true, jitService: 0 });
    const jc64 = await runPath('/corpus/threads_jit_invoke_f64.temenc', { jitCodegen: true, jitService: 1 });
    const rtUnit = await fetchBytes('/corpus/jit_rt_unit.temenc');
    const rtBlobs = [{ off: 0x6000, bytes: rtUnit }];
    const jr = await runPath('/corpus/jit_rt.temenc', { jitRuntime: true, jitRuntimeCodegen: true, jitBlobs: rtBlobs });
    const b2Blobs = [...rtBlobs, { off: 0x7000, bytes: await fetchBytes('/corpus/jit_b2_unit.temenc') }];
    const jb = await runPath('/corpus/jit_b2.temenc', { jitRuntime: true, jitRuntimeCodegen: true, jitBlobs: b2Blobs });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = th.value === 4000n && jt.value === 1136n && io.value === 8n && out === 'tick\n'.repeat(8) &&
      r.exit === 7 && letters === 'abcd' && total === 8060n && fb.value === 13n && ct.includes('DivByZero') &&
      fibers.every((f) => f.ok) && dl.includes('ThreadFault') && sv.value === 142n &&
      tu.value === 4000n && tu.tierups === 8 && sp.value === 42n && sp.tierups === 1 &&
      te.value === 4000n && te.tierups === 8 && fj.value === 1001500n && fj.tierups === 5 &&
      pg.value === 35n && pg.tierups === 5 && pgt.why.includes('MemoryFault') && pgt.at === 65536n &&
      pgi.why.includes('MemoryFault') && pgi.at === 65536n &&
      gs.value === 55n && gs.tierups === 5 &&
      jc32.value === 1136n && jc32.tierups > 0 && jc64.value === 1136n && jc64.tierups > 0 &&
      jr.value === 56n && jr.tierups > 0 && jb.value === 56n && jb.tierups > 0;
    set('x2', ok ? 'pass' : 'fail',
      `x2: threads ${th.value} (want 4000) across ${th.started} Workers · jit ${jt.value} (want 1136) · ` +
      `io ${io.value} (want 8), stdout ${JSON.stringify(out)} · on-ramp exit ${r.exit} (want 7), letters ` +
      `${letters}, total ${total} (want 8060) · migrated fiber ${fb.value} (want 13) · thread trap ` +
      `${JSON.stringify(ct)} (want DivByZero) · ` +
      fibers.map((f) => `${f.name} ${f.got}${f.ok ? '' : ` (want ${f.want})`}`).join(' · ') +
      ` · deadlock ${JSON.stringify(dl)} (want ThreadFault) · served live call ${sv.value} (want 142) ` +
      `· tier-up ${tu.value} (want 4000), ${tu.tierups} regions on emitted wasm (want 8) · ` +
      `stopped spinner ${sp.value} (want 42), ${sp.tierups} region (want 1) · ` +
      `thread entries ${te.value} (want 4000), ${te.tierups} on emitted wasm (want 8) · ` +
      `fork-join ${fj.value} (want 1001500), ${fj.tierups} regions with spawns, waits and joins in their ` +
      `bounces (want 5) · ` +
      `paged ${pg.value} (want 35), ${pg.tierups} regions (want 5), its store ${JSON.stringify(pgt.why)} ` +
      `at ${pgt.at} (want MemoryFault at 65536), interpreted ${JSON.stringify(pgi.why)} at ${pgi.at} · ` +
      `collector over spilled words ${gs.value} (want 55), ${gs.tierups} regions (want 5) · ` +
      `§22 units on emitted wasm: host's i32 ${jc32.value}/${jc32.tierups}, f64 ${jc64.value}/${jc64.tierups} ` +
      `(want 1136), runtime ${jr.value}/${jr.tierups}, B2 ${jb.value}/${jb.tierups} (want 56) ` +
      `${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`x2 → threads ${th.value} across ${th.started} Workers in ${ms}ms`);
  } catch (e) {
    set('x2', 'fail', `x2: error ${e}`);
  }

  // --- 3) wasm-JIT tier: Temen IR compiled to wasm, run in-browser (BROWSER.md wasm-JIT slice 2/3c) --
  // The `alu` compute kernel is emitted to a wasm module by the cdylib (`temen_wasmjit_compile`),
  // instantiated against the page's OWN linear memory, and its `f0` called directly on the page
  // (compute-only → no Atomics.wait, so the main thread is fine). Assert it equals the `temen_run`
  // interpreter over an arg sweep, then time a heavy run to show the JIT's win over interp-in-wasm.
  // Then a **mixed-tier** guest (3c): a JITted integer caller whose SIMD leaf runs on the
  // interpreter via `env.call_interp` — same result as the whole-guest interpreter.
  const interpBytes = (bytes, arg) => {
    const p = ex.temen_alloc(bytes.length);
    u8().set(bytes, p);
    const r = ex.temen_run(p, bytes.length, BigInt(arg));
    const st = ex.temen_status();
    ex.temen_dealloc(p, bytes.length);
    if (st !== 0) throw new Error(`temen_run status ${st}`);
    return BigInt.asIntN(64, r);
  };
  // Compile Temen text → encoded module via the cdylib's front end (temen_parse), like the playground.
  const encode = (src) => {
    const s = new TextEncoder().encode(src);
    const p = ex.temen_alloc(s.length);
    u8().set(s, p);
    const ok = ex.temen_parse(p, s.length);
    ex.temen_dealloc(p, s.length);
    const optr = ex.temen_parse_ptr();
    const out = u8().slice(optr, optr + ex.temen_parse_len());
    if (ok !== 1) throw new Error(`parse: ${new TextDecoder().decode(out)}`);
    return out;
  };
  const MIXED_SRC = `
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v0, v1, v2)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.lt_s v5 v3
  br_if v6 2(v3, v4, v5) 3(v4)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = call 1 (v9)
  v11 = i64.add v8 v10
  v12 = i64.const 1
  v13 = i64.add v9 v12
  br 1(v7, v11, v13)
}
block 3 (v14: i64) {
  return v14
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64x2.splat v0
  v2 = i64x2.add v1 v1
  v3 = i64x2.extract_lane 0 v2
  return v3
  }
}`;
  // call.dyn (wasm-JIT next slice): a loop dispatches to func1 (double) / func2 (+100) by index
  // parity through the emitted funcref table — the same masked index + wasm signature check the
  // interpreter's identity table does. All-in-subset, so it JITs whole-module.
  const DISPATCH_SRC = `
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v0, v1, v2)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.lt_s v5 v3
  br_if v6 2(v3, v4, v5) 3(v4)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = i64.const 1
  v11 = i64.and v9 v10
  v12 = i64.const 1
  v13 = i64.add v11 v12
  v14 = i32.wrap_i64 v13
  v15 = call.dyn (i64) -> (i64) v14 (v9)
  v16 = i64.add v8 v15
  v17 = i64.const 1
  v18 = i64.add v9 v17
  br 1(v7, v16, v18)
}
block 3 (v19: i64) {
  return v19
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 2
  v2 = i64.mul v0 v1
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 100
  v2 = i64.add v0 v1
  return v2
  }
}`;
  // SIMD (wasm-JIT next slice): v128 lane arithmetic + a compare/bitmask reduction + a store→load
  // through the confined window — the emitted 0xFD opcodes and the one 16-byte widened access run
  // in-browser against the page's own memory, matching the interpreter. All-in-subset.
  const SIMD_SRC = `
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i32x4.splat v1
  v3 = i32.const 3
  v4 = i32x4.splat v3
  v5 = i32x4.add v2 v4
  v6 = i32x4.mul v5 v4
  v7 = i64.const 16416
  v128.store v7 v6
  v8 = v128.load v7
  v9 = i32x4.max_s v8 v2
  v10 = i32x4.gt_s v9 v4
  v11 = i8x16.bitmask v10
  v12 = i32x4.extract_lane 0 v9
  v13 = i32.add v12 v11
  v14 = i64.extend_i32_s v13
  return v14
  }
}`;
  try {
    const bytes = await fetchBytes('/corpus/alu.temenc');
    const jit = await compileJit(ex, bytes, { memory });
    if (!jit) throw new Error('temen_wasmjit_compile refused an in-subset module');
    let eq = true;
    for (const arg of [0n, 1n, 2n, 5n, 1000n, -1n, 100000n]) {
      if (jit.call([arg]).value !== interpBytes(bytes, arg)) { eq = false; break; }
    }
    const N = 5_000_000n;
    const t0 = performance.now();
    const jv = jit.call([N]).value;
    const t1 = performance.now();
    const iv = interpBytes(bytes, N);
    const t2 = performance.now();
    const jitMs = t1 - t0, intMs = t2 - t1;

    // Mixed-tier: the JITted caller sums a SIMD leaf run on the interpreter via env.call_interp.
    const mbytes = encode(MIXED_SRC);
    const mjit = await compileJit(ex, mbytes, { memory });
    let mixEq = mjit !== null;
    if (mjit) for (const arg of [0n, 1n, 2n, 5n, 20n, 100n]) {
      if (mjit.call([arg]).value !== interpBytes(mbytes, arg)) { mixEq = false; break; }
    }

    // call_indirect: the emitted funcref table + masked-index dispatch matches the interpreter.
    const dbytes = encode(DISPATCH_SRC);
    const djit = await compileJit(ex, dbytes, { memory });
    let ciEq = djit !== null;
    if (djit) for (const arg of [0n, 1n, 2n, 5n, 64n, 1000n, -1n]) {
      if (djit.call([arg]).value !== interpBytes(dbytes, arg)) { ciEq = false; break; }
    }

    // SIMD: emitted v128 opcodes + a confined 16-byte access match the interpreter in-browser.
    const sbytes = encode(SIMD_SRC);
    const sjit = await compileJit(ex, sbytes, { memory });
    let simdEq = sjit !== null;
    if (sjit) for (const arg of [0n, 1n, 2n, 5n, 64n, 1000n, -1n, 100000n]) {
      if (sjit.call([arg], { winSize: 1 << 16 }).value !== interpBytes(sbytes, arg)) { simdEq = false; break; }
    }

    const ok = eq && jv === iv && mixEq && ciEq && simdEq;
    set('wasmjit', ok ? 'pass' : 'fail',
      `wasmjit: alu f0 in-browser → ${jv} (interp ${iv}) ${eq && jv === iv ? 'ok' : 'FAIL'} · ` +
      `mixed-tier (JIT caller + interp SIMD leaf) ${mixEq ? 'ok' : 'FAIL'} · ` +
      `call_indirect dispatch ${ciEq ? 'ok' : 'FAIL'} · ` +
      `SIMD v128 ${simdEq ? 'ok' : 'FAIL'} · ` +
      `alu n=${N}: jit ${jitMs.toFixed(1)}ms vs interp ${intMs.toFixed(1)}ms → ${(intMs / jitMs).toFixed(1)}× ` +
      `${ok ? 'PASS' : 'FAIL'}`);
    log(`wasmjit → ${jv}, ${(intMs / jitMs).toFixed(1)}× over the interpreter; mixed-tier ${mixEq ? 'ok' : 'FAIL'}; call_indirect ${ciEq ? 'ok' : 'FAIL'}; SIMD ${simdEq ? 'ok' : 'FAIL'}`);
  } catch (e) {
    set('wasmjit', 'fail', `wasmjit: error ${e}`);
  }

  // --- 4) §14 detached children **on emitted wasm** across real Web Workers (#1865, #1414 ED-2) ----
  // The root spawns a granted unit 8× as DETACHED children (op-17 records paid from its budget), each
  // in its own `WebAssembly.Memory`, each child's thread on a Worker of its own. Under `tierup` a
  // child's root task runs its entry on its granted module's emit over that memory, reading the "K"=75
  // its window was seeded with → 8 × 75 = 600. Run it BOTH ways, interpreted and emitted: both 600, and
  // every child must actually run on emitted wasm. The growth case: the unit `vm_map`s past its
  // declared 64 KiB in a helper the emitted entry bounces to the interpreter, then stores+loads 4242
  // above it — so it passes on the emitted tier only if the Worker re-reads the child's `"mapped"`
  // after the bounce.
  try {
    const guest = await fetchBytes('/corpus/threads_inst_detached.temenc');
    const unit = await fetchBytes('/corpus/threads_inst_unit.temenc');
    const opt = { unit, winSize: 1 << 20, minter: 8 * 65536, inst: true };
    const t0 = performance.now();
    const interp = await run(guest, opt);
    const emitted = await run(guest, { ...opt, ...emit });
    const growGuest = await fetchBytes('/corpus/threads_inst_detached_one.temenc');
    const growUnit = await fetchBytes('/corpus/threads_inst_unit_grow.temenc');
    // The budget pays for the child's 64 KiB window and the 64 KiB it grows (#1909).
    const growOpt = { unit: growUnit, winSize: 1 << 20, minter: 2 * 65536, inst: true };
    const growInterp = await run(growGuest, growOpt);
    const growEmitted = await run(growGuest, { ...growOpt, ...emit });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = interp.value === 600n && emitted.value === 600n && emitted.tierups === 8 &&
      growInterp.value === 4242n && growEmitted.value === 4242n && growEmitted.tierups === 1;
    set('instcodegen', ok ? 'pass' : 'fail',
      `instcodegen: ${emitted.started} Workers · interp → ${interp.value} · emitted → ${emitted.value} ` +
      `(want 600, ${emitted.tierups}/8 detached children ran on emitted wasm) · vm_map growth → interp ` +
      `${growInterp.value} / emitted ${growEmitted.value} (want 4242, ${growEmitted.tierups}/1 emitted) ` +
      `${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`instcodegen → interp ${interp.value} / emitted ${emitted.value} with ${emitted.tierups} emitted detached children; growth → ${growInterp.value}/${growEmitted.value} across ${emitted.started} Workers in ${ms}ms`);
  } catch (e) {
    set('instcodegen', 'fail', `instcodegen: error ${e}`);
  }

  // --- 5) §14 **VM-in-VM** (a detached child's spawn, #1865) ---------------------------------------
  // The detached root, granting each child the unit itself as `"unit"`, and the granted unit's entry
  // SPAWNS: it reads its "K"=75, spawns `"unit"` DETACHED with a payload that makes it a pure grandchild
  // (→ 9) — an op-17 record paid from its own `budget`, the node that paid for its window (#1944) —
  // joins it, returns 84 → 8 × 84 = 672, each grandchild in its own `Memory` (17 Workers). Under
  // `tierup` the entry runs emitted in every child and grandchild, its spawn and join each stopping
  // its task from inside a bounce (#2443). Both runs must agree.
  try {
    const guest = await fetchBytes('/corpus/threads_inst_nested.temenc');
    const unit = await fetchBytes('/corpus/threads_inst_nested_detached_unit.temenc');
    const opt = { unit, winSize: 1 << 20, minter: 16 * 65536, inst: true };
    const t0 = performance.now();
    const interp = await run(guest, opt);
    const emitted = await run(guest, { ...opt, ...emit });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = interp.value === 672n && emitted.value === 672n && emitted.tierups === 16;
    set('instnested', ok ? 'pass' : 'fail',
      `instnested: ${interp.started} Workers · interp → ${interp.value} · under tierup → ${emitted.value} ` +
      `(want 672), ${emitted.tierups} entries on emitted wasm (want 16) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`instnested → interp ${interp.value} / under tierup ${emitted.value} across ${interp.started} Workers in ${ms}ms`);
  } catch (e) {
    set('instnested', 'fail', `instnested: error ${e}`);
  }

  // --- 6) §14 **page-op children on emitted wasm** (#1151; detached, #1865; #1414 ED-2) -------------
  // Same detached root; the granted unit manages its own pages: its entry (emitted, PAGED) calls a helper
  // that `unmap`s one page of its window and `protect`s another read-only — an out-of-subset leaf the
  // emitted code bounces to the interpreter, after which the Worker copies the page-state table into
  // the child's header again — then reads "K"=75 on the Ro page and stores+loads a marker on an Rw page
  // → 7509; 8 × 7509 = 60072 both ways, every child emitted. The trap twin stores on the UNMAPPED page:
  // the child faults both ways (the root's join answers the fault and the root traps on it, #2066 →
  // the run rejects), so the emitted access provably honors the page state, not just the window bound.
  try {
    const guest = await fetchBytes('/corpus/threads_inst_detached.temenc');
    const unit = await fetchBytes('/corpus/threads_inst_paged_unit.temenc');
    const trapUnit = await fetchBytes('/corpus/threads_inst_paged_trap_unit.temenc');
    const opt = { winSize: 1 << 20, minter: 8 * 65536, inst: true };
    const t0 = performance.now();
    const interp = await run(guest, { ...opt, unit });
    const emitted = await run(guest, { ...opt, unit, ...emit });
    const outcome = (p) => p.then(() => 'done', () => 'trap');
    const trapInterp = await outcome(run(guest, { ...opt, unit: trapUnit }));
    const trapEmitted = await outcome(run(guest, { ...opt, unit: trapUnit, ...emit }));
    const ms = (performance.now() - t0).toFixed(0);
    const ok = interp.value === 60072n && emitted.value === 60072n && emitted.tierups === 8 &&
      trapInterp === 'trap' && trapEmitted === 'trap';
    set('instpaged', ok ? 'pass' : 'fail',
      `instpaged: ${emitted.started} Workers · interp → ${interp.value} · emitted → ${emitted.value} ` +
      `(want 60072, ${emitted.tierups}/8 page-op children ran on emitted wasm) · store to unmapped page → ` +
      `interp ${trapInterp} / emitted ${trapEmitted} (want trap) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`instpaged → interp ${interp.value} / emitted ${emitted.value} with ${emitted.tierups} emitted page-op children; unmapped-page store → ${trapInterp}/${trapEmitted} in ${ms}ms`);
  } catch (e) {
    set('instpaged', 'fail', `instpaged: error ${e}`);
  }

  // --- 7) §11 **threads inside a granted unit** (CONSOLIDATION.md §11 slice 3; detached, #1865) -------
  // The granted unit's entry thread.spawns its OWN f1 (→7), joins it, and does a mismatching
  // i32.atomic.wait (→1) → 71; 8 detached children → 568, each unit thread on a Worker of its own over
  // its child's Memory. Under `tierup` the entry runs emitted, its spawn, join and wait each stopping
  // its task from inside a bounce (#2443). Its threads interpret: a child's memory holds one context,
  // its root task's. Both runs must agree.
  try {
    const guest = await fetchBytes('/corpus/threads_inst_detached.temenc');
    const unit = await fetchBytes('/corpus/threads_inst_threads_unit.temenc');
    const opt = { unit, winSize: 1 << 20, minter: 8 * 65536, inst: true };
    const t0 = performance.now();
    const interp = await run(guest, opt);
    const emitted = await run(guest, { ...opt, ...emit });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = interp.value === 568n && emitted.value === 568n && emitted.tierups === 8;
    set('instthreads', ok ? 'pass' : 'fail',
      `instthreads: ${interp.started} Workers · interp → ${interp.value} · under tierup → ${emitted.value} ` +
      `(want 568), ${emitted.tierups} entries on emitted wasm (want 8) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`instthreads → interp ${interp.value} / under tierup ${emitted.value} across ${interp.started} Workers in ${ms}ms`);
  } catch (e) {
    set('instthreads', 'fail', `instthreads: error ${e}`);
  }
}

main().catch((e) => { log(`fatal: ${e}\n${e.stack ?? ''}`); set('x2', 'fail', `fatal: ${e}`); });
