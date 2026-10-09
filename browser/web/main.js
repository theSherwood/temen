// THREADS/BROWSER step 4c-wasm in a REAL browser — the page-side orchestrator. Proves two things run
// in an actual browser (Chromium via Playwright), not just Node:
//   1. the **powerbox** (`temen_run_pb`) — a guest writes to stdout, single-threaded on the page;
//   2. genuine **parallelism** — one guest's `thread.spawn`ed vCPUs run on separate Web Workers over a
//      shared `WebAssembly.Memory` (only available because the server sent COOP/COEP), synchronising
//      via `Atomics` → the 8-vCPU counter kernel returns 4000.
// The Worker orchestration itself lives in `par.js` (shared with the playground, `play.js`); this
// page mirrors `threads-spawn.mjs` exactly.

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
    set('threads', 'fail', 'threads: skipped (no SharedArrayBuffer)');
    return;
  }

  let eng;
  try {
    eng = await loadEngine();
  } catch (e) {
    set('threads', 'fail', `threads: ${e.message}`);
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

  // Each run takes an engine of its own (`loadEngine(prev)`: a fresh memory over the compiled module)
  // and returns it, for reading the run's output back. A run can end with Workers still running in its
  // engine (a thread's trap ends a run while the root still runs), and the page terminates them.
  // Chromium stops a terminated Worker that is still running about two seconds later, wherever it is,
  // and one stopped inside the engine's allocator never releases the allocator's lock: every later
  // allocation in that memory spins forever, the page's own and the next run's.
  const run = async (guest, opts) => {
    const own = await loadEngine(eng);
    return { ...(await makeRunner(own)(guest, opts)), eng: own };
  };
  const runPath = async (guestPath, opts = {}) => {
    const o = { ...opts };
    if (o.unitPath) {
      o.unit = await fetchBytes(o.unitPath);
      delete o.unitPath;
    }
    return run(await fetchBytes(guestPath), o);
  };

  // --- 2) one guest's vCPUs across real Web Workers ----------------------------------------------
  try {
    const t0 = performance.now();
    const { value, started } = await runPath('/corpus/threads.temenc');
    const ms = (performance.now() - t0).toFixed(0);
    const ok = value === 4000n;
    set('threads', ok ? 'pass' : 'fail',
      `threads: ${started} Workers (1 root + ${started - 1} spawned) → ${value} (want 4000) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`threads → ${value} across ${started} Workers in ${ms}ms`);
  } catch (e) {
    set('threads', 'fail', `threads: error ${e}`);
  }

  // --- 3) §22 guest-JIT across real Web Workers (THREADS.md 4c-domain C2) -------------------------
  // Each worker vCPU `install`s a host-compiled unit into the **shared** Domain and `call_indirect`s
  // its own raced slot — `service(6,7) = 142`, folded to 8 × 142 = 1136. The powerbox is Rust-side
  // (a leaked `Host` in shared memory); JIT is serviced inside `temen_par_run`, so no new page glue.
  try {
    const t0 = performance.now();
    const { value, started } = await runPath('/corpus/threads_jit_install.temenc', { jit: true });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = value === 1136n;
    set('jit', ok ? 'pass' : 'fail',
      `jit: ${started} Workers each install+call a unit on the shared Domain → ${value} (want 1136) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`jit → ${value} across ${started} Workers in ${ms}ms`);
  } catch (e) {
    set('jit', 'fail', `jit: error ${e}`);
  }

  // --- 5) host I/O from worker vCPUs across real Web Workers (THREADS.md 4d) ----------------------
  // 8 worker vCPUs each `call.cap`-write "tick\n" to the run's ONE shared powerbox (a Mutex<Host> in
  // shared memory — dispatch is in-Rust under the lock, no JS in the loop) and bump a shared counter.
  // Result 8 and stdout "tick\n"×8 are schedule-independent; the page reads stdout back afterward.
  try {
    const t0 = performance.now();
    const { value, started, eng: ioEng } = await runPath('/corpus/threads_io.temenc', { io: true });
    const out = readParStdout(ioEng);
    // #152 — the same shared-powerbox model with the **on-ramp** grants: a manifest-`_start` guest
    // (`tests/fixtures/threads_onramp.temt`) whose 4 threads each `write` a letter through the one
    // bound `write` import, then the root writes the 8-byte total (8060) and `exit`s 7.
    const r = await runPath('/corpus/threads_onramp.temenc', { onramp: true });
    const ob = readParStdoutBytes(r.eng);
    // #1761 — a fiber created on the root Worker, resumed on a spawned one (one run-shared fiber
    // registry), reading `vcpu.tls` on each: 0 on the root, 1 on the thread → 13.
    const fb = await runPath('/corpus/threads_fibers.temenc', {});
    // DESIGN.md §12 / I37 — a spawned thread's trap ends the run (it used to hang: the trap only
    // reached a `thread.join` the root, polling forever, never makes).
    const ct = await runPath('/corpus/threads_child_trap.temenc', {}).then(() => 'no trap', (e) => e.message);
    const letters = new TextDecoder().decode(ob.slice(0, 4)).split('').sort().join('');
    const total = ob.length === 12 ? new DataView(ob.buffer).getBigInt64(4, true) : null;
    const ms = (performance.now() - t0).toFixed(0);
    const ok = value === 8n && out === 'tick\n'.repeat(8) &&
      r.exit === 7 && letters === 'abcd' && total === 8060n && fb.value === 13n &&
      ct.includes('DivByZero');
    set('capio', ok ? 'pass' : 'fail',
      `capio: ${started} Workers → counter ${value} (want 8), stdout ${JSON.stringify(out)} ` +
      `(want 8 × "tick\\n") · on-ramp ×${r.started} Workers → exit ${r.exit} (want 7), ` +
      `letters ${letters} (want abcd), total ${total} (want 8060) · migrated fiber → ${fb.value} ` +
      `(want 13) · thread trap → ${JSON.stringify(ct)} (want DivByZero) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`capio → ${value}, stdout ${out.length}B across ${started} Workers; on-ramp exit ${r.exit} in ${ms}ms`);
  } catch (e) {
    set('capio', 'fail', `capio: error ${e}`);
  }

  // --- #1414 B6) the parallel driver (executor 2) across real Web Workers -------------------------
  // The run is one in-Rust call on the root's Worker, and each of its threads a Worker the page
  // starts. First the `threads`, `jit` and `capio` guests, with the same answers. Then the M:N
  // guests: a fiber parked on a futex and woken by its own thread, resumed on another, woken by the
  // park-time recheck, timed out while the root sleeps on the page's clock, and polled past its
  // deadline. Then a deadlock and a live call to a serving child. The per-Worker driver hangs on the
  // first two fiber guests and the deadlock, traps `FiberFault` on the next two, and declines the
  // serving guest. The native suites run the same fixtures.
  try {
    const t0 = performance.now();
    const x2 = { x2: true };
    const th = await runPath('/corpus/threads.temenc', x2);
    const jt = await runPath('/corpus/threads_jit_install.temenc', { ...x2, jit: true });
    const io = await runPath('/corpus/threads_io.temenc', { ...x2, io: true });
    const out = readParStdout(io.eng);
    const r = await runPath('/corpus/threads_onramp.temenc', { ...x2, onramp: true });
    const ob = readParStdoutBytes(r.eng);
    const fb = await runPath('/corpus/threads_fibers.temenc', x2);
    const ct = await runPath('/corpus/threads_child_trap.temenc', x2).then(() => 'no trap', (e) => e.message);
    const letters = new TextDecoder().decode(ob.slice(0, 4)).split('').sort().join('');
    const total = ob.length === 12 ? new DataView(ob.buffer).getBigInt64(4, true) : null;
    const fibers = [];
    for (const [name, want] of [['fiber_park_futex', 331100n], ['fiber_park_then_migrate', 3110n],
      ['fiber_park_not_equal', 30101n], ['fiber_park_timed_wait', 30102n], ['fiber_park_poll_loop', 2n]]) {
      const got = await runPath(`/corpus/${name}.temenc`, x2).then((v) => v.value, (e) => e.message);
      fibers.push({ name, got, ok: got === want, want });
    }
    const dl = await runPath('/corpus/join_a_forever_waiter.temenc', x2).then(() => 'no trap', (e) => e.message);
    const sv = await runPath('/corpus/live_caller.temenc', { ...x2, inst: true, winSize: 1 << 17, minter: 1 << 20 });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = th.value === 4000n && jt.value === 1136n && io.value === 8n && out === 'tick\n'.repeat(8) &&
      r.exit === 7 && letters === 'abcd' && total === 8060n && fb.value === 13n && ct.includes('DivByZero') &&
      fibers.every((f) => f.ok) && dl.includes('ThreadFault') && sv.value === 142n;
    set('x2', ok ? 'pass' : 'fail',
      `x2: threads ${th.value} (want 4000) across ${th.started} Workers · jit ${jt.value} (want 1136) · ` +
      `io ${io.value} (want 8), stdout ${JSON.stringify(out)} · on-ramp exit ${r.exit} (want 7), letters ` +
      `${letters}, total ${total} (want 8060) · migrated fiber ${fb.value} (want 13) · thread trap ` +
      `${JSON.stringify(ct)} (want DivByZero) · ` +
      fibers.map((f) => `${f.name} ${f.got}${f.ok ? '' : ` (want ${f.want})`}`).join(' · ') +
      ` · deadlock ${JSON.stringify(dl)} (want ThreadFault) · served live call ${sv.value} (want 142) ` +
      `${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`x2 → threads ${th.value} across ${th.started} Workers in ${ms}ms`);
  } catch (e) {
    set('x2', 'fail', `x2: error ${e}`);
  }

  // --- 6) wasm-JIT tier: Temen IR compiled to wasm, run in-browser (BROWSER.md wasm-JIT slice 2/3c) --
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

  // --- 7) wasm-JIT tier-up **across real Web Workers** (BROWSER.md § "wasm-JIT tier", per-Worker JIT) --
  // The flagship: the 4000 counter kernel where each spawned worker's compute leaf tiers up onto
  // emitted wasm on *its own* Worker (`compile_module_tierup` emits the leaf even though it's reachable
  // only through `thread.spawn`), while the guest keeps running on the interpreter for spawn/join +
  // atomics. Run it BOTH ways over real Workers: plain (all-interp) and tier-up. Both must return 4000,
  // and the tier-up run must actually fire the seam (counter > 0) — a result match alone couldn't
  // distinguish tiered-up from silently-interpreted.
  try {
    const guest = await fetchBytes('/corpus/threads_tierup.temenc');
    const t0 = performance.now();
    const plain = await run(guest);
    const tiered = await run(guest, { tierup: true });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = plain.value === 4000n && tiered.value === 4000n && tiered.tierups > 0;
    set('tierup', ok ? 'pass' : 'fail',
      `tierup: ${tiered.started} Workers · plain → ${plain.value} · tier-up → ${tiered.value} ` +
      `(want 4000, ${tiered.tierups} regions ran on emitted wasm) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`tierup → plain ${plain.value} / tier-up ${tiered.value} with ${tiered.tierups} tier-ups across ${tiered.started} Workers in ${ms}ms`);
  } catch (e) {
    set('tierup', 'fail', `tierup: error ${e}`);
  }

  // --- 8) §22 guest-JIT **real codegen** across real Web Workers (BROWSER.md § "wasm-JIT tier", slice 5) --
  // The guest holds a `Jit` cap + a host-compiled unit and `call.cap`s invoke; each worker runs the
  // submitted unit on EMITTED WASM on its own Worker (`temen_par_powerbox_jit_codegen` emits it at setup)
  // instead of the interpreter. Runs the **same i32 kernel the interp `jit` item runs** (the unit
  // `service` is `(i32,i32)->(i32)` — the Worker marshals args by type, this slice's generalization),
  // so codegen → 1136 = interp; the counter proves the emitted unit ran. `install`/`call_indirect`
  // (Model B2) + guest-compiled units stay on the interp.
  try {
    const t0 = performance.now();
    // i32 unit sig — the same kernel the interp #jit item runs.
    const i32 = await run(await fetchBytes('/corpus/threads_jit_invoke.temenc'), { jitCodegen: true, jitService: 0 });
    // f64 unit sig — args' slot bits ↔ JS Numbers, result ↔ bits (the float ABI path).
    const f64 = await run(await fetchBytes('/corpus/threads_jit_invoke_f64.temenc'), { jitCodegen: true, jitService: 1 });
    const ms = (performance.now() - t0).toFixed(0);
    // Ground truth 1136 (= the interpreter, differential-checked in the Node twin threads-spawn.mjs);
    // here we also require the seam actually fired (units ran on wasm) for each sig.
    const ok = i32.value === 1136n && i32.tierups > 0 && f64.value === 1136n && f64.tierups > 0;
    set('jitcodegen', ok ? 'pass' : 'fail',
      `jitcodegen: ${i32.started} Workers each Jit.invoke a unit on emitted wasm · i32 → ${i32.value} ` +
      `(${i32.tierups} units) · f64 → ${f64.value} (${f64.tierups} units) (want 1136) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`jitcodegen → i32 ${i32.value}/${i32.tierups} · f64 ${f64.value}/${f64.tierups} across ${i32.started} Workers in ${ms}ms`);
  } catch (e) {
    set('jitcodegen', 'fail', `jitcodegen: error ${e}`);
  }

  // --- 9) §14 detached children **on emitted wasm** across real Web Workers (#1865) -----------------
  // The root spawns a granted unit 8× as DETACHED children (op-17 v1 records paid from its budget), each
  // on its own Worker in its own `WebAssembly.Memory`. With codegen, a child whose unit entry is in the
  // emitter subset runs it on EMITTED WASM bound to that memory — reading the "K"=75 its window was
  // seeded with → 8 × 75 = 600. Run it BOTH ways: interp (`inst`) and codegen (`instCodegen`); both 600,
  // and codegen must actually run children on wasm. The growth case: the unit `vm_map`s past its
  // declared 64 KiB in a helper the emitted entry bounces to its own vCPU, then stores+loads 4242 above
  // it — so it passes on the emitted tier only if the Worker re-reads the child's `"mapped"` after the
  // bounce.
  try {
    const guest = await fetchBytes('/corpus/threads_inst_detached.temenc');
    const unit = await fetchBytes('/corpus/threads_inst_unit.temenc');
    const opt = { unit, winSize: 1 << 20, minter: 8 * 65536 };
    const t0 = performance.now();
    const interp = await run(guest, { ...opt, inst: true });
    const codegen = await run(guest, { ...opt, instCodegen: true });
    const growGuest = await fetchBytes('/corpus/threads_inst_detached_one.temenc');
    const growUnit = await fetchBytes('/corpus/threads_inst_unit_grow.temenc');
    // The budget pays for the child's 64 KiB window and the 64 KiB it grows (#1909).
    const growOpt = { unit: growUnit, winSize: 1 << 20, minter: 2 * 65536 };
    const growInterp = await run(growGuest, { ...growOpt, inst: true });
    const growCodegen = await run(growGuest, { ...growOpt, instCodegen: true });
    const ms = (performance.now() - t0).toFixed(0);
    const growOk = growInterp.value === 4242n && growCodegen.value === 4242n && growCodegen.tierups > 0;
    const ok = interp.value === 600n && codegen.value === 600n && codegen.tierups > 0 && growOk;
    set('instcodegen', ok ? 'pass' : 'fail',
      `instcodegen: ${codegen.started} Workers · interp → ${interp.value} · codegen → ${codegen.value} ` +
      `(want 600, ${codegen.tierups} detached children ran on emitted wasm) · vm_map growth → interp ` +
      `${growInterp.value} / codegen ${growCodegen.value} (want 4242) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`instcodegen → interp ${interp.value} / codegen ${codegen.value} with ${codegen.tierups} emitted detached children; growth → ${growInterp.value}/${growCodegen.value} across ${codegen.started} Workers in ${ms}ms`);
  } catch (e) {
    set('instcodegen', 'fail', `instcodegen: error ${e}`);
  }

  // --- 10) §14 **VM-in-VM real codegen** (a detached child's spawn on emitted wasm, #1865) -----------
  // Same detached root, but the granted unit's entry SPAWNS: it reads its "K"=75, spawns a pure
  // grandchild (→ 9) of its own module DETACHED — an op-17 v1 record paid from its own `budget`, the node
  // that paid for its window (#1944) — joins it, returns 84 → 8 × 84 = 672. With codegen the spawning
  // entry itself runs on EMITTED WASM: its spawn and join arrive as `env.instantiate_rec`/`env.join`,
  // admitted and resolved by the child's own vCPU, each grandchild on its own Worker in its own
  // `Memory` (17 Workers). Both tiers must agree, and codegen must actually emit.
  try {
    const guest = await fetchBytes('/corpus/threads_inst_detached.temenc');
    const unit = await fetchBytes('/corpus/threads_inst_nested_detached_unit.temenc');
    const opt = { unit, winSize: 1 << 20, minter: 16 * 65536 };
    const t0 = performance.now();
    const interp = await run(guest, { ...opt, inst: true });
    const codegen = await run(guest, { ...opt, instCodegen: true });
    const ms = (performance.now() - t0).toFixed(0);
    // Non-vacuity: all 16 ran emitted — the 8 spawning children too, not just their pure grandchildren.
    const ok = interp.value === 672n && codegen.value === 672n && codegen.tierups === 16;
    set('instnested', ok ? 'pass' : 'fail',
      `instnested: ${codegen.started} Workers · interp → ${interp.value} · codegen → ${codegen.value} ` +
      `(want 672, ${codegen.tierups}/16 emitted incl. the spawning children) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`instnested → interp ${interp.value} / codegen ${codegen.value} with ${codegen.tierups} emitted across ${codegen.started} Workers in ${ms}ms`);
  } catch (e) {
    set('instnested', 'fail', `instnested: error ${e}`);
  }

  // --- 10b) §14 **page-op child real codegen** (#1151; detached, #1865) ------------------------------
  // Same detached root; the granted unit manages its own pages: its entry (emitted, PAGED) calls a helper
  // that `unmap`s one page of its window and `protect`s another read-only — an out-of-subset leaf the
  // Worker bounces onto the child's own vCPU (`temen_par_inst_call_interp`), re-syncing the page-state
  // table into the child's header after — then reads "K"=75 on the Ro page and stores+loads a marker on
  // an Rw page → 7509;
  // 8 × 7509 = 60072 on both tiers, codegen actually emitting. The trap twin stores on the UNMAPPED
  // page: the child faults on both tiers (the root's join propagates it → the run rejects), so the
  // emitted access provably honors the page state, not just the window bound.
  try {
    const guest = await fetchBytes('/corpus/threads_inst_detached.temenc');
    const unit = await fetchBytes('/corpus/threads_inst_paged_unit.temenc');
    const trapUnit = await fetchBytes('/corpus/threads_inst_paged_trap_unit.temenc');
    const opt = { winSize: 1 << 20, minter: 8 * 65536 };
    const t0 = performance.now();
    const interp = await run(guest, { ...opt, unit, inst: true });
    const codegen = await run(guest, { ...opt, unit, instCodegen: true });
    const outcome = (p) => p.then(() => 'done', () => 'trap');
    const trapInterp = await outcome(run(guest, { ...opt, unit: trapUnit, inst: true }));
    const trapCodegen = await outcome(run(guest, { ...opt, unit: trapUnit, instCodegen: true }));
    const ms = (performance.now() - t0).toFixed(0);
    const ok = interp.value === 60072n && codegen.value === 60072n && codegen.tierups > 0 &&
      trapInterp === 'trap' && trapCodegen === 'trap';
    set('instpaged', ok ? 'pass' : 'fail',
      `instpaged: ${codegen.started} Workers · interp → ${interp.value} · codegen → ${codegen.value} ` +
      `(want 60072, ${codegen.tierups} page-op children ran on emitted wasm) · store to unmapped page → ` +
      `interp ${trapInterp} / codegen ${trapCodegen} (want trap) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`instpaged → interp ${interp.value} / codegen ${codegen.value} with ${codegen.tierups} emitted page-op children; unmapped-page store → ${trapInterp}/${trapCodegen} in ${ms}ms`);
  } catch (e) {
    set('instpaged', 'fail', `instpaged: error ${e}`);
  }

  // --- 11) §22 **runtime-`Jit.compile` across Workers** (the multi-Worker runtime twin) -----------
  // 8 worker vCPUs each compile their OWN unit at runtime through the shared Mutex<Host> powerbox
  // (concurrent mutating compiles serialize on the lock) and `invoke` it (→ 7), folding an atomic
  // counter → 56. Interp vs codegen must agree; with codegen every invoke runs the unit's emitted
  // wasm per-Worker (tierups counts them). The browser twin of
  // bytecode_parallel_jit.rs::parallel_runtime_compile_invoke_is_sound.
  try {
    const guest = await fetchBytes('/corpus/jit_rt.temenc');
    const unit = await fetchBytes('/corpus/jit_rt_unit.temenc');
    const opt = { jitRuntime: true, jitBlobs: [{ off: 0x6000, bytes: unit }] };
    const t0 = performance.now();
    const interp = await run(guest, opt);
    const codegen = await run(guest, { ...opt, jitRuntimeCodegen: true });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = interp.value === 56n && codegen.value === 56n && codegen.tierups > 0;
    set('jitruntime', ok ? 'pass' : 'fail',
      `jitruntime: ${codegen.started} Workers · interp → ${interp.value} · codegen → ${codegen.value} ` +
      `(want 56, ${codegen.tierups} runtime units ran on emitted wasm) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`jitruntime → interp ${interp.value} / codegen ${codegen.value} with ${codegen.tierups} emitted across ${codegen.started} Workers in ${ms}ms`);
  } catch (e) {
    set('jitruntime', 'fail', `jitruntime: error ${e}`);
  }

  // --- 12) §22 **cross-Worker Model B2** (install → call_indirect through the table mirror) --------
  // 8 workers each runtime-compile a leaf (→ 7), `install` it into the shared dispatch table (raced
  // slots), runtime-compile a dispatcher whose body `call_indirect`s a slot, and `invoke` it passing
  // their slot. Interp: the dispatch runs over the shared interpreter Domain. Codegen (+B2): the
  // dispatcher runs on B2-emitted wasm — its call_indirect goes through THIS Worker's
  // WebAssembly.Table mirror of the shared slot→unit map (worker.js syncs before each invoke). Both
  // → 56. The browser twin of b2_install.rs + parallel_install_call_indirect_matches_oracle.
  try {
    const guest = await fetchBytes('/corpus/jit_b2.temenc');
    const leaf = await fetchBytes('/corpus/jit_rt_unit.temenc');
    const disp = await fetchBytes('/corpus/jit_b2_unit.temenc');
    const opt = { jitRuntime: true, jitBlobs: [{ off: 0x6000, bytes: leaf }, { off: 0x7000, bytes: disp }] };
    const t0 = performance.now();
    const interp = await run(guest, opt);
    const codegen = await run(guest, { ...opt, jitRuntimeCodegen: true, jitB2: true });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = interp.value === 56n && codegen.value === 56n && codegen.tierups > 0;
    set('jitb2', ok ? 'pass' : 'fail',
      `jitb2: ${codegen.started} Workers · interp → ${interp.value} · B2 codegen → ${codegen.value} ` +
      `(want 56, ${codegen.tierups} B2 dispatches on emitted wasm) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`jitb2 → interp ${interp.value} / B2 ${codegen.value} with ${codegen.tierups} emitted across ${codegen.started} Workers in ${ms}ms`);
  } catch (e) {
    set('jitb2', 'fail', `jitb2: error ${e}`);
  }

  // --- 13) §11 **threads inside a granted unit** (CONSOLIDATION.md §11 slice 3; detached, #1865) ------
  // The granted unit's entry thread.spawns its OWN f1 (→7), joins it, and does a mismatching
  // i32.atomic.wait (→1) → 71; 8 detached children → 568. Interp: module-aware spawn through the
  // relay (each unit thread a real Worker over the child's own Memory). Codegen: the entry runs on
  // EMITTED WASM, its thread/futex ops arriving as env.thread_spawn/join + env.mem_wait imports —
  // serviced through the same completion-slot protocol. Both tiers must agree.
  try {
    const guest = await fetchBytes('/corpus/threads_inst_detached.temenc');
    const unit = await fetchBytes('/corpus/threads_inst_threads_unit.temenc');
    const opt = { unit, winSize: 1 << 20, minter: 8 * 65536 };
    const t0 = performance.now();
    const interp = await run(guest, { ...opt, inst: true });
    const codegen = await run(guest, { ...opt, instCodegen: true });
    const ms = (performance.now() - t0).toFixed(0);
    const ok = interp.value === 568n && codegen.value === 568n && codegen.tierups > 0;
    set('instthreads', ok ? 'pass' : 'fail',
      `instthreads: ${codegen.started} Workers · interp → ${interp.value} · codegen → ${codegen.value} ` +
      `(want 568, ${codegen.tierups} emitted incl. threading units) ${ok ? 'PASS' : 'FAIL'} [${ms}ms]`);
    log(`instthreads → interp ${interp.value} / codegen ${codegen.value} with ${codegen.tierups} emitted across ${codegen.started} Workers in ${ms}ms`);
  } catch (e) {
    set('instthreads', 'fail', `instthreads: error ${e}`);
  }
}

main().catch((e) => { log(`fatal: ${e}\n${e.stack ?? ''}`); set('threads', 'fail', `fatal: ${e}`); });
