// THREADS.md step 4c-wasm — the real thing: **one** guest's `thread.spawn`ed vCPUs run on **separate
// Web Workers** (here Node worker_threads, the same SharedArrayBuffer + Atomics primitives a browser
// uses) over the **one** shared linear-memory window. Each Worker runs one vCPU through the engine's
// resumable `Vcpu` API (temen_par_run → an event → the host services it → deliver → run again); the host
// services the events with genuine cross-Worker primitives:
//   * thread.spawn  → ask main to start a new Worker for the child vCPU;
//   * thread.join   → Atomics.wait on the child's completion slot, then read its result;
//   * memory.wait   → Atomics.wait on the futex word in the window;
//   * memory.notify → Atomics.notify on it.
// So this is genuinely parallel (N vCPUs, N OS threads, one shared memory). The native
// `bytecode_vcpu_orchestration.rs` test is its differential oracle. Main creates every Worker (no
// nested Workers); each vCPU runs on a Worker (never the main thread, which can't Atomics.wait).
//
// Usage:  node threads-spawn.mjs <module.wasm> [guest.temenc] [expected]
import { readFileSync } from 'node:fs';
import { Worker, isMainThread, workerData, parentPort } from 'node:worker_threads';
import { engineImports } from './engine-imports.mjs';

const WASM = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/temen_browser.wasm';
const GUEST = process.argv[3] ?? 'corpus/threads.temenc';
const EXPECT = BigInt(process.argv[4] ?? 4000);

const STACK = 1 << 20; // per-Worker stack
const SLOT = 16; // completion slot: [done:i32 @0][result:i64 @8]
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);

// Event codes (must match browser/src/lib.rs PAR_*).
const DONE = 0, TRAP = 1, SPAWN = 2, JOIN = 3, WAIT = 4, NOTIFY = 5, INSTANTIATE = 6, TIERUP = 7, JIT_INVOKE = 8;

// §22 codegen arg/result marshalling by scalar type code (0=i32, 1=i64, 2=f32, 3=f64) — see worker.js.
const _sdv = new DataView(new ArrayBuffer(8));
const jitArg = (slot, tc) => tc === 0 ? Number(BigInt.asIntN(32, slot))
  : tc === 1 ? slot
  : tc === 2 ? (_sdv.setInt32(0, Number(BigInt.asIntN(32, slot)), true), _sdv.getFloat32(0, true))
  : (_sdv.setBigInt64(0, slot, true), _sdv.getFloat64(0, true));
const jitRes = (ret, tc) => tc === 0 ? BigInt(ret)
  : tc === 1 ? ret
  : tc === 2 ? (_sdv.setFloat32(0, ret, true), BigInt(_sdv.getUint32(0, true)))
  : (_sdv.setFloat64(0, ret, true), _sdv.getBigInt64(0, true));

// ---- a single vCPU on this Worker ---------------------------------------------------------------
async function worker() {
  const { module, memory, prog, win, winSize, role, func, sp, arg, slot, stackTop, tlsBase,
    smod, entry, slog, fuel, tierup, tierupPaged, gptr, glen, tierupCell, jitCodegen, instCodegen, jitService } = workerData;
  const { exports: ex } = await WebAssembly.instantiate(module, engineImports(memory));
  ex.__stack_pointer.value = stackTop; // this Worker's private stack...
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(tlsBase); // ...and TLS block (per 4b)
  // Views over the shared memory, refreshed when stale: a shared `WebAssembly.Memory` can GROW
  // mid-run (any Worker's in-wasm allocation — e.g. a §14 module compile+push), and views created
  // before a growth don't cover the new region (an Atomics access past the old length throws).
  let i32v = new Int32Array(memory.buffer), i64v = new BigInt64Array(memory.buffer);
  const i32 = () =>
    i32v.byteLength === memory.buffer.byteLength ? i32v : (i32v = new Int32Array(memory.buffer));
  const i64 = () =>
    i64v.byteLength === memory.buffer.byteLength ? i64v : (i64v = new BigInt64Array(memory.buffer));
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  // wasm-JIT tier-up (per-Worker JIT): this Worker enables the tier-up bitmap for its instance
  // (`temen_par_enable_jit` emits the tier-up module — a pure leaf reachable only via `thread.spawn`
  // still emits) and instantiates the emitted module against the ONE shared memory. Each Worker
  // instantiates its own (wasm tables aren't shareable across Workers). On TIERUP it runs `f{func}`.
  let emitted = null, envCell = 0;
  // #750: `tierupPaged` (TEMEN_TIERUP_PAGED=1) opts the run into the paged tier, exactly as worker.js.
  const enableJit = tierupPaged ? ex.temen_par_enable_jit_paged : ex.temen_par_enable_jit;
  if (tierup && enableJit(gptr, glen) === 1) {
    const wptr = Number(ex.temen_wasmjit_ptr()), wlen = ex.temen_wasmjit_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const emod = await WebAssembly.instantiate(await WebAssembly.compile(bytes), {
      env: {
        memory,
        trap: () => {}, // Temen fault; the following `unreachable` throws, caught below as a vCPU trap
        call_interp: (f, a) => { if (ex.temen_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    emitted = emod.exports;
    envCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes()));
  }

  // §22 guest-JIT real codegen: the run's single §22 unit was emitted + stashed once at powerbox
  // setup; each Worker instantiates its own instance and runs `f0(win, env, args)` on JIT_INVOKE.
  let jitUnit = null, jitEnvCell = 0;
  if (jitCodegen) ex.temen_par_jit_codegen_service(jitService | 0); // 0=i32, 1=f64 service (per-instance)
  if (jitCodegen && ex.temen_par_enable_jit_codegen() === 1 && ex.temen_par_jit_unit_wasm_len() > 0) {
    const wptr = Number(ex.temen_par_jit_unit_wasm_ptr()), wlen = ex.temen_par_jit_unit_wasm_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const uinst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
      env: {
        memory,
        trap: () => {},
        call_interp: (f, a) => { if (ex.temen_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    jitUnit = uinst.exports;
    jitEnvCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes()));
  }

  // §14 instantiate real codegen: a confined child whose granted-unit entry is eligible runs it on
  // emitted wasm here and fills the completion slot the parent joins (no vCPU). With the nested emit
  // a cap-using entry is ALSO eligible — its instantiate/join arrive as the env.instantiate/env.join
  // imports, serviced through the same completion-slot protocol (see web/worker.js, the browser twin).
  if (role === 'confined' && instCodegen && ex.temen_par_enable_inst_codegen() === 1
      && ex.temen_par_inst_eligible(entry) === 1) {
    const wptr = Number(ex.temen_par_inst_unit_wasm_ptr()), wlen = ex.temen_par_inst_unit_wasm_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const childSlots = []; // env.instantiate handle (index) → grandchild completion slot ptr
    const threadSlots = []; // env.thread_spawn handle (index) → thread completion slot ptr
    const uinst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
      env: {
        memory,
        trap: () => {},
        call_interp: (f, a) => { if (ex.temen_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); },
        instantiate: (cwin, _inst, centry, off, cslog, quota) => {
          const gsize = 1 << Number(cslog), goff = Number(off);
          if (gsize > winSize || (goff & (gsize - 1)) !== 0 || goff + gsize > winSize)
            throw new Error('bad nested carve');
          const gslot = ex.temen_par_alloc(SLOT);
          const gstackTop = ex.temen_par_alloc(STACK) + STACK;
          const gtlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
          const pf = BigInt(fuel);
          const gfuel = quota > 0n && quota < pf ? quota : pf;
          parentPort.postMessage({
            kind: 'spawn', role: 'confined', smod, entry: Number(centry), slog: Number(cslog),
            fuel: gfuel.toString(), win: cwin + goff, winSize: gsize,
            slot: gslot, stackTop: gstackTop, tlsBase: gtlsBase,
          });
          const h = childSlots.length;
          childSlots.push(gslot);
          return h;
        },
        join: (_inst, child) => {
          const gslot = childSlots[child];
          if (gslot === undefined) throw new Error('join of unknown child');
          Atomics.wait(i32(), gslot >> 2, 0);
          if (Atomics.load(i32(), gslot >> 2) === 2) throw new Error('nested child trapped');
          return i64()[(gslot + 8) >> 3];
        },
        // §11 slice 3 — thread/futex ops from an EMITTED unit, serviced through the same spawn
        // relay + completion-slot protocol as the interpreter's SPAWN/JOIN arms. The spawned vCPU
        // runs the granted unit's own `func` (smod — this Worker knows its module), over THIS
        // child's window (a thread shares its spawner's window = the carve).
        thread_spawn: (func, sp, arg) => {
          const tslot = ex.temen_par_alloc(SLOT);
          const tstackTop = ex.temen_par_alloc(STACK) + STACK;
          const ttlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
          parentPort.postMessage({
            kind: 'spawn', smod, func, sp: sp.toString(), arg: arg.toString(),
            win, winSize, fuel,
            slot: tslot, stackTop: tstackTop, tlsBase: ttlsBase,
          });
          const h = threadSlots.length;
          threadSlots.push(tslot);
          return h;
        },
        thread_join: (h) => {
          const tslot = threadSlots[h];
          if (tslot === undefined) throw new Error('join of unknown thread');
          Atomics.wait(i32(), tslot >> 2, 0);
          if (Atomics.load(i32(), tslot >> 2) === 2) throw new Error('unit thread trapped');
          return i64()[(tslot + 8) >> 3];
        },
        // Futex over the shared window (addr confined by the window mask, as the engine does).
        mem_wait: (cwin, addr, expected, timeout, is64) => {
          const a = cwin + (Number(addr) & (winSize - 1));
          const ms = timeout <= 0n ? Infinity : Number(timeout) / 1e6;
          const r = is64
            ? Atomics.wait(i64(), a >> 3, expected, ms)
            : Atomics.wait(i32(), a >> 2, Number(BigInt.asIntN(32, expected)), ms);
          return r === 'ok' ? 0 : r === 'not-equal' ? 1 : 2;
        },
        mem_notify: (cwin, addr, count) => {
          const a = cwin + (Number(addr) & (winSize - 1));
          return Atomics.notify(i32(), a >> 2, count >>> 0);
        },
      },
    });
    const envCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes()));
    uinst.exports.fuel.value = 1n << 61n; // fuel now lives in the emitted `fuel` global
    const args = new Array(Number(ex.temen_par_inst_nparams(entry))).fill(0n);
    if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1);
    try {
      const ret = uinst.exports['f' + entry](win, envCell, ...args);
      i64()[(slot + 8) >> 3] = BigInt(ret);
      Atomics.store(i32(), slot >> 2, 1);
      Atomics.notify(i32(), slot >> 2);
    } catch {
      Atomics.store(i32(), slot >> 2, 2);
      Atomics.notify(i32(), slot >> 2);
    }
    return;
  }

  // A §14 'confined' child's `win`/`winSize` are already its carve (the parent's window + the event's
  // offset) — a confined child is just a child with a shifted, smaller window (DESIGN.md §14).
  const v = role === 'root'
    ? ex.temen_par_root(prog, win, winSize, func)
    : role === 'confined'
      ? ex.temen_par_child_confined(prog, win, slog, smod, entry, BigInt(fuel))
      : ex.temen_par_child(prog, win, winSize, smod | 0, func, BigInt(sp), BigInt(arg));
  if (v === 0) { parentPort.postMessage({ kind: 'fail', why: 'vcpu build failed' }); return; }

  const handles = []; // local spawn handle (index) → child completion slot ptr

  for (;;) {
    const ev = ex.temen_par_run(v);
    if (ev === DONE) {
      const value = ex.temen_par_ev_a(v); // i64 → BigInt
      i64()[(slot + 8) >> 3] = value; // publish result...
      Atomics.store(i32(), slot >> 2, 1); // ...set done flag...
      Atomics.notify(i32(), slot >> 2); // ...and wake a joiner
      if (role === 'root') parentPort.postMessage({ kind: 'done', value: value.toString() });
      ex.temen_par_free(v);
      return;
    }
    if (ev === TRAP) {
      Atomics.store(i32(), slot >> 2, 2); // 2 = trapped
      Atomics.notify(i32(), slot >> 2);
      if (role === 'root') parentPort.postMessage({ kind: 'trap' });
      ex.temen_par_free(v);
      return;
    }
    if (ev === SPAWN) {
      // ev_a packs (spawning frame's module << 32) | func, as the INSTANTIATE event does — the
      // child resolves `func` in that module (an installed §22 unit spawns its own functions).
      const cam = ex.temen_par_ev_a(v);
      const csmod = Number(cam >> 32n), cfunc = Number(BigInt.asUintN(32, cam));
      const csp = ex.temen_par_ev_b(v), carg = ex.temen_par_ev_c(v);
      // Allocate the child's completion slot + stack + TLS (shared, thread-safe allocator), then ask
      // main to start a Worker for it. We continue immediately with the handle (the child runs async).
      const cslot = ex.temen_par_alloc(SLOT);
      const cstackTop = ex.temen_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      parentPort.postMessage({
        kind: 'spawn', smod: csmod, func: cfunc, sp: csp.toString(), arg: carg.toString(),
        win, winSize,
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      const handle = handles.length;
      handles.push(cslot);
      ex.temen_par_deliver_handle(v, handle);
      continue;
    }
    if (ev === JOIN) {
      const handle = Number(ex.temen_par_ev_a(v));
      const cslot = handles[handle];
      if (cslot === undefined) { ex.temen_par_deliver_join(v, 0n, 1); continue; } // bad handle -> trap, never wait(0)
      Atomics.wait(i32(), cslot >> 2, 0); // block until the child sets its done flag
      const trapped = Atomics.load(i32(), cslot >> 2) === 2;
      const result = i64()[(cslot + 8) >> 3];
      ex.temen_par_deliver_join(v, result, trapped ? 1 : 0);
      continue;
    }
    if (ev === INSTANTIATE) {
      // §14 confined executor child: the engine already validated the carve + built everything
      // authority-bearing; the operands are inert integers we shuttle into a new Worker (whose
      // window IS the carve), joined via the same completion-slot protocol as SPAWN.
      const am = ex.temen_par_ev_a(v); // (module << 32) | entry
      const csmod = Number(am >> 32n), centry = Number(BigInt.asUintN(32, am));
      const carve = Number(ex.temen_par_ev_b(v)), cslog = Number(ex.temen_par_ev_c(v));
      const cfuel = ex.temen_par_ev_d(v); // i64 → BigInt, shuttled verbatim
      const cslot = ex.temen_par_alloc(SLOT);
      const cstackTop = ex.temen_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      parentPort.postMessage({
        kind: 'spawn', role: 'confined', smod: csmod, entry: centry, slog: cslog,
        fuel: cfuel.toString(), win: win + carve, winSize: 1 << cslog,
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      const handle = handles.length;
      handles.push(cslot);
      ex.temen_par_deliver_handle(v, handle);
      continue;
    }
    if (ev === WAIT) {
      const addr = Number(ex.temen_par_ev_a(v)), expected = Number(BigInt.asIntN(32, ex.temen_par_ev_b(v)));
      const timeoutNs = ex.temen_par_ev_d(v);
      const idx = (win + addr) >> 2;
      const ms = timeoutNs <= 0n ? Infinity : Number(timeoutNs) / 1e6;
      const r = Atomics.wait(i32(), idx, expected, ms); // 'ok' | 'not-equal' | 'timed-out'
      ex.temen_par_deliver_code(v, r === 'ok' ? 0 : r === 'not-equal' ? 1 : 2);
      continue;
    }
    if (ev === NOTIFY) {
      const addr = Number(ex.temen_par_ev_a(v)), count = Number(ex.temen_par_ev_b(v));
      const woke = Atomics.notify(i32(), (win + addr) >> 2, count);
      ex.temen_par_deliver_code(v, woke);
      continue;
    }
    if (ev === TIERUP) {
      // Run the emitted `f{func}(win, env, ...i64 args)` on this Worker instead of interpreting. A
      // trap throws (Temen fault → env.trap + unreachable, or a wasm trap) → surface as a vCPU trap.
      const tfunc = Number(ex.temen_par_ev_a(v));
      const argvPtr = Number(ex.temen_par_tierup_argv_ptr(v)), n = Number(ex.temen_par_tierup_argv_len(v));
      const args = [];
      for (let i = 0; i < n; i++) args.push(i64()[(argvPtr >> 3) + i]);
      emitted.fuel.value = 1n << 61n; // ample fuel (emitted `fuel` global)
      // #717 host sync: the event's committed-extent snapshot → the emitted `"mapped"` global
      // (operand b; on a #750 paged run it is the page-state table's coverage). This driver
      // predates the contract and ran unsynced — idempotent for fully-mapped guests, wrong the
      // day one grows; now it follows worker.js line-for-line.
      emitted.mapped.value = ex.temen_par_ev_b(v);
      // #750 paged runs: the emitted `"pagestate"` global ← the engine-built table's address
      // (one shared memory, zero copies). Empty (and the global absent) on unpaged runs.
      if (Number(ex.temen_par_tierup_pagestate_len(v)) > 0)
        emitted.pagestate.value = Number(ex.temen_par_tierup_pagestate_ptr(v));
      if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count tier-ups (non-vacuity)
      try {
        const ret = emitted['f' + tfunc](win, envCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rptr = Number(ex.temen_par_alloc(Math.max(1, rets.length) * 8));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = BigInt(rets[i]);
        ex.temen_par_deliver_tierup(v, rptr, rets.length);
      } catch {
        ex.temen_par_deliver_tierup_trap(v);
      }
      continue;
    }
    if (ev === JIT_INVOKE) {
      // §22 guest-JIT real codegen: run the emitted unit's `f0(win, env, ...args)` instead of the
      // interpreter, then deliver its result slots. Args marshal by type (i32 → Number, i64 → BigInt).
      const argvPtr = Number(ex.temen_par_jit_argv_ptr(v)), n = Number(ex.temen_par_jit_argv_len(v));
      const ptypes = new Uint8Array(memory.buffer, Number(ex.temen_par_jit_param_types_ptr(v)), n);
      const args = [];
      for (let i = 0; i < n; i++) args.push(jitArg(i64()[(argvPtr >> 3) + i], ptypes[i]));
      jitUnit.fuel.value = 1n << 61n; // ample fuel (emitted `fuel` global)
      if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // reuse the counter (non-vacuity)
      try {
        const ret = jitUnit['f0'](win, jitEnvCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rn = Number(ex.temen_par_jit_result_types_len(v));
        const rtypes = new Uint8Array(memory.buffer, Number(ex.temen_par_jit_result_types_ptr(v)), rn);
        const rptr = Number(ex.temen_par_alloc(Math.max(1, rets.length) * 8));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = jitRes(rets[i], rtypes[i]);
        ex.temen_par_deliver_jit_invoke(v, rptr, rets.length);
      } catch {
        ex.temen_par_deliver_jit_invoke_trap(v);
      }
      continue;
    }
  }
}

// ---- main: compile, carve the window, start the root Worker, fan out child Workers on request ----
async function main() {
  const module = await WebAssembly.compile(readFileSync(WASM));
  if (!WebAssembly.Module.imports(module).some((i) => i.kind === 'memory')) {
    console.log('FAIL: not a threads build (module does not import a shared memory)');
    process.exit(1);
  }
  const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, engineImports(memory));
  const u8 = () => new Uint8Array(memory.buffer);
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  // Compile the guest once → a program pointer shared (read-only) by every Worker.
  const guest = readFileSync(GUEST);
  const gptr = ex.temen_par_alloc(guest.length);
  u8().set(guest, gptr);
  // §22-JIT mode (TEMEN_JIT=1): build the Rust-side shared powerbox (sets a process-wide static visible
  // to every Worker) and reserve the dispatch table. The worker loop below is unchanged — JIT events
  // are serviced entirely in-Rust against the shared powerbox + Domain (the host never sees them).
  const jitMode = process.env.TEMEN_JIT === '1';
  if (jitMode && ex.temen_par_powerbox(gptr, guest.length) !== 1) {
    console.log('FAIL: temen_par_powerbox returned 0 (powerbox build failed)'); process.exit(1);
  }
  // §22 real-codegen mode (TEMEN_JIT_CODEGEN=1): the host-compiled unit's wasm is emitted + stashed, and
  // each worker's `Jit.invoke` runs it on emitted wasm (the JIT_INVOKE handler above).
  const jitCodegen = process.env.TEMEN_JIT_CODEGEN === '1';
  const jitService = Number(process.env.TEMEN_JIT_SERVICE ?? 0); // 0 = i32 service, 1 = f64 service
  if (jitCodegen) ex.temen_par_jit_codegen_service(jitService);
  if (jitCodegen && ex.temen_par_powerbox_jit_codegen(gptr, guest.length) !== 1) {
    console.log('FAIL: temen_par_powerbox_jit_codegen returned 0'); process.exit(1);
  }
  const prog = (jitMode || jitCodegen) ? ex.temen_par_compile_jit(gptr, guest.length) : ex.temen_par_compile(gptr, guest.length);
  if (prog === 0) { console.log('FAIL: temen_par_compile returned null (decode/unsupported)'); process.exit(1); }

  // The one shared guest window every vCPU runs over (TEMEN_WIN sizes it — the §14 kernels declare a
  // 1 MiB window so their 64 KiB carves stay wasm-page-aligned).
  const winSize = Number(process.env.TEMEN_WIN ?? 1 << 16);
  const win = ex.temen_par_alloc(winSize);

  // 4d I/O mode (TEMEN_IO=1): publish the run's shared powerbox — a `Mutex<Host>` in shared linear
  // memory every vCPU dispatches `call.cap` through, so worker vCPUs do host I/O with no JS in the
  // loop. Stdout accumulates in the powerbox; main reads it back after the run.
  if (process.env.TEMEN_IO === '1' && ex.temen_par_powerbox_io() !== 1) {
    console.log('FAIL: temen_par_powerbox_io returned 0'); process.exit(1);
  }
  // §14 mode (TEMEN_INST=1): publish the run recipe — the root's `Instantiator` spans the window, plus
  // the optional granted module (TEMEN_INST_UNIT) for `instantiate_module`. The root vCPU builds its own
  // powerbox from it (temen_par_root); confined children build theirs in-engine.
  // §14 real-codegen mode (TEMEN_INST_CODEGEN=1): same recipe as TEMEN_INST, but each confined child whose
  // granted-unit entry is in-subset runs it on emitted wasm (the confined-child block in worker()).
  const instCodegen = process.env.TEMEN_INST_CODEGEN === '1';
  if (process.env.TEMEN_INST === '1' || instCodegen) {
    let uptr = 0, ulen = 0;
    if (process.env.TEMEN_INST_UNIT) {
      const unit = readFileSync(process.env.TEMEN_INST_UNIT);
      uptr = ex.temen_par_alloc(unit.length);
      u8().set(unit, uptr);
      ulen = unit.length;
    }
    if (ex.temen_par_powerbox_inst(BigInt(winSize), uptr, ulen) !== 1) {
      console.log('FAIL: temen_par_powerbox_inst returned 0'); process.exit(1);
    }
  }

  console.log(`module: ${WASM}  shared=${memory.buffer instanceof SharedArrayBuffer}`);
  console.log(`  prog@0x${prog.toString(16)}  window@0x${win.toString(16)} (${winSize >> 10}KiB)  TLS ${tlsSize}B`);

  // wasm-JIT tier-up (TEMEN_TIERUP=1): each Worker enables the tier-up bitmap from the guest bytes
  // (kept live at `gptr`) and runs eligible compute regions on emitted wasm. The guest still runs on
  // the interpreter — only direct calls to emitted pure leaves tier up.
  const tierup = process.env.TEMEN_TIERUP === '1';
  const tierupPaged = process.env.TEMEN_TIERUP_PAGED === '1'; // #750: paged tier opt-in
  // A shared i32 cell every Worker atomically bumps on each tier-up / JIT-codegen invoke — proves the
  // seam actually fired (a result match alone couldn't distinguish "ran emitted wasm" from "silently
  // interpreted"). Shared by the tier-up and §22-codegen paths (they never run in the same run).
  const tierupCell = (tierup || jitCodegen || instCodegen) ? ex.temen_par_alloc(4) : 0;

  const workers = new Set();
  let started = 0;
  const t0 = process.hrtime.bigint();

  const startVcpu = (cfg) => {
    started++;
    const w = new Worker(new URL(import.meta.url), {
      workerData: { module, memory, prog, win, winSize, tierup, tierupPaged, jitCodegen, instCodegen, jitService, gptr, glen: guest.length, tierupCell, ...cfg },
    });
    workers.add(w);
    w.on('message', (m) => {
      if (m.kind === 'spawn') {
        // A vCPU asked to spawn a (plain or §14-confined) child: start its Worker with the message's
        // cfg verbatim (slot/stack/TLS already allocated by the parent; a confined child's message
        // carries its own win/winSize — the carve — overriding the run defaults).
        const { kind, ...cfg } = m;
        startVcpu({ role: 'child', ...cfg });
      } else if (m.kind === 'done') {
        finish(BigInt(m.value));
      } else if (m.kind === 'trap' || m.kind === 'fail') {
        finish(null, m.why || 'guest trap');
      }
    });
    w.on('error', (e) => finish(null, String(e)));
  };

  let finished = false;
  const finish = (value, err) => {
    if (finished) return;
    finished = true;
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    for (const w of workers) w.terminate();
    let ok = err == null && value === EXPECT;
    console.log(`  vCPUs started: ${started} (1 root + ${started - 1} spawned), ${ms.toFixed(0)} ms`);
    if (err) console.log(`  error: ${err}`);
    else console.log(`  root returned ${value}  expect ${EXPECT}  ${ok ? '✓' : '✗'}`);
    // Non-vacuity: with TEMEN_TIERUP / TEMEN_JIT_CODEGEN / TEMEN_INST_CODEGEN the workers must have actually
    // run emitted wasm.
    if (tierup || jitCodegen || instCodegen) {
      const ran = Atomics.load(new Int32Array(memory.buffer), tierupCell >> 2);
      const ranOk = ran > 0;
      const label = jitCodegen ? 'JIT-codegen invokes' : instCodegen ? 'inst-codegen children' : 'tier-ups';
      console.log(`  ${label} fired: ${ran}  ${ranOk ? '✓ (ran emitted wasm)' : '✗ (vacuous — never ran emitted wasm)'}`);
      ok = ok && ranOk;
    }
    // 4d I/O mode: read the shared powerbox's accumulated stdout back and check the expected
    // schedule-independent bytes ("tick\n" × TEMEN_IO_LINES, default 8).
    if (process.env.TEMEN_IO === '1') {
      const len = ex.temen_par_stdout_len();
      const out = Buffer.from(u8().slice(ex.temen_par_stdout_ptr(), ex.temen_par_stdout_ptr() + len)).toString();
      const want = 'tick\n'.repeat(Number(process.env.TEMEN_IO_LINES ?? 8));
      const outOk = out === want;
      console.log(`  stdout: ${JSON.stringify(out)}  ${outOk ? '✓' : `✗ (want ${JSON.stringify(want)})`}`);
      ok = ok && outOk;
    }
    console.log(`\n${ok ? 'PASS' : 'FAIL'}: one guest's vCPUs ran on ${started} separate Workers over ` +
      `one shared memory, synchronising via Atomics (join) ${ok ? '— genuine wasm parallelism' : ''}`);
    process.exit(ok ? 0 : 1);
  };

  // The root vCPU runs on its own Worker (it blocks on join/futex, which the main thread may not do).
  const rootSlot = ex.temen_par_alloc(SLOT);
  const rootStackTop = ex.temen_par_alloc(STACK) + STACK;
  const rootTlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
  startVcpu({ role: 'root', func: 0, slot: rootSlot, stackTop: rootStackTop, tlsBase: rootTlsBase });
}

if (isMainThread) main(); else worker();
