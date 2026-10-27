// The Web Worker of a parallel run (executor 2, #1414 B6). The root's Worker (`x2root`) opens the
// run and pumps it (`temen_x2_open`, `temen_x2_run`), and each thread of the run gets a Worker of its
// own (`x2thread`, `temen_x2_thread`), all over the ONE shared linear memory. The page creates every
// Worker (no nested Workers) and never blocks: a browser permits a blocking `Atomics.wait` only off
// the main thread, so the Workers do all the blocking.

import { adoptEngine, foreignMemory, registerForeign, releaseForeign } from './foreign-mem.js';
import { engineImports } from './engine-imports.js';
import { emittedTier, tierupJitArg, tierupJitRes } from './coop-driver.js';
const STACK = 1 << 20; // per-Worker stack
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);

self.onmessage = async (e) => {
  const { module, memory, win, winSize, role, stackTop, tlsBase, tierup, gptr, glen, tierupCell, start,
    seedLen, x2Mem, x2MemId = -1, stream = false } = e.data;
  // Liveness backstop: a trap here (the instantiate, or the engine past it) rejects this async
  // onmessage, and a Worker's unhandled rejection does NOT fire `Worker.onerror` on the page, so the
  // page would wait on a run that has ended. Wrap the entire body so ANY trap is reported as `fail`,
  // with the panic site the hook captured.
  let ex;
  try {
  // The engine imports `temen_host.webgpu_op` (the `webgpu` capability's host seam). A Worker has
  // no GPU surface (the playground's GPU reactor runs on the main thread via par.js), so stub it to a
  // no-op — a guest that resolves the `webgpu` cap here gets -1 and skips. Without it the instantiate
  // fails with "Import temen_host: module is not an object or function".
  // `stream_chunk` (the live output tee, #1414 S5a) posts each chunk a thread of this Worker wrote to
  // the page, which orders them by `seq`, when the page asked for the run's output (`onOutput`).
  // The parallel driver's platform (#1414 B6, `X2_PLATFORM` in lib.rs): each thread of its run is a
  // Worker the page starts on `x2spawn`, with a stack and TLS block of its own allocated here, and
  // its clock is one every Worker reads alike. A thread of a detached child steps over the child's own
  // memory (`memId`, B6-3c), which this Worker minted for the child or registered for its own task:
  // the new Worker registers it under the same id. A child's memory this Worker only minted is the
  // child's threads' once its first thread has it, so the memory goes with the child's last thread.
  const x2_spawn = (threadStart, memId) => {
    const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;
    const stackTop = ex.temen_par_alloc(STACK) + STACK;
    const tlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
    const msg = { kind: 'x2spawn', start: threadStart, stackTop, tlsBase };
    if (memId >= 0) Object.assign(msg, { x2Mem: foreignMemory(memId), x2MemId: memId });
    self.postMessage(msg);
    if (memId >= 0 && memId !== x2MemId) releaseForeign(memId);
    return 0;
  };
  // The thread's emitted tiers over the engine's memory: the coop driver's (`emittedTier`), each made
  // at the thread's first call that needs it. The run's (#1414 B6-3) has the run's tier-up module as
  // its program, over the run's dispatch table as `temen_x2_*` reads it during a call. A leaf's
  // (#1414 5c) has each leaf image the thread runs whole as an instance on it, as the pump's driver
  // runs one. A leaf calls its own functions, so its tier's table stays empty, as a child's does
  // (below) and the pump's does for a root leaf. A call runs `f(win, env, ...argv)` with `"mapped"`
  // set (and, on a page-checked tier, `"pagestate"` at its thread's page-state table) and writes its
  // results at `out`, returning how many, or -(1 + the code the emitted code trapped with). On a
  // spilling tier its frames push to the thread's spill stack, which lives as long as the thread. The
  // emitted code runs on a slice of fuel: each time it spends one it asks for another, which it gets
  // unless the run's `over` byte is set, so a run that ends while this thread is in emitted code
  // stops it within a slice. The emitted tier is not metered: the slice is only how often it looks.
  const X2_SLICE = 1n << 20n;
  let x2Tier = null, x2LeafTier = null, x2Over = 0, x2Win = 0, x2Mapped = 0n;
  // #2474: how many slices the emitted code has spent, so a tier-up can report the work it did.
  let x2Slices = 0;
  const x2Fuel = () => {
    if (Atomics.load(new Uint8Array(memory.buffer), x2Over) !== 0) return undefined;
    x2Slices++;
    return X2_SLICE;
  };
  const x2Bytes = (len) => {
    if (len === 0) return null;
    const p = Number(ex.temen_x2_bytes_ptr());
    return new Uint8Array(memory.buffer).slice(p, p + len);
  };
  // How a tier over the engine's memory reaches the run: the run's tier as it is, and a leaf's with an
  // empty table.
  const x2Opts = {
    tableLog2: () => ex.temen_x2_table_log2(),
    nfuncs: () => ex.temen_x2_nfuncs(),
    tableGen: () => ex.temen_x2_table_gen(),
    slotUnit: (slot) => ex.temen_x2_slot_unit(slot),
    slotUnitBytes: (slot) => x2Bytes(ex.temen_x2_slot_wasm_len(slot)),
    shimBytes: (slot) => x2Bytes(ex.temen_x2_shim_len(slot)),
    callInterp: (target, argsPtr, spillLen) => ex.temen_x2_call_interp(target, argsPtr, spillLen),
    paged: () => ex.temen_x2_paged(),
    mapped: () => x2Mapped,
    mappedNow: () => ex.temen_x2_mapped_now(),
    pagestatePtr: () => ex.temen_x2_pagestate_ptr(),
    winPtr: () => x2Win,
    deliverFaultAddr: (addr) => ex.temen_x2_deliver_fault_addr(addr),
    spillBytes: () => ex.temen_x2_spill_bytes(),
    spillPtr: () => ex.temen_x2_spill_ptr(),
    fuelOut: x2Fuel,
  };
  // What a tier-up of program `module` runs over the engine's memory, and the tier it runs on: its
  // leaf image, which this Worker instantiates the first time (#1414 5c: the root's is program 0,
  // and an image a process execs or spawns has its own), or else the run's own region emit. The run
  // takes a program as a leaf before any of its tier-ups, so what a Worker finds first holds for
  // the run. An invoke (`undefined`) runs a unit, on the run's tier.
  const x2Leaves = new Map();
  const x2Program = (module) => {
    let p = module === undefined ? null : x2Leaves.get(module);
    if (p === undefined) {
      const bytes = x2Bytes(ex.temen_x2_leaf_len(module));
      if (bytes) {
        x2LeafTier ??= emittedTier(ex, memory, {
          ...x2Opts,
          tableLog2: () => 0,
          nfuncs: () => 0,
          slotUnit: () => -1n,
          slotUnitBytes: () => null,
          shimBytes: () => null,
        });
        p = new WebAssembly.Instance(new WebAssembly.Module(bytes), x2LeafTier.imports()).exports;
        x2LeafTier.register(p);
      }
      x2Leaves.set(module, p);
    }
    if (p) return [x2LeafTier, p];
    if (x2Tier === null) {
      x2Tier = emittedTier(ex, memory, x2Opts);
      const at = Number(ex.temen_x2_wasm_ptr()), n = Number(ex.temen_x2_wasm_len());
      if (n > 0) {
        const bytes = new Uint8Array(memory.buffer).slice(at, at + n);
        x2Tier.setProgram(new WebAssembly.Instance(new WebAssembly.Module(bytes), x2Tier.imports()).exports);
      }
    }
    return [x2Tier, x2Tier.program()];
  };
  const x2Enter = (module, win, mapped, over) => {
    x2Win = win;
    x2Mapped = mapped;
    x2Over = over;
    // Before the globals below are armed: they arm a leaf's instance made here too.
    const [tier, program] = x2Program(module);
    tier.syncTableSync();
    for (const g of tier.mappedGlobals) g.value = mapped;
    if (ex.temen_x2_paged()) {
      const ps = Number(ex.temen_x2_pagestate_ptr());
      for (const g of tier.pagestateGlobals) g.value = ps;
    }
    for (const g of tier.fuelGlobals) g.value = X2_SLICE;
    tier.armEnv();
    if (tierupCell) Atomics.add(new Int32Array(memory.buffer), tierupCell >> 2, 1); // the run's count
    return [tier, x2Win, program];
  };
  // A detached child's tier (#1414 ED-2): its granted module's emit over the child's own memory,
  // which this Worker registered for the child's root task. The call's context is the first in the
  // child's header, and the window starts past the header. The module's emit dispatches through a
  // table of its own functions, so the tier's shared table stays empty. The engine reads a bounce's
  // slots from its own memory, so a bounce mirrors the context through engine scratch. A paged
  // emit's page-state table is copied into the header after the contexts, at each call and after
  // each bounce.
  let x2Child = null, x2ChildModule = -1, x2ChildScratch = 0;
  const x2ChildTable = () => {
    const off = ex.temen_detached_pagestate_off();
    const p = Number(ex.temen_x2_pagestate_ptr()), n = ex.temen_x2_pagestate_len();
    if (off + n > ex.temen_detached_header_bytes()) {
      throw new Error('pagestate table does not fit the detached header');
    }
    new Uint8Array(x2Mem.buffer).set(new Uint8Array(memory.buffer).subarray(p, p + n), off);
    return off;
  };
  const x2ChildEnter = (mem, module, mapped, over) => {
    if (mem !== x2MemId) throw new Error(`memory ${mem} is not this Worker's`);
    x2Mapped = mapped;
    x2Over = over;
    if (x2Child === null) {
      const envBytes = ex.temen_wasmjit_env_bytes();
      if (envBytes > ex.temen_detached_pagestate_off()) {
        throw new Error('env cell does not fit the detached header');
      }
      x2ChildScratch = Number(ex.temen_alloc(envBytes));
      x2Child = emittedTier(ex, x2Mem, {
        envCell: 0,
        tableLog2: () => 0,
        nfuncs: () => 0,
        tableGen: () => ex.temen_x2_table_gen(),
        slotUnit: () => -1n,
        slotUnitBytes: () => null,
        shimBytes: () => null,
        callInterp: (target, argsPtr, spillLen) => {
          const s = x2ChildScratch;
          new Uint8Array(memory.buffer).set(new Uint8Array(x2Mem.buffer).subarray(0, envBytes), s);
          const rc = ex.temen_x2_call_interp(target, s + argsPtr, spillLen);
          new Uint8Array(x2Mem.buffer).set(new Uint8Array(memory.buffer).subarray(s, s + envBytes), 0);
          return rc;
        },
        paged: () => ex.temen_x2_paged(),
        mapped: () => x2Mapped,
        mappedNow: () => ex.temen_x2_mapped_now(),
        pagestatePtr: x2ChildTable,
        winPtr: () => ex.temen_detached_header_bytes(),
        deliverFaultAddr: (addr) => ex.temen_x2_deliver_fault_addr(addr),
        spillBytes: () => 0,
        spillPtr: () => 0,
        fuelOut: x2Fuel,
      });
    }
    if (module !== x2ChildModule) {
      const bytes = x2Bytes(ex.temen_x2_program_len(module));
      if (bytes === null) throw new Error(`program ${module} has no emit`);
      x2Child.setProgram(new WebAssembly.Instance(new WebAssembly.Module(bytes), x2Child.imports()).exports);
      x2ChildModule = module;
    }
    for (const g of x2Child.mappedGlobals) g.value = mapped;
    if (ex.temen_x2_paged()) {
      const at = x2ChildTable();
      for (const g of x2Child.pagestateGlobals) g.value = at;
    }
    for (const g of x2Child.fuelGlobals) g.value = X2_SLICE;
    x2Child.armEnv();
    if (tierupCell) Atomics.add(new Int32Array(memory.buffer), tierupCell >> 2, 1); // the run's count
    return [x2Child, ex.temen_detached_header_bytes(), x2Child.program()];
  };
  // Enter a tier (`enter` returns it, the window's base, the emitted function and its arguments),
  // call the function, and write its results at `out` (`res` makes each a slot). Whatever throws
  // comes back as the task's trap, never out of the Worker: the emitted code's own, or MALFORMED when
  // the tier could not be entered.
  const x2Call = (enter, out, res) => {
    let tier, win, f, args;
    try {
      [tier, win, f, args] = enter();
    } catch {
      return -1 - 12; // temen_ir::trap_code::MALFORMED: e.g. a module that does not instantiate
    }
    try {
      const ret = f(win, tier.envCell, ...args);
      const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
      const view = new BigInt64Array(memory.buffer); // the emitted code may have grown the memory
      for (let i = 0; i < rets.length; i++) view[(out >>> 3) + i] = res(rets[i], i);
      return rets.length;
    } catch (e) {
      return -1 - tier.trapOf(e);
    }
  };
  // A tier-up of the thread's task: `f{func}` of program `module`'s emit, its slots all i64. Over
  // the engine's memory (`mem` -1) it is a leaf image the run runs whole, or the run's own program;
  // over a child's own memory, the child's granted module. The fuel the call spent goes to `work`,
  // for the run's tier-up profile.
  const x2_tierup = (module, func, argv, argc, win, mapped, over, out, mem, work) => {
    let fuel, slices;
    const n = x2Call(() => {
      const [tier, base, program] =
        mem < 0 ? x2Enter(module, win, mapped, over) : x2ChildEnter(mem, module, mapped, over);
      const at = argv >>> 3, view = new BigInt64Array(memory.buffer);
      const args = [];
      for (let i = 0; i < argc; i++) args.push(view[at + i]);
      fuel = program.fuel;
      slices = x2Slices;
      return [tier, base, program['f' + func], args];
    }, out, (r) => BigInt(r));
    if (fuel) {
      const spent = BigInt(x2Slices - slices) * X2_SLICE + X2_SLICE - fuel.value;
      new BigUint64Array(memory.buffer)[work >>> 3] = spent > 0n ? spent : 0n;
    }
    return n;
  };
  // A §22 invoke of the unit with code handle `code`: its emitted module's `f0`, instantiated once
  // per Worker, its slots marshalled by the types at `types`.
  const x2_invoke = (code, wasm, wasmLen, types, nparams, nresults, argv, win, mapped, over, out) => {
    const tc = new Uint8Array(memory.buffer).slice(types, types + nparams + nresults);
    return x2Call(() => {
      const [tier, base] = x2Enter(undefined, win, mapped, over);
      const unit = tier.unitForSync(code, new Uint8Array(memory.buffer).slice(wasm, wasm + wasmLen));
      const at = argv >>> 3, view = new BigInt64Array(memory.buffer);
      const args = [];
      for (let i = 0; i < nparams; i++) args.push(tierupJitArg(view[at + i], tc[i]));
      return [tier, base, unit['f0'], args];
    }, out, (r, i) => tierupJitRes(r, tc[nparams + i]));
  };
  const stream_chunk = (fd, seq, ptr, len) => {
    if (!stream) return;
    const bytes = new Uint8Array(memory.buffer, ptr >>> 0, len >>> 0).slice();
    self.postMessage({ kind: 'out', fd, seq: seq >>> 0, bytes }, [bytes.buffer]);
  };
  // A call this Worker's thread makes to a JS capability (#2579), whose function lives on the page:
  // the call goes there, and the thread waits inside it until the page has answered in this
  // Worker's cell (`[flag: i32, _, result: i64]`). Its arguments and its window view (`mem`) stay
  // where they are, live while the call is, and the page reads and writes the window through that
  // view alone (par.js).
  let jsCell = 0;
  const js_cap_call = (slot, op, argsPtr, nArgs, mem) => {
    jsCell ||= ex.temen_par_alloc(16);
    const flag = new Int32Array(memory.buffer, jsCell, 1);
    Atomics.store(flag, 0, 0);
    self.postMessage({ kind: 'jscap', cell: jsCell, slot, op, argsPtr, nArgs, mem });
    while (Atomics.load(flag, 0) === 0) Atomics.wait(flag, 0, 0);
    return new BigInt64Array(memory.buffer, jsCell + 8, 1)[0];
  };
  ({ exports: ex } = await WebAssembly.instantiate(module, engineImports(memory, { stream_chunk, x2_spawn, x2_tierup, x2_invoke, js_cap_call })));
  ex.__stack_pointer.value = stackTop; // this Worker's private stack...
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(tlsBase); // ...and TLS block (per 4b)
  adoptEngine(ex, memory); // after the stack: it calls into the engine
  // #1414 B6 — the parallel driver: the root's Worker pumps the run, and each of its threads is a call
  // on a Worker of its own (`x2_spawn` above). The root reports how the run ended.
  if (role === 'x2root') {
    // A runtime-compile guest reads its unit blobs from the window: the page staged them in `win`.
    if (ex.temen_x2_open(gptr, glen, winSize, tierup ? 1 : 0, win, seedLen) === 0) {
      // #1414 5b-2: a run the page has a part in tells this Worker of each call the page answers and
      // of each idle, as the pump's driver hears of them. The page leaves its answers and input in
      // the engine's memory, where each pump takes them, and wakes this Worker for them.
      for (;;) {
        const ev = ex.temen_x2_run();
        if (ev === 6 /* COOP_RUN_CAP_PARK */) {
          const n = ex.temen_coop_cap_len();
          const words = new BigInt64Array(memory.buffer, Number(ex.temen_coop_cap_ptr()), n).slice();
          self.postMessage({ kind: 'cap', id: words[0], index: Number(words[1]), op: Number(words[2]), args: Array.from(words.slice(3)) });
        } else if (ev === 7 /* COOP_RUN_IDLE */) {
          self.postMessage({ kind: 'idle' });
          ex.temen_x2_wait(); // until the page acts, a task makes a call, or the run ends
        } else {
          break; // 0 = done, 2 = trapped: the run's report is published
        }
      }
    }
    // The root's thread ran its task here.
    for (const tier of [x2Tier, x2LeafTier]) tier?.free();
    const status = ex.temen_status(); // lib.rs STATUS_*: 0 ok · 3 trap · 5 exit
    if (status === 0) {
      self.postMessage({ kind: 'done', value: ex.temen_run_value().toString() });
    } else if (status === 5) {
      self.postMessage({ kind: 'exit', code: ex.temen_exit_code() });
    } else if (status === 3) {
      const p = ex.temen_trap_ptr(), n = ex.temen_trap_len();
      const name = new TextDecoder().decode(new Uint8Array(memory.buffer).slice(p, p + n));
      self.postMessage({ kind: 'trap', why: `guest trap: ${name}`, faultAddr: ex.temen_fault_addr() });
    } else {
      self.postMessage({ kind: 'fail', why: `the parallel driver did not run the guest (status ${status})` });
    }
    return;
  }
  if (role === 'x2thread') {
    if (x2Mem) registerForeign(x2Mem, ex.temen_detached_header_bytes(), x2MemId);
    ex.temen_x2_thread(start);
    for (const tier of [x2Tier, x2LeafTier]) tier?.free();
    if (x2Child !== null) ex.temen_dealloc(x2ChildScratch, ex.temen_wasmjit_env_bytes());
    self.close(); // the thread has counted itself out of its run: this Worker has nothing left to do
    return;
  }
  throw new Error(`unknown role ${role}`);
  } catch (err) {
    let why = `${role} setup/host trap: ${err && err.message ? err.message : err}`;
    try {
      const plen = ex && ex.temen_par_last_panic_len ? ex.temen_par_last_panic_len() : 0;
      if (plen > 0) {
        const p = Number(ex.temen_par_last_panic_ptr());
        why += ` | panic: ${new TextDecoder().decode(new Uint8Array(memory.buffer).slice(p, p + plen))}`;
      }
    } catch { /* accessor absent or memory unusable — the trap text alone still ships */ }
    self.postMessage({ kind: 'fail', why });
  }
};
