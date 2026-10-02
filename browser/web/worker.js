// THREADS/BROWSER step 4c-wasm in a REAL browser — the per-vCPU Web Worker. One guest vCPU runs here
// through the engine's resumable `Vcpu` API (`temen_par_run` → a host-serviced event → deliver → run
// again) over the ONE shared linear memory. This is the browser twin of `threads-spawn.mjs`'s
// `worker()`: the only differences are init delivery (a `postMessage` instead of Node `workerData`)
// and that a spawn request is posted to the page (which creates every Worker — no nested Workers).
//
// The host services events with genuine browser primitives: `thread.join` → `Atomics.wait` on the
// child's completion slot; `memory.wait`/`notify` → `Atomics.wait`/`notify` on the futex word. A Worker
// (not the page) is the only place a browser permits a blocking `Atomics.wait`.

import { foreignImports, registerForeign } from './foreign-mem.js';
const STACK = 1 << 20; // per-Worker stack
const SLOT = 16; // completion slot: [done:i32 @0][result:i64 @8]
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);
// Event codes — must match browser/src/lib.rs PAR_*.
const DONE = 0, TRAP = 1, SPAWN = 2, JOIN = 3, WAIT = 4, NOTIFY = 5, TIERUP = 7, JIT_INVOKE = 8,
  INSTANTIATE_DETACHED = 9;

// §22 codegen arg/result marshalling by scalar type code (0=i32, 1=i64, 2=f32, 3=f64): the engine
// carries every arg/result as a raw i64 slot; the Worker converts to/from the JS value the emitted
// wasm function uses (an integer's value, a float's bits reinterpreted). A tiny scratch DataView does
// the float bit-casts.
const _sdv = new DataView(new ArrayBuffer(8));
const jitArg = (slot, tc) => tc === 0 ? Number(BigInt.asIntN(32, slot)) // i32 → Number
  : tc === 1 ? slot // i64 → BigInt
  : tc === 2 ? (_sdv.setInt32(0, Number(BigInt.asIntN(32, slot)), true), _sdv.getFloat32(0, true)) // f32
  : (_sdv.setBigInt64(0, slot, true), _sdv.getFloat64(0, true)); // f64
const jitRes = (ret, tc) => tc === 0 ? BigInt(ret) // i32 value
  : tc === 1 ? ret // i64
  : tc === 2 ? (_sdv.setFloat32(0, ret, true), BigInt(_sdv.getUint32(0, true))) // f32 bits (zero-ext)
  : (_sdv.setFloat64(0, ret, true), _sdv.getBigInt64(0, true)); // f64 bits

self.onmessage = async (e) => {
  const { module, memory, prog, win, winSize, role, func, sp, arg, slot, stackTop, tlsBase,
    smod, entry, slog, vcpu, rootDomain, tierup, gptr, glen, tierupCell, jitCodegen, jitService, instCodegen,
    jitB2, jitRuntime, tierupPaged, childMem, ticket } = e.data;
  // I22 liveness backstop. The `temen_par_run` loop below already catches host traps, but the SETUP +
  // codegen calls before it (WebAssembly.instantiate, temen_par_enable_jit / _jit_codegen /
  // _inst_codegen, temen_par_child*) are the ones a rare shared-memory race actually trips (a double-free
  // in the shared codegen stash → `memory access out of bounds` or a panic=abort `unreachable`). An
  // uncaught trap there rejects this async onmessage, and a Worker's unhandled rejection does NOT fire
  // `Worker.onerror` on the page — so a child that dies here never fills its completion slot and the
  // root's join `Atomics.wait` hangs the whole page (the 30s-timeout flake). Wrap the entire body so
  // ANY trap becomes a clean vCPU trap: wake any joiner, and report `fail` with the captured panic site.
  let ex;
  try {
  // The engine imports `temen_host.webgpu_op` (the `webgpu` capability's host seam). A Worker vCPU has
  // no GPU surface (the playground's GPU reactor runs on the main thread via par.js), so stub it to a
  // no-op — a guest that resolves the `webgpu` cap here gets -1 and skips. Without it the instantiate
  // fails with "Import temen_host: module is not an object or function".
  // `stdout_chunk` (the live-stdout tee) is likewise stubbed — a Worker vCPU streams no card output.
  ({ exports: ex } = await WebAssembly.instantiate(module, { env: { memory }, temen_host: { ...foreignImports(memory), webgpu_op: () => -1n, stdout_chunk: () => {}, js_cap_call: () => -38n } }));
  ex.__stack_pointer.value = stackTop; // this Worker's private stack...
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(tlsBase); // ...and TLS block (per 4b)
  // Views over the shared memory, refreshed when stale: the shared WebAssembly.Memory can GROW
  // mid-run (any Worker's in-wasm allocation — e.g. a §14 module compile+push), and views created
  // before a growth don't cover the new region (an Atomics access past the old length throws).
  let i32v = new Int32Array(memory.buffer), i64v = new BigInt64Array(memory.buffer);
  const i32 = () =>
    i32v.byteLength === memory.buffer.byteLength ? i32v : (i32v = new Int32Array(memory.buffer));
  const i64 = () =>
    i64v.byteLength === memory.buffer.byteLength ? i64v : (i64v = new BigInt64Array(memory.buffer));
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;
  // A §5 'detached' child (#1286) runs over its OWN shared `WebAssembly.Memory` (`childMem`, minted +
  // seeded by the spawning Worker below), reached by the engine through `Region::Foreign` — so its
  // futex words live there, one host header page in, not in the engine memory at `win`. The completion
  // slot protocol stays in the engine memory (`i32()`); only WAIT/NOTIFY switch to these views.
  const fmem = childMem || memory, fbase = childMem ? ex.temen_detached_header_bytes() : win;
  let fi32v = new Int32Array(fmem.buffer);
  const fi32 = () => fi32v.byteLength === fmem.buffer.byteLength ? fi32v : (fi32v = new Int32Array(fmem.buffer));

  // Start a §5 detached child the engine admitted (ticket `t`, window `1 << cslog`, `am` =
  // `(module << 32) | entry`): mint it a fresh shared Memory — one host header page (env cell,
  // page-state, DETACHED_JIT.md §3.1) + the declared window, growable in place to the detached
  // ceiling — and post it with the ticket to a new Worker, which starts the child over it; the engine
  // seeds it there. A shared Memory posts by reference. Returns the child's completion slot. Used for
  // an interpreted spawn (`PAR_INSTANTIATE_DETACHED`) and one from emitted code (`env.instantiate_rec`).
  const postDetached = (t, cslog, am) => {
    const csmod = Number(am >> 32n), centry = Number(BigInt.asUintN(32, am));
    const hdr = ex.temen_detached_header_bytes(), PAGE = 65536;
    const cmem = new WebAssembly.Memory({
      initial: Math.ceil((hdr + 2 ** cslog) / PAGE),
      maximum: Math.ceil((hdr + Number(ex.temen_detached_max_bytes())) / PAGE),
      shared: true,
    });
    const cslot = ex.temen_par_alloc(SLOT);
    const cstackTop = ex.temen_par_alloc(STACK) + STACK;
    const ctlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
    // No `win`/`winSize` in the engine memory: the child's window is in its own. No tier-up bitmap:
    // it binds the engine memory (a child of the granted unit runs that unit's emit instead).
    self.postMessage({
      kind: 'spawn', role: 'detached', ticket: t.toString(), childMem: cmem, slog: cslog,
      smod: csmod, entry: centry, win: 0, winSize: 2 ** cslog, tierup: false,
      slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
    });
    return cslot;
  };

  // #1822 — the code the running emitted frames last passed to `env.trap` before aborting (a memory
  // fault, spent fuel, a spill overflow); the trap deliver hands it to the engine so the guest sees
  // that trap. Reset before each emitted call; `0` (no `env.trap`) is a native wasm trap, kindless.
  let lastTrap = 0;
  const recordTrap = (code) => { lastTrap = code; };

  // wasm-JIT tier-up (threads slice): this Worker enables the tier-up bitmap in this instance —
  // `temen_par_enable_jit` emits the tier-up module (a pure leaf reachable only via `thread.spawn`
  // still emits, since the guest keeps interpreting), stashes its bytes + the decoded module (so a
  // cross-tier leaf's `call_interp` works), and reports whether anything tier-ups. This Worker then
  // instantiates the emitted module against the ONE shared memory (each Worker instantiates its own —
  // wasm tables aren't shareable across Workers). On PAR_TIERUP it calls `f{func}` here.
  let emitted = null, envCell = 0;
  // #750: `tierupPaged` opts the run into the paged tier — unmap/protect guests keep their pure
  // leaves eligible; each TIERUP then also carries a page-state table (see the handler below).
  const enableJit = tierupPaged ? ex.temen_par_enable_jit_paged : ex.temen_par_enable_jit;
  if (tierup && enableJit(gptr, glen) === 1) {
    const wptr = Number(ex.temen_wasmjit_ptr()), wlen = ex.temen_wasmjit_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const emod = await WebAssembly.instantiate(await WebAssembly.compile(bytes), {
      env: {
        memory,
        trap: recordTrap, // a Temen fault's code; the following `unreachable` throws, caught below
        call_interp: (f, argsPtr) => { if (ex.temen_wasmjit_call_interp(f, argsPtr) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    emitted = emod.exports;
    envCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes())); // fuel counter + cross-tier scratch
  }

  // §22 guest-JIT real codegen (BROWSER.md slice 5): the run's single §22 unit was emitted + stashed
  // once at powerbox setup (temen_par_powerbox_jit_codegen); every Worker instantiates its own instance
  // against the ONE shared memory. On PAR_JIT_INVOKE this Worker runs the emitted `f0(win, env, args)`
  // instead of the interpreter. A `new WebAssembly.Module`/`Instance` here is synchronous (the unit is
  // small) so it needs no await inside the event loop.
  let jitUnit = null, jitEnvCell = 0;
  if (jitCodegen) ex.temen_par_jit_codegen_service(jitService | 0); // 0=i32, 1=f64 service (per-instance)
  if (jitCodegen && ex.temen_par_enable_jit_codegen() === 1 && ex.temen_par_jit_unit_wasm_len() > 0) {
    const wptr = Number(ex.temen_par_jit_unit_wasm_ptr()), wlen = ex.temen_par_jit_unit_wasm_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const umod = new WebAssembly.Module(bytes);
    const uinst = new WebAssembly.Instance(umod, {
      env: {
        memory,
        trap: recordTrap,
        call_interp: (f, argsPtr) => { if (ex.temen_wasmjit_call_interp(f, argsPtr) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    jitUnit = uinst.exports;
    jitEnvCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes()));
  }

  // §22 Model B2 cross-Worker (BROWSER.md § "wasm-JIT tier"): a runtime-`Jit.compile`d unit's
  // `call_indirect` must reach units another Worker `install`ed. wasm funcrefs can't cross Workers,
  // so this Worker holds its OWN funcref table mirroring the shared interpreter `Domain`'s slot→unit
  // map, and instantiates each installed unit locally (the emitted units import this table — the Rust
  // emitter runs in B2 mode, `temen_par_jit_set_b2`). Enabled by `jitB2` (the page sets both).
  //   Verified end-to-end by the CI-gated `jitb2` work item (main.js item 12): 8 Workers each
  //   runtime-compile + `install` a unit into the shared table (raced slots) and dispatch it on
  //   B2-emitted wasm through this mirror, interp ≡ B2 codegen ≡ 56. The emitter-level cross-instance
  //   semantics are pinned native by `crates/temen-wasm-jit/tests/b2_install.rs`.
  let jitTable = null;
  const jitInstCache = new Map(); // code handle → instance.exports (per-Worker instantiation)
  if (jitB2) {
    const size = 1 << ex.temen_par_jit_table_log2();
    jitTable = new WebAssembly.Table({ initial: size, maximum: size, element: 'anyfunc' });
  }
  if ((jitB2 || jitRuntime) && !jitEnvCell) jitEnvCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes()));
  // Instantiate a unit's emitted bytes importing this Worker's shared table, or null if not emitted.
  const jitInstantiate = (bytes) =>
    new WebAssembly.Instance(new WebAssembly.Module(bytes), {
      env: {
        memory,
        trap: recordTrap,
        call_interp: (f, a) => { if (ex.temen_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); },
        __indirect_function_table: jitTable,
      },
    }).exports;
  // #1339: get-or-instantiate the unit installed at `slot`, keyed and fetched by its `(domain, unit)`
  // identity — never by the §22 code handle, which the guest revokes right after `install` (the unit
  // stays installed, only its handle dies), so a by-handle fetch came back empty and nulled a live
  // slot. Cached per Worker; null if the unit has no emitted wasm (interpreter-only).
  // The **pending invoke's** unit: its code handle is live for the duration of the `Jit.invoke`, so
  // it is the right cache key here (unlike an installed slot's — see `jitUnitForSlot`). Bytes come
  // from the per-vCPU pending stash the event published.
  const jitUnitForPending = () => {
    const code = ex.temen_par_jit_code(v);
    let inst = jitInstCache.get(code);
    if (inst) return inst;
    const len = ex.temen_par_jit_code_wasm_len(v);
    if (len === 0) return null;
    const ptr = Number(ex.temen_par_jit_code_wasm_ptr(v));
    inst = jitInstantiate(new Uint8Array(memory.buffer).slice(ptr, ptr + len));
    jitInstCache.set(code, inst);
    return inst;
  };
  const jitUnitForSlot = (slot) => {
    const uid = ex.temen_par_jit_slot_unit(slot);
    if (uid < 0n) return null;
    let inst = jitInstCache.get(uid);
    if (inst) return inst;
    const len = ex.temen_par_jit_unit_wasm_by_slot_len(slot);
    if (len === 0) return null;
    const ptr = Number(ex.temen_par_jit_unit_wasm_by_slot_ptr(slot));
    inst = jitInstantiate(new Uint8Array(memory.buffer).slice(ptr, ptr + len));
    jitInstCache.set(uid, inst);
    return inst;
  };
  // #1347: a natural-prefix slot (a program function) that the tier-up module did not emit gets a
  // bounce shim — a one-function module whose `"t"` bounces to `env.call_interp(slot, …)` on THIS
  // Worker's vCPU (`temen_par_inst_call_interp` → `bounce_call`, the live window + powerbox), the
  // coop driver's `shimFor`. Instantiated once per slot (the program never changes within a run).
  const jitShims = new Map();
  // #1339: a shim bounce on the ROOT vCPU lends the process-global §22 slot mirror, so a guest that
  // `Jit.install`s from its *emitted* frame (Forth's outer interpreter) moves the mirror before the
  // frame resumes; rebuild the table right there, since no event boundary intervenes. A §14
  // child uses `temen_par_inst_call_interp` instead — its installs stay in its own table (#1296).
  const rootCallInterp = (t, a) => {
    const bounce = role === 'root' ? ex.temen_par_root_call_interp : ex.temen_par_inst_call_interp;
    if (bounce(v, t, a) !== 0) throw new Error('cross-tier trap');
    if (role === 'root' && ex.temen_par_jit_table_gen() !== jitSyncedGen) jitSyncTable();
  };
  const jitShimFor = (slot) => {
    let f = jitShims.get(slot);
    if (f !== undefined) return f;
    const len = ex.temen_par_shim_wasm_len(slot);
    if (len === 0) { jitShims.set(slot, null); return null; }
    const ptr = Number(ex.temen_par_shim_wasm_ptr(slot));
    const bytes = new Uint8Array(memory.buffer).slice(ptr, ptr + len);
    f = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
      env: {
        memory,
        trap: () => {},
        call_interp: rootCallInterp,
      },
    }).exports['t'];
    jitShims.set(slot, f);
    return f;
  };
  // Mirror the shared `Domain` dispatch table into this Worker's table, exactly as the coop driver's
  // `syncTable`: a slot in the **natural prefix** (`slot < nfuncs`) holds program function `slot` —
  // the tier-up module's emitted `f{slot}` if it emitted, else a bounce shim (#1347); a slot past it
  // holds an installed unit's `f0`, or null when empty/uninstalled (so a stale `call_indirect`
  // traps). Called before each invoke.
  // #1339: rebuild only when the shared slot mirror actually moved (`temen_par_jit_table_gen`) — a
  // run that never installs syncs once, not per invoke. The generation also covers an install made
  // from INSIDE a bounce (see `rootCallInterp`), which the per-event sync alone would miss.
  let jitSyncedGen = -1;
  const jitSyncTable = () => {
    const gen = ex.temen_par_jit_table_gen();
    if (gen === jitSyncedGen) return;
    const size = 1 << ex.temen_par_jit_table_log2();
    const nfuncs = ex.temen_par_nfuncs();
    for (let slot = 0; slot < size; slot++) {
      if (slot < nfuncs) {
        jitTable.set(slot, (emitted && emitted['f' + slot]) || jitShimFor(slot));
        continue;
      }
      const inst = jitUnitForSlot(slot);
      jitTable.set(slot, inst ? inst['f0'] : null);
    }
    jitSyncedGen = gen;
  };

  // §14 instantiate real codegen (BROWSER.md slice 5, detached #1865): a §5 detached child of the
  // granted unit whose entry is eligible runs it on EMITTED WASM here and fills the completion slot
  // its parent joins. The emit binds the child's OWN `WebAssembly.Memory` (`childMem`; the unit's
  // `min 0 / max 65536, shared` memory import binds any shared memory): its window starts one header
  // page in (`fbase`) and its env cell sits at the bottom of that header. The engine sent `smod ≠ 0`
  // only for a child that runs the granted unit (a child of any other module stays interpreted).
  //
  // The child vCPU is built anyway — never `temen_par_run` — to service every `env.call_interp` leaf
  // with the child's OWN powerbox (`temen_par_inst_call_interp` → `bounce_call`), so a leaf may store,
  // `vm_map`, `unmap` or `protect` exactly as the interpreter would. A bounce mirrors the env cell into
  // engine-side scratch (the engine reads a bounce's slots from its own memory), then back, as
  // `driveDetachedRun` does for the op-13 loop. The child's `"mapped"` is its vCPU's committed extent,
  // re-read after each bounce, since a bounced leaf's `vm_map` grows its memory; a paged unit's
  // page-state table is re-synced into the header after each bounce. Its op-17 spawns are detached
  // grandchildren (`env.instantiate_rec`), its threads run over the same child memory, and a carve
  // spawn traps, as the interpreter's detached vCPU does.
  if (role === 'detached' && smod !== 0 && instCodegen
      && ex.temen_par_enable_inst_codegen() === 1 && ex.temen_par_inst_eligible(entry) === 1) {
    const wptr = Number(ex.temen_par_inst_unit_wasm_ptr()), wlen = ex.temen_par_inst_unit_wasm_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const envBytes = ex.temen_wasmjit_env_bytes();
    if (envBytes > ex.temen_detached_pagestate_off()) {
      throw new Error('env cell does not fit the detached header');
    }
    const cv = ex.temen_par_child_detached(prog, BigInt(ticket), registerForeign(childMem, fbase), slog);
    if (cv === 0) {
      Atomics.store(i32(), slot >> 2, 2); Atomics.notify(i32(), slot >> 2);
      self.postMessage({ kind: 'fail', why: 'detached child vcpu build failed (codegen path)' });
      return;
    }
    const paged = ex.temen_par_inst_paged() === 1;
    let uexports = null;
    // A paged unit's page-state table and its coverage, rebuilt from the child's live map. The emitted
    // code reads the table through the child's memory, so it is copied into the header slot reserved
    // for it (as `driveDetachedRun` does).
    const syncPaged = () => {
      uexports.mapped.value = ex.temen_par_ev_b(cv);
      const p = Number(ex.temen_par_tierup_pagestate_ptr(cv));
      const n = ex.temen_par_tierup_pagestate_len(cv), off = ex.temen_detached_pagestate_off();
      if (off + n > fbase) throw new Error('pagestate table does not fit the detached header');
      new Uint8Array(childMem.buffer).set(new Uint8Array(memory.buffer).subarray(p, p + n), off);
      uexports.pagestate.value = off;
    };
    const syncMapped = () => {
      if (paged) syncPaged();
      else uexports.mapped.value = ex.temen_par_win_len(cv);
    };
    const envCell = 0;
    const scratch = Number(ex.temen_par_alloc(envBytes));
    const bounce = (f, a) => {
      new Uint8Array(memory.buffer).set(new Uint8Array(childMem.buffer).subarray(envCell, envCell + envBytes), scratch);
      const rc = ex.temen_par_inst_call_interp(cv, f, scratch + (a - envCell));
      new Uint8Array(childMem.buffer).set(new Uint8Array(memory.buffer).subarray(scratch, scratch + envBytes), envCell);
      return rc;
    };
    const threadSlots = []; // env.thread_spawn handle (index) → thread completion slot ptr
    const uinst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
      env: {
        memory: childMem,
        trap: () => {},
        call_interp: (f, a) => {
          if (bounce(f, a) !== 0) throw new Error('cross-tier trap');
          syncMapped();
        },
        // A carve spawn (op 0/5/13): a detached window has no carve to give a grandchild.
        instantiate: () => { throw new Error('carve spawn from a detached child'); },
        // An op-17 spawn from emitted code: the engine admits the record against THIS child's vCPU as
        // the interpreted op does, a v1 record as a DETACHED grandchild, started here exactly as the
        // interpreter's INSTANTIATE_DETACHED event is and filed in the vCPU's own child table. A
        // refusal is the spawn's `-EINVAL` result; a forged handle throws → this child's slot reads
        // trapped, as the interpreter traps the parent.
        instantiate_rec: (_cwin, inst, rec) => {
          const t = ex.temen_par_inst_instantiate_rec(cv, inst, rec);
          if (t === 0n) throw new Error('instantiate_rec trapped');
          if (t < 0n) return Number(t);
          const gslot = postDetached(t, Number(ex.temen_par_ev_b(cv)), ex.temen_par_ev_d(cv));
          return ex.temen_par_inst_file_child(cv, BigInt(gslot));
        },
        // A join from emitted code resolves in the vCPU's child table, by the rule the interpreted
        // join uses; then wait on the child's slot, as the interpreter's JOIN arm does, and return
        // its window's bytes to the budget that paid.
        join: (_inst, child) => {
          const gslot = Number(ex.temen_par_inst_join(cv, child));
          if (gslot === 0) throw new Error('join of a spent or unissued child');
          Atomics.wait(i32(), gslot >> 2, 0);
          ex.temen_par_inst_end_join(cv);
          if (Atomics.load(i32(), gslot >> 2) === 2) throw new Error('nested child trapped');
          return i64()[(gslot + 8) >> 3];
        },
        // §11 slice 3 — thread/futex ops from an EMITTED unit, serviced through the same spawn
        // relay + completion-slot protocol as the interpreter's SPAWN/JOIN arms. The spawned vCPU
        // runs the granted unit's own `func` (smod — this Worker knows its module) over THIS child's
        // window, in its memory.
        thread_spawn: (func, sp, arg) => {
          const tslot = ex.temen_par_alloc(SLOT);
          const tstackTop = ex.temen_par_alloc(STACK) + STACK;
          const ttlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
          self.postMessage({
            kind: 'spawn', smod, func, sp: sp.toString(), arg: arg.toString(),
            rootDomain, // a thread joins its spawner's domain
            win, winSize, childMem, tierup: false,
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
        // Futex over the child's window (addr confined by the window mask, as the engine does).
        mem_wait: (cwin, addr, expected, timeout, is64) => {
          const a = cwin + (Number(addr) & (winSize - 1));
          const ms = timeout <= 0n ? Infinity : Number(timeout) / 1e6;
          const r = is64
            ? Atomics.wait(new BigInt64Array(childMem.buffer), a >> 3, expected, ms)
            : Atomics.wait(new Int32Array(childMem.buffer), a >> 2, Number(BigInt.asIntN(32, expected)), ms);
          return r === 'ok' ? 0 : r === 'not-equal' ? 1 : 2;
        },
        mem_notify: (cwin, addr, count) => {
          const a = cwin + (Number(addr) & (winSize - 1));
          return Atomics.notify(new Int32Array(childMem.buffer), a >> 2, count >>> 0);
        },
      },
    });
    new DataView(childMem.buffer).setBigInt64(envCell, 1n << 61n, true); // ample fuel
    uexports = uinst.exports;
    if (paged) ex.temen_par_inst_pagestate_sync(cv); // seed the table from the child's live map
    syncMapped();
    // The entry args: the child's starter cap handles, staged by `temen_par_child_detached`.
    const nargs = Number(ex.temen_par_tierup_argv_len(cv)), aptr = Number(ex.temen_par_tierup_argv_ptr(cv));
    const args = [];
    for (let i = 0; i < nargs; i++) args.push(i64()[(aptr >> 3) + i]);
    if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count emitted children (non-vacuity)
    try {
      const ret = uinst.exports['f' + entry](fbase, envCell, ...args);
      i64()[(slot + 8) >> 3] = BigInt(ret); // publish result...
      Atomics.store(i32(), slot >> 2, 1); // ...set done flag...
      Atomics.notify(i32(), slot >> 2); // ...and wake the joiner
    } catch {
      Atomics.store(i32(), slot >> 2, 2); // 2 = trapped (the joiner traps on deliver_join)
      Atomics.notify(i32(), slot >> 2);
    }
    ex.temen_par_free(cv);
    return;
  }

  const v = role === 'root'
    ? ex.temen_par_root(prog, win, winSize, func)
    : role === 'detached'
      ? ex.temen_par_child_detached(prog, BigInt(ticket), registerForeign(childMem, fbase), slog)
      : childMem // a thread of a detached child: over the same child memory (#1865)
        ? ex.temen_par_thread_detached(prog, registerForeign(childMem, fbase), winSize, smod | 0, func, BigInt(sp), BigInt(arg), BigInt(vcpu ?? 0))
        : ex.temen_par_child(prog, win, winSize, smod | 0, func, BigInt(sp), BigInt(arg), BigInt(vcpu ?? 0));
  if (v === 0) { self.postMessage({ kind: 'fail', why: 'vcpu build failed' }); return; }

  for (;;) {
    // I22 hang site. A host wasm trap escaping `temen_par_run` — `memory access out of bounds`, or
    // `unreachable` from a panic=abort engine panic — unwinds into this async `onmessage`, rejecting
    // it. A Worker's unhandled rejection does NOT fire `Worker.onerror` on the page, so par.js's
    // promise would never settle: the vCPU's DOM item would sit `pending` until the harness's 30s
    // `waitForFunction` times out (the silent-flake signature). Convert it into a structured failure —
    // wake any joiner (a non-root vCPU's completion slot) so a parent's `Atomics.wait` doesn't
    // cascade-hang, then report `fail` with the trap text so the page/harness self-identifies.
    let evc;
    try {
      evc = ex.temen_par_run(v);
    } catch (err) {
      if (role !== 'root') {
        const iv = new Int32Array(memory.buffer);
        Atomics.store(iv, slot >> 2, 2); // 2 = trapped
        Atomics.notify(iv, slot >> 2);
      }
      let why = `vcpu ${role} host trap: ${err && err.message ? err.message : err}`;
      // If the trap was a panic=abort engine panic (surfaces as `unreachable`), the Rust panic hook
      // stashed FILE:LINE + message; the trap left memory intact, so read it back here (I22 (a)).
      try {
        const plen = ex.temen_par_last_panic_len ? ex.temen_par_last_panic_len() : 0;
        if (plen > 0) {
          const p = Number(ex.temen_par_last_panic_ptr());
          why += ` | panic: ${new TextDecoder().decode(new Uint8Array(memory.buffer).slice(p, p + plen))}`;
        }
      } catch { /* accessor absent (older build) or read failed — the trap text alone still ships */ }
      self.postMessage({ kind: 'fail', why });
      return; // don't temen_par_free(v): the instance just trapped; the page terminates this Worker
    }
    if (evc === DONE) {
      const value = ex.temen_par_ev_a(v); // i64 → BigInt
      i64()[(slot + 8) >> 3] = value; // publish result...
      Atomics.store(i32(), slot >> 2, 1); // ...set done flag...
      Atomics.notify(i32(), slot >> 2); // ...and wake a joiner
      if (role === 'root') self.postMessage({ kind: 'done', value: value.toString() });
      ex.temen_par_free(v);
      return;
    }
    if (evc === TRAP) {
      Atomics.store(i32(), slot >> 2, 2); // 2 = trapped
      Atomics.notify(i32(), slot >> 2);
      // A member's trap or `exit` is terminal for its whole domain (DESIGN.md §12, I37 — the
      // cooperative driver's `teardown_domains`). For the root domain (the root and its threads) that
      // is the run: report it, and the page tears every Worker down. A §5 detached child's
      // domain ends with it here; its joiner observes the trap through the slot above.
      // ev_b = 1: the guest called `exit(ev_a)`. Otherwise ev_c/ev_d are the trap name's bytes (a
      // `&'static str` in the shared memory).
      if (rootDomain && ex.temen_par_ev_b(v) === 1n) {
        self.postMessage({ kind: 'exit', code: Number(ex.temen_par_ev_a(v)) });
      } else if (rootDomain) {
        const p = Number(ex.temen_par_ev_c(v)), n = Number(ex.temen_par_ev_d(v));
        const name = new TextDecoder().decode(new Uint8Array(memory.buffer).slice(p, p + n));
        self.postMessage({ kind: 'trap', why: `guest trap: ${name}${role === 'root' ? '' : ` (in a spawned thread)`}` });
      }
      ex.temen_par_free(v);
      return;
    }
    if (evc === SPAWN) {
      // ev_a packs (spawning frame's module << 32) | func — the child resolves `func` in that module
      // (an installed §22 unit spawns its own functions).
      const cam = ex.temen_par_ev_a(v);
      const csmod = Number(cam >> 32n), cfunc = Number(BigInt.asUintN(32, cam));
      const csp = ex.temen_par_ev_b(v), carg = ex.temen_par_ev_c(v);
      // Allocate the child's completion slot + stack + TLS, then ask the page to start its Worker.
      const cslot = ex.temen_par_alloc(SLOT);
      const cstackTop = ex.temen_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      self.postMessage({
        kind: 'spawn', smod: csmod, func: cfunc, sp: csp.toString(), arg: carg.toString(),
        vcpu: ex.temen_par_ev_d(v).toString(), // the child's dense vCPU id (seeds its `vcpu.tls`)
        rootDomain, // a thread joins its spawner's domain
        win, winSize,
        // #1865: a detached vCPU's thread shares its window, which is in the child's own memory.
        ...(childMem ? { childMem, tierup: false } : {}),
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      ex.temen_par_deliver_child(v, BigInt(cslot)); // the join hands the slot back
      continue;
    }
    if (evc === JOIN) {
      // The engine resolved the guest's handle (a bad one traps the vCPU) and hands back the slot this
      // Worker delivered for that child, once.
      const cslot = Number(ex.temen_par_ev_a(v));
      Atomics.wait(i32(), cslot >> 2, 0); // block until the child sets its done flag
      const trapped = Atomics.load(i32(), cslot >> 2) === 2;
      ex.temen_par_deliver_join(v, i64()[(cslot + 8) >> 3], trapped ? 1 : 0);
      continue;
    }
    if (evc === INSTANTIATE_DETACHED) {
      // §5 detached child (#1286 slice 3b): the engine admitted it (the Instantiator resolved, the
      // budget charged, the module compiled, its powerbox built); what is left is the window itself.
      const cslot = postDetached(ex.temen_par_ev_a(v), Number(ex.temen_par_ev_b(v)), ex.temen_par_ev_d(v));
      ex.temen_par_deliver_child(v, BigInt(cslot)); // the join hands the slot back
      continue;
    }
    if (evc === WAIT) {
      const addr = Number(ex.temen_par_ev_a(v));
      const expected = Number(BigInt.asIntN(32, ex.temen_par_ev_b(v)));
      const timeoutNs = ex.temen_par_ev_d(v);
      const ms = timeoutNs <= 0n ? Infinity : Number(timeoutNs) / 1e6;
      const r = Atomics.wait(fi32(), (fbase + addr) >> 2, expected, ms); // 'ok' | 'not-equal' | 'timed-out'
      ex.temen_par_deliver_code(v, r === 'ok' ? 0 : r === 'not-equal' ? 1 : 2);
      continue;
    }
    if (evc === NOTIFY) {
      const addr = Number(ex.temen_par_ev_a(v)), count = Number(ex.temen_par_ev_b(v));
      ex.temen_par_deliver_code(v, Atomics.notify(fi32(), (fbase + addr) >> 2, count));
      continue;
    }
    if (evc === TIERUP) {
      // Run the emitted `f{func}(win, env, ...i64 args)` over the shared window instead of
      // interpreting. A trap throws (Temen fault → `env.trap` + `unreachable`, or a wasm trap) — we
      // surface it as a vCPU trap. Otherwise marshal the i64 result slots back to the engine.
      const func = Number(ex.temen_par_ev_a(v));
      const argvPtr = Number(ex.temen_par_tierup_argv_ptr(v)), n = Number(ex.temen_par_tierup_argv_len(v));
      const args = [];
      for (let i = 0; i < n; i++) args.push(i64()[(argvPtr >> 3) + i]); // i64 args → BigInt
      // #717 host sync: the event's committed-extent snapshot → the emitted `"mapped"` global, so
      // the emitted bounds check admits exactly what the interpreter would (idempotent over today's
      // fully-mapped par window; load-bearing once the window can `vm_map`-grow). On a #750 paged
      // run, operand b is the page-state table's COVERAGE (the engine computes it with the table).
      emitted.mapped.value = ex.temen_par_ev_b(v);
      // #750 paged runs: point the emitted `"pagestate"` global at the engine-built table — its
      // Rust-heap address is a linear-memory address (one shared memory, zero copies). Empty (and
      // the global absent) on unpaged runs.
      if (Number(ex.temen_par_tierup_pagestate_len(v)) > 0)
        emitted.pagestate.value = Number(ex.temen_par_tierup_pagestate_ptr(v));
      new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true); // ample fuel; preempt = write < 0
      if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count tier-ups (non-vacuity)
      lastTrap = 0;
      try {
        const ret = emitted['f' + func](win, envCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rptr = Number(ex.temen_par_alloc(Math.max(1, rets.length) * 8));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = BigInt(rets[i]);
        ex.temen_par_deliver_tierup(v, rptr, rets.length);
      } catch {
        ex.temen_par_deliver_tierup_trap(v, lastTrap);
      }
      continue;
    }
    if (evc === JIT_INVOKE) {
      // §22 guest-JIT real codegen: the guest `Jit.invoke`d a unit — run the emitted unit's
      // `f0(win, env, ...args)` over the shared window instead of the interpreter, then deliver its
      // result slots. Args marshal by declared type (i32 → JS Number, i64 → BigInt) so a unit need not
      // be all-i64; results go back as `BigInt(ret)` (the engine re-tags by result type). A trap
      // throws and surfaces as a vCPU trap (as an interp invoke would).
      const argvPtr = Number(ex.temen_par_jit_argv_ptr(v)), n = Number(ex.temen_par_jit_argv_len(v));
      const ptypes = new Uint8Array(memory.buffer, Number(ex.temen_par_jit_param_types_ptr(v)), n);
      const args = [];
      for (let i = 0; i < n; i++) args.push(jitArg(i64()[(argvPtr >> 3) + i], ptypes[i]));
      new DataView(memory.buffer).setBigInt64(jitEnvCell, 1n << 61n, true); // ample fuel
      if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count emitted invokes (non-vacuity)
      // Model B2: mirror the shared dispatch table into this Worker's table, then run the *invoked*
      // unit (resolved by its code handle) — whose `call_indirect`s now reach installed units locally.
      // Otherwise the fixed-unit codegen path runs the run's single pre-instantiated `jitUnit`.
      let unit = jitUnit;
      if (jitB2) {
        jitSyncTable();
        unit = jitUnitForPending();
      } else if (!unit) {
        // Runtime-`Jit.compile` path without B2: resolve the invoked unit by its code handle (each
        // Worker instantiates + caches per handle; the emitted bytes live on the shared host).
        unit = jitUnitForPending();
      }
      if (!unit) { ex.temen_par_deliver_jit_invoke_trap(v, 0); continue; }
      // #717 host sync: the event's committed-extent snapshot → the unit instance's `"mapped"`
      // global (same contract as TIERUP above; an invoke the scalar can't describe never surfaces
      // here — the engine services it on the interpreter instead).
      unit.mapped.value = ex.temen_par_ev_b(v);
      // #1347: the unit's `call_indirect` may land in the tier-up module's emitted `f{i}` through
      // the natural prefix — prime its globals as the TIERUP arm does (on a paged run the engine
      // rebuilt the page-state table for this event, and `ev_b` is its coverage).
      if (emitted) {
        emitted.mapped.value = ex.temen_par_ev_b(v);
        if (Number(ex.temen_par_tierup_pagestate_len(v)) > 0)
          emitted.pagestate.value = Number(ex.temen_par_tierup_pagestate_ptr(v));
        new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true);
      }
      lastTrap = 0;
      try {
        const ret = unit['f0'](win, jitEnvCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rn = Number(ex.temen_par_jit_result_types_len(v));
        const rtypes = new Uint8Array(memory.buffer, Number(ex.temen_par_jit_result_types_ptr(v)), rn);
        const rptr = Number(ex.temen_par_alloc(Math.max(1, rets.length) * 8));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = jitRes(rets[i], rtypes[i]);
        ex.temen_par_deliver_jit_invoke(v, rptr, rets.length);
      } catch {
        ex.temen_par_deliver_jit_invoke_trap(v, lastTrap);
      }
      continue;
    }
  }
  } catch (err) {
    // Liveness backstop (see the note at the top): a trap escaped the setup/codegen path above.
    // Wake any joiner so the parent's `Atomics.wait` on our completion slot doesn't cascade-hang,
    // then report a structured failure carrying the Rust panic location the hook stashed.
    try {
      if (role !== 'root' && slot !== undefined) {
        const iv = new Int32Array(memory.buffer);
        Atomics.store(iv, slot >> 2, 2); // 2 = trapped → the parent's deliver_join sees a trap
        Atomics.notify(iv, slot >> 2);
      }
    } catch { /* memory unusable — nothing more we can do */ }
    let why = `vcpu ${role} setup/host trap: ${err && err.message ? err.message : err}`;
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
