// The Temen **playground** — the human-facing demo the THREADS/BROWSER work builds toward: type Temen
// text, it parses/verifies/encodes *inside the wasm sandbox* (`temen_parse`), and runs across real Web
// Workers (`par.js`, the same orchestration the validation page uses). The powerbox select picks the
// run's recipe: none (compute only), 4d host I/O (stdout read back onto the page), §22 guest-JIT, or
// a §14 root `Instantiator` (sandboxed children on their own Workers). The page services no
// authority either way — all of it is Rust-side, in shared linear memory.

import { loadEngine, makeRunner, readParStdout } from './par.js';
import { openJitReactor } from './wasmjit-reactor.js';
import { runJitModule, runWarmJit, runJitCompiler, runJitSelfhost, runJitNifler } from './wasmjit-module.js';
import { SnapshotClient } from './snapshot-client.js';
import { createDapClient } from './dap.js';
import { initWebGPU, teardownWebGPU, webgpuAvailable } from './webgpu.js';
import { createEditor, setVimAll, refreshAll } from './editor.js';
import { formatPgOutput } from './pg-format.js';

const $ = (id) => document.getElementById(id);

// Each example: the Temen text, its powerbox mode, and what to expect. The kernels are the proven
// schedule-independent ones from `gencorpus.rs` (same ground truths the validation page asserts).
const EXAMPLES = {
  hello: {
    mode: 'io',
    desc: 'One vCPU call.cap-writes a greeting through the host-I/O powerbox and returns the byte ' +
      'count (14). stdout comes back onto the page after the run.',
    src: `memory 16
data 16384 "hello, world!\\n"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 16384
  v2 = i64.const 14
  v3 = call.cap 0 1 (i64, i64) -> (i64) v0(v1, v2)
  return v3
  }
}
`,
  },
  threads: {
    mode: 'plain',
    desc: 'thread.spawn fans 8 vCPUs out — each onto its own real Web Worker — every one ' +
      'atomic.rmw.adds a shared counter 500 times, the root joins them and returns 4000 on every ' +
      'interleaving.',
    src: `memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1) 3()
}
block 2 (v4: i64) {
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16400
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br 1(v12)
}
block 3 () {
  v13 = i64.const 0
  br 4(v13)
}
block 4 (v14: i64) {
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 5(v14) 6()
}
block 5 (v17: i64) {
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16400
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br 4(v25)
}
block 6 () {
  v26 = i64.const 16384
  v27 = i64.atomic.load v26
  return v27
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 3() 2(v1)
}
block 2 (v4: i64) {
  v5 = i64.const 16384
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br 1(v9)
}
block 3 () {
  v10 = i64.const 0
  return v10
  }
}
`,
  },
  io: {
    mode: 'io',
    desc: '8 worker vCPUs (one Web Worker each) all call.cap-write "tick\\n" through the run\'s ONE ' +
      'shared powerbox and bump a shared counter — result 8, stdout "tick\\n" × 8, on every schedule.',
    src: `memory 16
data 16384 "tick\\n"
func (i32) -> (i64) {
block 0 (v0: i32) {
  vh0 = i64.extend_i32_u v0
  v1 = i64.const 0
  br 1(v1, vh0)
}
block 1 (vi: i64, vhh: i64) {
  v2 = i64.const 8
  v3 = i64.lt_u vi v2
  br_if v3 2(vi, vhh) 3()
}
block 2 (vi2: i64, vhh2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vhh2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16400
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br 1(v9, vhh2)
}
block 3 () {
  v10 = i64.const 0
  br 4(v10)
}
block 4 (vj: i64) {
  v11 = i64.const 8
  v12 = i64.lt_u vj v11
  br_if v12 5(vj) 6()
}
block 5 (vj2: i64) {
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16400
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br 4(v20)
}
block 6 () {
  v21 = i64.const 16392
  v22 = i64.atomic.load v21
  return v22
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vh: i64) {
  vhandle = i32.wrap_i64 vh
  vptr = i64.const 16384
  vlen = i64.const 5
  vw = call.cap 0 1 (i64, i64) -> (i64) vhandle(vptr, vlen)
  v1 = i64.const 16392
  v2 = i64.const 1
  v3 = i64.atomic.rmw.add v1 v2
  v4 = i64.const 0
  return v4
  }
}
`,
  },
  jit: {
    mode: 'jit',
    desc: '§22 guest-JIT: 8 worker vCPUs each install a host-compiled unit into the SHARED Domain ' +
      '(a freshly raced dispatch slot) and call_indirect it — service(6,7) = 142, folded to 1136.',
    src: `memory 16
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br 1(vi0, vpacked)
}
block 1 (vi: i64, vp: i64) {
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt 2(vi, vp) 3()
}
block 2 (vi2: i64, vp2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16400
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br 1(v9, vp2)
}
block 3 () {
  vj0 = i64.const 0
  br 4(vj0)
}
block 4 (vj: i64) {
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 5(vj) 6()
}
block 5 (vj2: i64) {
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16400
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br 4(v20)
}
block 6 () {
  v21 = i64.const 16392
  v22 = i64.atomic.load v21
  return v22
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vp: i64) {
  vmask = i64.const 4294967295
  vjit64 = i64.and vp vmask
  vjit = i32.wrap_i64 vjit64
  vsh = i64.const 32
  vcode = i64.shr_u vp vsh
  vslot = call.cap 11 3 (i64) -> (i64) vjit (vcode)
  vslot32 = i32.wrap_i64 vslot
  va = i32.const 6
  vb = i32.const 7
  vr = call.dyn (i32, i32) -> (i32) vslot32 (va, vb)
  vr64 = i64.extend_i32_u vr
  vc8 = i64.const 16392
  vold = i64.atomic.rmw.add vc8 vr64
  vret = i64.const 0
  return vret
  }
}
`,
  },
  inst: {
    mode: 'inst',
    desc: '§14 sandboxing: the root instantiates 8 confined children — each on its OWN Web Worker, ' +
      'confined to a 64 KiB carve of the 1 MiB window with an attenuated powerbox — joins them and ' +
      'sums 8 × 5 = 40.',
    src: `memory 20
func (i32) -> (i64) {
block 0 (v0: i32) {
  vi0 = i64.const 0
  br 1(vi0, v0)
}
block 1 (vi: i64, vinst: i32) {
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt 2(vi, vinst) 3(vinst)
}
block 2 (vi2: i64, vinst2: i32) {
  vone = i64.const 1
  viplus = i64.add vi2 vone
  v64k = i64.const 65536
  voff = i64.mul viplus v64k
  ventry = i64.const 1
  vslog = i64.const 16
  vquota = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vinst2 (ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16400
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  vinext = i64.add vi2 vone
  br 1(vinext, vinst2)
}
block 3 (vinst3: i32) {
  vj0 = i64.const 0
  vs0 = i64.const 0
  br 4(vj0, vs0, vinst3)
}
block 4 (vj: i64, vs: i64, vinst4: i32) {
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 5(vj, vs, vinst4) 6(vs)
}
block 5 (vj2: i64, vs2: i64, vinst5: i32) {
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16400
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = call.cap 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br 4(vjn, vsn, vinst5)
}
block 6 (vs3: i64) {
  return vs3
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 5
  return v1
  }
}
`,
  },

  'Debugger (Temen — breakpoints, step, variables)': {
    debug: true,
    bp: 7, // a breakpoint pre-placed on line 8 (0-based 7), the loop body
    mode: 'plain',
    desc: 'The §DEBUGGING Debug Adapter Protocol debugger, running on the bytecode engine right here ' +
      'in the sandbox — no `debug` section needed. The engine auto-derives a line table and names the ' +
      'SSA values straight from the Temen text, so any program you write here is debuggable. Click the ' +
      'gutter to set/clear breakpoints (one is pre-placed on line 8), then press Debug: it stops at the ' +
      'line, highlights it, and shows the in-scope values (i / acc) in the Variables pane. Step and ' +
      'Continue walk the loop — watch acc accumulate. Run executes it normally (→ 15). Same DAP server ' +
      'VS Code speaks, driven over the wasm FFI.',
    src: `; Sum i = n..1 into acc. Click the gutter to set a breakpoint, then press Debug.
func () -> (i64) {
block 0 () {
  n = i64.const 5
  acc0 = i64.const 0
  br 1(n, acc0)
}
block 1 (i: i64, acc: i64) {
  sum = i64.add acc i
  one = i64.const 1
  next = i64.sub i one
  br_if next 1(next, sum) 2(sum)
}
block 2 (r: i64) {
  return r
  }
}
`,
  },

  'Debugger (Temen — watchpoints / data breakpoints)': {
    debug: true,
    bp: 11, // a breakpoint pre-placed on line 12 (0-based 11), the loop body, so a session pauses to arm
    mode: 'plain',
    desc: 'The same DAP debugger, showing data breakpoints (watchpoints) on the bytecode engine. ' +
      'A counter lives at a fixed window address, named `count` by the `debug` section. Press Debug: it ' +
      'stops at the pre-placed breakpoint (line 12) with `count` in the Variables pane. Click the ● next ' +
      'to `count` to break when it changes, then Continue — the debugger stops the instant the loop-body ' +
      'store writes it (stop reason “data breakpoint”), before the write lands. Continue again to watch ' +
      'it climb 0 → 1 → 2. A promoted SSA value has no address, so its ● is greyed — honestly ' +
      'unwatchable. This is DEBUGGING.md slice 5 (the engine-side watchpoints) reaching the playground.',
    src: `; A counter lives at a fixed window address. Set a watch on \`count\` in the
; Variables pane (click its ● toggle), then Continue: the debugger stops the
; instant a store changes it — stop reason "data breakpoint".
memory 16
func () -> (i64) {
block 0 () {
  a0 = i64.const 16384
  z = i64.const 0
  i64.store a0 z
  br 1(z)
}
block 1 (i: i64) {
  a1 = i64.const 16384
  one = i64.const 1
  n = i64.add i one
  i64.store a1 n
  limit = i64.const 3
  done = i64.ge_s n limit
  br_if done 2(n) 1(n)
}
block 2 (r: i64) {
  a2 = i64.const 16384
  out = i64.load a2
  return out
  }
}

debug.file 0 "counter.temt"
debug.fname 0 "count_up"
debug.loc 0 0 0 0 7 3
debug.loc 0 0 1 0 8 3
debug.loc 0 0 2 0 9 3
debug.loc 0 1 0 0 12 3
debug.loc 0 1 1 0 13 3
debug.loc 0 1 2 0 14 3
debug.loc 0 1 3 0 15 3
debug.loc 0 1 4 0 16 3
debug.loc 0 1 5 0 17 3
debug.loc 0 2 0 0 20 3
debug.loc 0 2 1 0 21 3
debug.type 0 base "long" signed 8
debug.var 0 "count" fixed 16384 "long" 0
`,
  },

  'Debugger (Temen — threads)': {
    debug: true,
    bp: 19, // a breakpoint pre-placed on line 20 (0-based 19), the worker's atomic increment
    mode: 'plain',
    desc: 'The DAP debugger going multithreaded, on the bytecode engine. The root spawns two worker ' +
      'vCPUs that each atomically bump a shared counter, then joins them. A breakpoint is pre-placed in ' +
      'the worker (line 20): press Debug and it stops in the first worker. The Variables pane grows a ' +
      'thread selector — one chip per live vCPU (the one that hit the stop is marked ●). Click another ' +
      'thread to inspect its stack without resuming; Step/Continue always drive the stopped thread. ' +
      'Continue again to catch the second worker, then finish (→ 2). Because the schedule is ' +
      'deterministic, ◀◀ Reverse walks *backward* to the previous worker breakpoint (replay to an ' +
      'earlier global turn), and a data breakpoint fires in whichever thread touches the range. A ' +
      'cooperative debug scheduler runs the whole interleaving right here in the sandbox — DEBUGGING.md ' +
      'Milestone B on the bytecode engine.',
    src: `; The root spawns two worker vCPUs; each atomically bumps a shared counter,
; then the root joins them and returns the total (→ 2). A breakpoint is pre-
; placed in the worker — press Debug, then use the thread selector that appears.
memory 16
func () -> (i64) {
block 0 () {
  sp = i64.const 0
  one = i64.const 1
  h0 = thread.spawn 1 sp one
  h1 = thread.spawn 1 sp one
  j0 = thread.join h0
  j1 = thread.join h1
  addr = i64.const 16384
  total = i64.atomic.load addr
  return total
  }
}
func (i64, i64) -> (i64) {
block 0 (sp: i64, inc: i64) {
  addr = i64.const 16384
  old = i64.atomic.rmw.add addr inc
  z = i64.const 0
  return z
  }
}
`,
  },

  'Debugger (Temen — wait / notify)': {
    debug: true,
    bp: 27, // a breakpoint pre-placed on line 28 (0-based 27), the worker's read *after* the futex wait
    mode: 'plain',
    desc: 'A futex handoff, debugged on the multithreaded bytecode engine. The root (producer) stores a ' +
      'value into mem[8], sets a flag and atomic.notify’s mem[0], then joins; the worker (consumer) ' +
      'atomic.wait’s on mem[0] until woken, then reads mem[8] (→ 987654). A breakpoint is pre-placed on ' +
      'the worker’s read (line 28) — press Debug: the worker parks on the wait, the root’s notify wakes ' +
      'it, and the debugger stops at the read. The thread selector shows both vCPUs; the debug scheduler ' +
      'drives the whole wait/notify handoff deterministically right here in the sandbox.',
    src: `; A futex handoff. The root stores 987654 to mem[8], flags + notifies mem[0],
; and joins. The worker waits on mem[0], then reads mem[8]. A breakpoint sits
; on the worker's read (line 28) — it fires once the root's notify wakes it.
memory 16
func () -> (i64) {
block 0 () {
  a8 = i64.const 16392
  val = i64.const 987654
  i64.atomic.store a8 val
  sp = i64.const 0
  h = thread.spawn 1 sp sp
  a0 = i64.const 16384
  one = i32.const 1
  i32.atomic.store a0 one
  a0n = i64.const 16384
  n1 = i32.const 1
  woke = atomic.notify a0n n1
  r = thread.join h
  return r
  }
}
func (i64, i64) -> (i64) {
block 0 (sp: i64, arg: i64) {
  a0 = i64.const 16384
  exp = i32.const 0
  tmo = i64.const 1000000000
  st = i32.atomic.wait a0 exp tmo
  a8 = i64.const 16392
  got = i64.atomic.load a8
  return got
  }
}
`,
  },

  'Debugger (Temen — fibers / generators)': {
    debug: true,
    bp: 18, // a breakpoint pre-placed on line 19 (0-based 18), inside the fiber body
    mode: 'plain',
    desc: 'Debugging §12 fibers (cooperative coroutines / generators) on the bytecode engine. The root ' +
      'cont.new’s a fiber, then cont.resume’s it twice: the fiber runs, suspends a value back, and on ' +
      'the second resume finishes — the root sums the two (→ 36). A breakpoint sits inside the fiber ' +
      '(line 19): press Debug and the debugger follows cont.resume *into* the fiber and stops there, its ' +
      'own frame live. Step walks the fiber; Continue runs the suspend/resume handoff. ◀◀ Reverse ' +
      'replays across the switches too. The debugger tracks the active continuation as it switches ' +
      'between the root and the fiber, right here in the sandbox.',
    src: `; A generator fiber. The root creates it (cont.new) and resumes it twice; the
; fiber suspends 11 back, then on the next resume returns 25 — the root sums
; them (→ 36). The breakpoint on line 19 fires once cont.resume enters the fiber.
func () -> (i64) {
block 0 () {
  mk = ref.func 1
  z = i64.const 0
  gen = cont.new mk z
  a = i64.const 10
  s0, v0 = cont.resume gen a
  b = i64.const 20
  s1, v1 = cont.resume gen b
  sum = i64.add v0 v1
  return sum
  }
}
func (i64, i64) -> (i64) {
block 0 (sp: i64, arg: i64) {
  one = i64.const 1
  bumped = i64.add arg one
  got = suspend bumped
  five = i64.const 5
  out = i64.add got five
  return out
  }
}
`,
  },

  'Debugger (Temen — fibers + threads)': {
    debug: true,
    bp: 36, // a breakpoint pre-placed on line 37 (0-based 36), inside the fiber body
    mode: 'plain',
    desc: 'Fibers composed with threads on the scheduled bytecode engine. Two worker threads each run a ' +
      '§12 generator fiber: the worker cont.new’s the fiber, cont.resume’s it twice (it suspends 11, then ' +
      'returns 25), and atomically adds the 25 into mem[0] — two workers → 50. A breakpoint sits inside ' +
      'the fiber body (line 37): press Debug and a *worker* vCPU (not the root) stops there, its own ' +
      'fiber frame live. The thread selector switches between the workers; Continue lets the other ' +
      'worker’s fiber hit it too, then the run finishes. ◀◀ Reverse replays the whole schedule — ' +
      'fibers-on-threads included — deterministically. Fibers and threads, composed under one debugger.',
    src: `; Two worker threads, each running a generator fiber. Each worker cont.new’s a
; fiber, resumes it twice (it suspends 11, then returns 25), and atomically adds
; the returned 25 into mem[0]. Two workers → 50. The breakpoint on line 37 fires
; once a worker’s cont.resume enters the fiber — a *worker* vCPU, not the root.
memory 16
func () -> (i64) {
block 0 () {
  sp = i64.const 0
  a = i64.const 0
  w0 = thread.spawn 1 sp a
  w1 = thread.spawn 1 sp a
  j0 = thread.join w0
  j1 = thread.join w1
  addr = i64.const 16384
  total = i64.atomic.load addr
  return total
  }
}
func (i64, i64) -> (i64) {
block 0 (sp: i64, arg: i64) {
  mk = ref.func 2
  z = i64.const 0
  gen = cont.new mk z
  a = i64.const 10
  s0, v0 = cont.resume gen a
  b = i64.const 20
  s1, v1 = cont.resume gen b
  addr = i64.const 16384
  rmw = i64.atomic.rmw.add addr v1
  zero = i64.const 0
  return zero
  }
}
func (i64, i64) -> (i64) {
block 0 (sp2: i64, arg2: i64) {
  one = i64.const 1
  bumped = i64.add arg2 one
  got = suspend bumped
  five = i64.const 5
  out = i64.add got five
  return out
  }
}
`,
  },

  // ---- on-ramp modules: real C/C++ guests, compiled through clang → temen-llvm and run as a
  //      pre-built .temen via `temen_run_onramp` (no in-browser parse). Built by
  //      `build-onramp-assets.mjs` at `--host-page 65536` (the wasm page). ------------------------
  'hello (C → Temen)': {
    kind: 'module',
    jit: true, // _start is wasm-JIT-emittable (proven byte-identical by browser-jit-module-test)
    url: './assets/hello_c.temen',
    mode: 'io',
    desc: 'crates/temen-run/demos/hello.c — a C program compiled with stock clang, translated by the ' +
      'LLVM on-ramp, and run through the powerbox: it write(1, …)s a greeting and exits. The output ' +
      'below is the guest’s real stdout. Toggle "wasm-JIT" to run the whole program (_start) on ' +
      'emitted wasm instead of the interpreter — "Prove interp ≡ JIT" checks the stdout matches.',
  },
  'gradient (C → framebuffer)': {
    kind: 'module',
    url: './assets/gradient.temen',
    mode: 'io',
    desc: 'crates/temen-run/demos/display/gradient.c — a C guest renders a 128×128 RGBA image and ' +
      'presents one frame through the `display` capability (resolved by name, like Lua’s io / ' +
      'SQLite’s VFS). The host reads the frame out of guest memory and blits it to the canvas on the ' +
      'right. This is the framebuffer output path the graphical demos (Doom) ride.',
  },
  'bounce (interactive — arrow keys)': {
    kind: 'reactor',
    jit: true, // tick() is wasm-JIT-emittable (proven byte-identical by browser-jit-reactor-test)
    url: './assets/bounce.temen',
    jit: true, // tick() emits after cap-call outlining — toggle "wasm-JIT" to run it near-natively
    mode: 'io',
    desc: 'crates/temen-run/demos/display/bounce.c — a C guest whose exported tick() runs one frame. ' +
      'Click Run, then steer the box with the arrow keys: the page calls tick() once per animation ' +
      'frame (the reactor run model), feeding key events in through the `keyboard` capability and ' +
      'blitting the frame it presents through `display`. State persists between frames. This is the ' +
      'interactive per-frame loop + input path Doom rides. Toggle "wasm-JIT" to run the whole tick() ' +
      'on emitted wasm instead of the interpreter. Click Stop to end the loop.',
  },
  'life (Conway — heap persistence)': {
    kind: 'reactor',
    jit: true, // tick() is wasm-JIT-emittable (proven byte-identical by browser-jit-reactor-test)
    url: './assets/life.temen',
    jit: true, // tick() emits after cap-call outlining — toggle "wasm-JIT" to run it near-natively
    mode: 'io',
    desc: 'crates/temen-run/demos/display/life.c — Conway’s Game of Life. Its cell grid lives in the ' +
      'malloc heap (which the on-ramp grows above the mapped window — exactly where Doom’s allocator ' +
      'will sit). Each tick computes the next generation from the current one, so the glider only ' +
      'advances if the reactor persists the guest’s whole memory (heap included) between frames. ' +
      'Click Run to watch it evolve; Stop to end. Toggle "wasm-JIT" to run the whole tick() on ' +
      'emitted wasm instead of the interpreter. This is the heap-persistence proof Doom needs.',
  },
  'Mandelbrot zoom (interactive — arrow keys)': {
    kind: 'reactor',
    jit: true, // tick() is wasm-JIT-emittable (proven byte-identical by browser-jit-reactor-test)
    url: './assets/mandelzoom.temen',
    jit: true, // f64 tick() emits after cap-call outlining — toggle "wasm-JIT" for a ~24× speedup
    mode: 'io',
    desc: 'crates/temen-run/demos/display/mandelzoom.c — a C guest whose exported tick() computes a ' +
      'full double-precision Mandelbrot for the current view (in the sandbox, on the CPU — no GPU) ' +
      'and presents the RGBA frame through the `display` capability. Click Run: it auto-zooms toward ' +
      'a seahorse valley with a cycling rainbow palette; steer the zoom target with the arrow keys. ' +
      'Every frame is a fresh ~43k-pixel escape-time render; on the wasm interpreter that runs at a ' +
      'few FPS, so toggle "wasm-JIT" to run the whole tick() on emitted wasm — the f64 escape loop ' +
      'then runs near-natively (~24× faster here) and the frame rate jumps (shown live). The compute ' +
      'is all guest code; only the finished frame crosses the capability boundary. Click Stop to end.',
  },
  'GPU: Mandelbrot zoom (WebGPU shader)': {
    kind: 'reactor',
    url: './assets/gpu_shader.temen',
    mode: 'io',
    webgpu: true,
    desc: 'crates/temen-run/demos/display/gpu_shader.c — a sandboxed C guest ships a WGSL fragment ' +
      'shader once through a `webgpu` capability, then asks the host to present a frame each tick. ' +
      'The Mandelbrot escape-time loop runs on the **GPU** (via the browser’s WebGPU / navigator.gpu), ' +
      'so it stays smooth at 640×480 while zooming into a seahorse valley — only the tiny (frame, w, h) ' +
      'scalars cross the capability boundary per frame, and the guest never holds a GPU pointer. ' +
      'Needs a WebGPU-capable browser (Chrome/Edge, recent Firefox). Click Stop to end.',
  },
  'DOOM (1993 — arrow keys, Ctrl fires)': {
    kind: 'reactor',
    url: './assets/doom.temen',
    wad: './assets/doom1.wad',
    jit: true, // the whole tick() is wasm-JIT-emittable — the "wasm-JIT" toggle runs it near-natively
    mode: 'io',
    desc: 'Shareware DOOM (via doomgeneric), compiled from id Software’s C through the LLVM on-ramp ' +
      'and run in the sandbox. Click Run: _start reads the IWAD through the `fs` capability and boots ' +
      'Doom’s whole engine, then the page calls the guest’s tick() once per animation frame (the ' +
      'reactor loop), blitting each 320×200 frame it presents through `display`. Arrow keys move, ' +
      'Ctrl fires, Space uses doors/switches, Esc/Enter drive the menus. The zone heap persists in ' +
      'the guest window between frames (slice 3a). Boot takes a few seconds on the wasm interpreter — ' +
      'the renderer is byte-exact to a native build (the §18 differential). Toggle "wasm-JIT" to run ' +
      'the whole tick() on emitted wasm (near-native) instead of the interpreter — it multiplies the ' +
      'frame rate (shown live). Click Stop to end.',
  },
  'Lua (5.4.7 — write & run)': {
    kind: 'module',
    warm: true, // issue #805: the two-phase `lua_snapshot` driver (warmup + eval_run) — init the Lua
    // runtime + editor libs once, snapshot, then restore-and-eval per Run. Runs on the snapshot worker
    // (pre-warmed off the main thread), so the first Run is instant like the QuickJS card.
    jit: true, // eval_run is also wasm-JIT-emittable; ticking "wasm-JIT" runs warm+JIT (near-native eval
    // over the restored snapshot), falling back to warm-interp if the eval declines.
    editable: true,
    lang: 'lua',
    url: './assets/lua_snapshot.temen',
    mode: 'io',
    desc: 'Lua 5.4.7 — its core (lexer, parser, GC, bytecode VM) plus the base/string/table/math/' +
      'coroutine/io/os libraries, compiled through the LLVM on-ramp. Edit the Lua on the left and ' +
      'click Run: your code is piped to the guest as stdin, evaluated, and its output appears below. ' +
      'Real Lua, running client-side in the sandbox. By default it uses a warm-runtime snapshot ' +
      '(pre-warmed on a worker at page load): the Lua runtime + libraries are initialized once, then ' +
      'every Run restores that warm image and evaluates only your code (each Run starts clean). Tick ' +
      '"wasm-JIT" to evaluate on emitted wasm over that same warm image (warm+JIT — near-native eval); ' +
      '"Prove interp ≡ JIT" checks the stdout is byte-identical on both tiers.',
    src: `-- Write Lua here, then click Run.
print("Hello from " .. _VERSION)

-- recursion
local function fib(n) return n < 2 and n or fib(n - 1) + fib(n - 2) end
local out = {}
for i = 1, 10 do out[i] = fib(i) end
print("fib(1..10):", table.concat(out, " "))

-- tables + sort
local t = { 5, 3, 8, 1, 9, 2, 7 }
table.sort(t)
print("sorted:", table.concat(t, ", "))

-- string.format + math
print(string.format("pi ~ %.4f, 255 in hex = 0x%X", math.pi, 255))

-- io.write (stdout via the Stream capability — no trailing newline)
io.write("counting: ")
for i = 1, 5 do io.write(i, " ") end
io.write("\\n")

-- coroutines: a lazy generator
local function squares(n)
  return coroutine.wrap(function()
    for i = 1, n do coroutine.yield(i * i) end
  end)
end
local sq = {}
for v in squares(6) do sq[#sq + 1] = v end
print("squares:", table.concat(sq, " "))
`,
  },
  'C compiler (chibicc → Temen — compile & run)': {
    kind: 'chibicc',
    debug: true, // source-level C debugging: tick "debug info (-g)", then Debug (see the debugger below)
    jit: true, // chibicc's _start emits to wasm (333 funcs; cap-call/float helpers bounce cross-tier) —
    //          toggle "wasm-JIT" to run the compile several× faster (byte-identical IR, gated by chibicc_jit.rs)
    editable: true,
    lang: 'c',
    url: './assets/chibicc.temen',
    mode: 'io',
    desc: 'A real C compiler — chibicc, itself compiled through the LLVM on-ramp — running client-side ' +
      'in the sandbox. Edit the C on the left and click Run: the page runs chibicc.temen over your source ' +
      '(seeded on an fs capability at /in.c), which emits Temen IR; the page then temen_parse-es that IR into ' +
      'a module and runs it. Your program’s output (what printf writes) appears in the pane, with the ' +
      'emitted Temen IR below it, and main()’s return value as the result. A libc ships as headers ' +
      '(seeded under /include) — #include <stdio.h>/<string.h>/<stdlib.h>/<ctype.h>/<math.h>/<assert.h>/' +
      '<limits.h>/<stddef.h>/<errno.h> as guest C over the powerbox’s ambient write, including %f/%e/%g float ' +
      'formatting (correctly rounded to the requested precision — not a bignum shortest-round-trip, so a few ' +
      'exact-tie roundings can differ from glibc). Split the editor into a multi-file project with `//// file: name` ' +
      'marker lines — the code above the first marker is /in.c, and it can #include "name" the sibling files ' +
      '(headers or extra .c, unity-build style). Compile a program and run it, entirely in the browser, on the Temen.',
    src: `// Write C here, then click Run. printf output shows in the pane on the
// right; the emitted Temen IR appears below it, and main()'s return is the result.
#include <stdio.h>

int main(void) {
  printf("Hello from C — compiled to Temen IR in your browser!\\n\\n");

  // A little numerical integration: estimate pi via the Leibniz series.
  double pi = 0.0;
  for (int k = 0; k < 20000; k++)
    pi += (k % 2 ? -4.0 : 4.0) / (2 * k + 1);

  printf("  pi  ~ %.6f   (%%e: %e)\\n", pi, pi);
  printf("  e   ~ %.10g\\n", 2.718281828459045);
  printf("  1/7 = %.4f,  2^0.5 rounds to %.3g\\n", 1.0 / 7.0, 1.41421356);
  return 0;
}
`,
  },
  'C source-level debugging (chibicc → Temen — breakpoints on C lines)': {
    kind: 'chibicc',
    debug: true,
    gOn: true, // debug info starts ON: this card is ready to Debug (the compiler emits chibicc's -g waist)
    bp: 8, // pre-place a breakpoint on the `acc += i;` line (0-based editor line 8)
    jit: false,
    editable: true,
    lang: 'c',
    url: './assets/chibicc.temen',
    mode: 'io',
    desc: 'Debug a **C program at source level**, entirely in the browser. chibicc compiles this C with ' +
      '`-g` (the DEBUGGING.md §6 debug-info waist — source lines + variable names), and the DAP debugger ' +
      'runs the emitted IR on the bytecode engine: set a breakpoint in the gutter on a C line, press ' +
      'Debug, and it stops on that C line with the C locals (i, acc) named in the Variables pane. This ' +
      'program **`printf`s** — the debugger runs it under the on-ramp **I/O powerbox**, so its output ' +
      'streams into the stdout pane as you step, and **reverse debugging rewinds the output** (the ' +
      'capability-input tape replays faithfully). Step / Continue / Step Back / Reverse all work; a ' +
      'breakpoint is pre-placed on the `acc += i;` line. Untick "debug info (-g)" to compile clean, ' +
      'faster IR (and disable the debugger).',
    src: `// Debug a C program with printf — the debugger runs it under the I/O powerbox,
// so its output streams into the pane and reverse rewinds it. A breakpoint is
// pre-placed on "acc += i;": press Debug, step, watch i/acc, then Continue.
#include <stdio.h>

int main(void) {
  int acc = 0;
  for (int i = 3; i > 0; i--) {
    acc += i;
    printf("i=%d, acc=%d\\n", i, acc);
  }
  printf("sum = %d\\n", acc);
  return 0;
}
`,
  },
  'chibicc compiles its own source (self-host → Temen)': {
    kind: 'selfhost',
    jit: true, // chibicc's _start emits to wasm; every cc1 TU (giants included) compiles in a few hundred ms
    editable: false,
    url: './assets/chibicc.temen',
    image: './assets/chibicc_selfhost.img',
    // The tractable cc1 TUs (SELFHOST_C.md); the giants (preprocess/parse/codegen_ir) are added next.
    tus: ['strings.c', 'hashmap.c', 'unicode.c', 'type.c', 'tokenize.c'],
    mode: 'io',
    desc: 'The self-host capstone (SELFHOST_C.md): **chibicc compiles its own source, in your browser.** ' +
      'Pick one of chibicc’s own cc1 translation units — its tokenizer, parser, type system — and click ' +
      'Run: chibicc.temen (itself a C compiler, compiled to Temen IR through the LLVM on-ramp) compiles that ' +
      'file in `--emit-object` mode into a linkable TEMEN-IR object, reading the ~96-file glibc header ' +
      'closure `chibicc.h` pulls from an in-memory filesystem seeded into the sandbox — no server, no ' +
      '/usr/include, all client-side. The emitted object is **byte-identical to a native chibicc build** ' +
      '(gated in CI). On the wasm-JIT even the 3400-line giants compile in a few hundred milliseconds; ' +
      '“Prove interp ≡ JIT” recompiles on both engines and checks the objects match to the byte.',
  },
  'nim (Nim → Temen, runs)': {
    kind: 'module',
    url: './assets/nim_hello.temen',
    mode: 'io',
    desc: "A **real Nim program** — `import std/syncio` / `write(stdout, \"hello, temen\\n\")` — compiled " +
      "all the way to a runnable Temen module and **run client-side in the sandbox**. The full nimony " +
      "toolchain (nifler → nimony → hexer) lowered the Nim to Leng, `temen-leng` translated + linked it " +
      "against the real compiled `system` module, and the nim→powerbox bridge wired its bottom edge to " +
      "the sandbox's caps (nimony's `write(fd,buf,len)` → the powerbox `write` stream). Click Run: the " +
      "output below is the guest's **real stdout** — a Nim program printing on the Temen. (The Nim→Leng " +
      "front end runs at build time for now, unlike the `temen-leng` card below, which runs the translator " +
      "itself in your browser; committed `nim_hello.temen`, gated by `nim_hello_asset.rs`.)",
  },
  'nifler: parse real Nim → NIF (nimony front-end, in your browser)': {
    kind: 'nifler',
    editable: true,
    lang: 'nim',
    url: './assets/nifler.temen.gz',
    mode: 'io',
    desc: "**Compile Nim in your browser** (NIM.md §3c/§3e slice 4): `nifler` — the *first real nimony " +
      "compiler phase* (Nim source → parsed NIF) — is itself a Nim program, on-ramped to a verified Temen " +
      "module through the LLVM/C on-ramp (slice 1), now **running client-side in the sandbox** over your " +
      "own code. Edit the Nim on the left and click Run: the page seeds it as `/in.nim` on an in-memory " +
      "`fs` cap, runs `nifler p /in.nim /out.p.nif`, and shows the `.p.nif` it emitted — the same real " +
      "nifler that parses Nim natively, **byte-identical to a native run** (gated by `nifler_asset.rs`). " +
      "This is the **front edge** of the toolchain (Nim → NIF), the complement to the `temen-leng` card " +
      "below (Leng → Temen IR); unlike the `nim (Nim → Temen, runs)` card above, whose front-end ran at " +
      "*build* time, here a front-end phase runs **in the browser**. The ~17.7 MB module ships gzipped " +
      "(~3.8 MB) and inflates client-side; the guest reaches only the seeded `fs` — no ambient authority. " +
      "No server, all in your browser, on the Temen.",
    src: `# Edit this Nim, then Run: the real nifler (nimony's parser, compiled to Temen)
# parses it into nimony's NIF — the first compiler phase, in your browser.
proc fib(n: int): int =
  if n < 2: n
  else: fib(n - 1) + fib(n - 2)

let xs = @[1, 2, 3]
for x in xs:
  echo fib(x)
`,
  },
  'nim: compile & run a whole Nim program → Temen (the full toolchain, in your browser)': {
    kind: 'nimc',
    editable: true,
    lang: 'nim',
    mode: 'io',
    // Four assets: the three phase guests (gzipped `.temen`) + the nimony stdlib image (gzipped).
    urls: {
      nifler: './assets/nifler.temen.gz',
      nimsem: './assets/nimsem.temen.gz',
      hexer: './assets/hexer.temen.gz',
      stdlib: './assets/nim_stdlib.img.gz',
    },
    desc: "**Compile a whole Nim program in your browser** (NIM.md §3c/§3e; #958) — the capstone of the " +
      "nimony-on-Temen slices. The `nifler` card above runs *one* phase (parse); this runs the **entire " +
      "nimony toolchain client-side**: the page plays nifmake itself — computes each module's cache stem " +
      "exactly as nimony does, crawls your program's `import` graph with `nifler`, then runs `nimsem` " +
      "(sema — itself spawning `nifler` as a sandboxed `exec` child over a shared in-memory `fs`) and " +
      "`hexer` (lower) over the whole closure, links the result through the nim→powerbox bridge with " +
      "`temen-leng`, and **runs `_start` under the powerbox**. Every phase is a verified Temen guest; the " +
      "stdlib is mounted from a committed `temen_fs` image. Edit the Nim on the left and click Run — the " +
      "output below is your program's **real stdout**, produced by a Nim program the Temen compiled and " +
      "ran end-to-end, no server. The default below shows a `proc`, a `string` parameter, and string " +
      "concatenation (`&`) all compiling through; the language conformance suite (#956) runs **15/15** " +
      "features end-to-end on the Temen — generics, exceptions, methods, closures, `seq`/`string`/`Table`, " +
      "floats, iterators, variant/`ref` objects, ARC destructors. (`echo` isn't an identifier nimony " +
      "resolves yet — a front-end gap; use `write(stdout, …)` for output. The four assets total ~6.5 MB " +
      "gzipped and inflate in-browser.)",
    src: `# Edit this Nim, then Run. The whole nimony toolchain compiles it in your
# browser — nifler (parse) -> nimsem (sema) -> hexer (lower) -> temen-leng
# (translate + link) — and the result runs on the Temen. The text below is
# your program's real stdout.
import std/syncio

proc greet(name: string): string =
  "hello, " & name & "\\n"

write(stdout, greet("Nim"))
write(stdout, greet("the Temen"))
`,
  },
  'temen-leng: translate real nimony Leng → Temen IR (self-host)': {
    kind: 'module',
    jit: true, // #1011: `compile_jit(Batch)` is WasmDriven with ~281/282 funcs emitted — the whole
    // translator runs as emitted wasm (the old "folds to the tree-walker" note conflated this with the
    // *native* Cranelift JIT's 64 MiB window cap, which the wasm-JIT doesn't have). Defaults to the
    // wasm-JIT tier; a decline/trap falls back to the interpreter, and "Prove interp ≡ JIT" checks parity.
    editable: true,
    lang: 'temen',
    url: './assets/temen-leng.temen',
    mode: 'io',
    desc: "The **leng self-host capstone** (NIM.md §3e): temen-leng — the Leng→TEMEN-IR translator, itself compiled to a verified Temen module through the LLVM on-ramp — **running client-side in the sandbox**. The editor holds a **real hexer Leng file** (verbatim `hexer c` output from Nim's `system/stringimpl` — string types, `=wasMoved`, ARC). Click Run: the page pipes it to `temen-leng.temen` on stdin, the translator parses the NIF and emits **Temen IR text** on stdout (shown below), and the run's exit code is the result (0 = ok, 2 = an unsupported/malformed Leng construct). The emitted IR is **byte-identical to running temen-leng natively** (gated in CI by `leng_selfhost_asset.rs`). By default the translator runs on the **wasm-JIT tier** (the whole `_start` emitted to wasm — #1011); untick *wasm-JIT* to compare the interpreter, or *Prove interp ≡ JIT* to check they agree byte-for-byte. Edit the Leng to translate your own — the same real translator, no server, all in your browser, on the Temen.",
    src: `(stmts
 (type :string.0. . (object . (fld :bytes.0 . (u 64)) (fld :more.0 . (ptr LongString.0.))))
 (type :LongString.0. . (object . (fld :fullLen.0 . (i +64)) (fld :rc.0 . (i +64)) (fld :capImpl.0 . (i +64)) (fld :data.0 . (uarray (c 8)))))
 (proc@,1g,nimony/lib/std/system/stringimpl.nim :=wasMoved.2.@5
  (params@H
   (param@1 :s.40 .
    (ptr@3 string.0.@4))).
  (pragmas@X
   (exportc "nimStrWasMoved")
   (inline 0 0)
   (smry~X
    (param 0 0 reads writes)))
  (stmts@2,1
   (asgn@8
    (dot~7
     (deref~1 s.40)bytes.0 0)
    (conv@2
     (u@B,c,nimony/lib/std/system/basic_types.nim 64)0))))
 (proc@,1j,nimony/lib/std/system/stringimpl.nim :=destroy.2.@5
  (params@G
   (param@1 :s.41 . string.0.@3)).
  (pragmas@S
   (exportc "nimStrDestroy")
   (inline 0 150)
   (smry~S writeGlobal readGlobal callsUnknown
    (param 0 0 reads writes escapes)))
  (stmts@2,1
   (if
    (elif@3
     (eq@9
      (conv~9,~13
       (i@U,E,nimony/lib/std/system/defaults.nim 64)
       (deref@1
        (cast
         (ptr@5
          (u@4 8))
         (addr@K
          (dot@1 s.41~G,13 bytes.0 0)))))
      (suf@1,~1Y 255 "i64"))
     (stmts~1,1
      (var@3 :\`x.223 .
       (bool@J,A,nimony/lib/std/system/arcops.nim)
       (call arcDec.0.
        (addr@D
         (dot
          (deref
           (dot~5 s.41~1 more.0 0))rc.0 0))))
      (if
       (elif@3 \`x.223
        (stmts~1,1
         (call dealloc.1.
          (conv@9
           (ptr@J,2f,nimony/lib/std/system/memory.nim
            (void))
           (dot s.41~1 more.0 0)))))))))))
 (proc@,1o,nimony/lib/std/system/stringimpl.nim :=copy.2.@5
  (params@D
   (param@1 :dest.11 .
    (ptr@6 string.0.@4))
   (param@J :src.6 . string.0.@5)).
  (pragmas@j
   (exportc "nimStrCopy")
   (smry~j writeGlobal readGlobal callsUnknown
    (param 0 0 reads writes escapes)
    (param 1 1 reads escapes)))
  (stmts@2,1
   (var@4 :ssrc.0 .
    (i@7,~1B 64)
    (conv~1,~18
     (i@U,E,nimony/lib/std/system/defaults.nim 64)
     (deref@1
      (cast
       (ptr@5
        (u@4 8))
       (addr@K
        (dot@1 src.6~8,18 bytes.0 0))))))
   (if@,1
    (elif@3
     (le@5 ssrc.0~5
      (suf@a,~1b 14 "i64"))
     (stmts~1,2
      (var@4 :sdest.0 .
       (i@5,~1E 64)
       (conv~3,~1B
        (i@U,E,nimony/lib/std/system/defaults.nim 64)
        (deref@1
         (cast
          (ptr@5
           (u@4 8))
          (addr@K
           (dot@1
            (deref~5,1B dest.11)bytes.0 0))))))
      (if@,1
       (elif@3
        (eq@6 sdest.0~6
         (suf@2,~1h 255 "i64"))
        (stmts~1,1
         (var@3 :\`x.224 .
          (bool@J,A,nimony/lib/std/system/arcops.nim)
          (call arcDec.0.
           (addr@G
            (dot
             (deref
              (dot~5
               (deref~4 dest.11)more.0 0))rc.0 0))))
         (if
          (elif@3 \`x.224
           (stmts~1,1
            (call dealloc.1.
             (conv@C
              (ptr@J,2f,nimony/lib/std/system/memory.nim
               (void))
              (dot
               (deref~4 dest.11)more.0 0)))))))))
      (call@,4 copyMem.0.
       (conv@D
        (ptr@P,O,nimony/lib/std/system/memory.nim
         (void))
        (addr
         (dot@4
          (deref~4 dest.11)bytes.0 0)))
       (conv@U
        (ptr@P,O,nimony/lib/std/system/memory.nim
         (void))
        (addr
         (dot@3 src.6~3 bytes.0 0)))
       (sizeof@l string.0.@1))))
    (else
     (stmts
      (stmts
       (stmts
        (stmts@2,9
         (if
          (elif@3
           (eq@B
            (addr~7
             (deref@1 dest.11))
            (addr@7 src.6@1))
           (stmts@P
            (ret .))))
         (var@4,1 :sdest.1 .
          (i@5,~1M 64)
          (conv~3,~1J
           (i@U,E,nimony/lib/std/system/defaults.nim 64)
           (deref@1
            (cast
             (ptr@5
              (u@4 8))
             (addr@K
              (dot@1
               (deref~5,1J dest.11)bytes.0 0))))))
         (if@,2
          (elif@3
           (eq@6 sdest.1~6
            (suf@2,~1p 255 "i64"))
           (stmts~1,1
            (var@3 :\`x.225 .
             (bool@J,A,nimony/lib/std/system/arcops.nim)
             (call arcDec.0.
              (addr@G
               (dot
                (deref
                 (dot~5
                  (deref~4 dest.11)more.0 0))rc.0 0))))
            (if
             (elif@3 \`x.225
              (stmts~1,1
               (call dealloc.1.
                (conv@C
                 (ptr@J,2f,nimony/lib/std/system/memory.nim
                  (void))
                 (dot
                  (deref~4 dest.11)more.0 0)))))))))
         (if@,5
          (elif@3
           (eq@5 ssrc.0~5
            (suf@3,~1s 255 "i64"))
           (stmts~1,1
            (call arcInc.0.
             (addr@F
              (dot
               (deref
                (dot~5 src.6~3 more.0 0))rc.0 0))))))
         (call@,7 copyMem.0.
          (conv@D
           (ptr@P,O,nimony/lib/std/system/memory.nim
            (void))
           (addr
            (dot@4
             (deref~4 dest.11)bytes.0 0)))
          (conv@U
           (ptr@P,O,nimony/lib/std/system/memory.nim
            (void))
           (addr
            (dot@3 src.6~3 bytes.0 0)))
          (sizeof@l string.0.@1)))))))))))
`,
  },
  'Shell (temen-posix — write & run a script)': {
    kind: 'shell',
    jit: false, // the shell carries Instantiator/SharedRegion call.caps → bytecode cooperative engine
    editable: true,
    lang: 'shell',
    url: './assets/shell.temen',
    // The shell's PATH registry: the __stage ring-filter runner (concurrent pipelines) and the `primes`
    // external command (a separate compiled-C program the shell exec's as an op-13 §14 child).
    cmds: [
      { name: '__stage', url: './assets/stage_runner.temen' },
      { name: 'primes', url: './assets/primes.temen' },
      { name: 'upper', url: './assets/upper.temen' },
    ],
    mode: 'io',
    desc: 'A real POSIX-style shell — a command interpreter compiled by the in-tree chibicc C ' +
      'compiler onto the temen-posix personality — running client-side in the sandbox. The same shell ' +
      'the differential test suite runs (crates/temen/tests/c_shell.rs), on the bytecode cooperative ' +
      'engine. Type a script on the left and click Run: it is fed to the shell as stdin and the ' +
      'output appears below. Builtins include echo (with $VARs), cd/pwd, cat/grep/wc/head/tail/sort/' +
      'uniq/ls, test/[ ], redirection (> >> <), command lists (; && ||), if/then/else, and globbing ' +
      '— all over an in-memory filesystem. Pipelines run as concurrent stages over shared-memory ' +
      'rings (op 11 + SharedRegion + futex); an unknown name like `primes` (a generator) or `upper` ' +
      '(a filter that reads stdin) is exec’d as a separate compiled-C program (op 13 §14 child) — all ' +
      'spawned client-side in the sandbox.',
    src: `# A real shell, running in the sandbox. Type commands, then click Run.
echo hello from the sandbox

# variables
NAME=world
echo hi $NAME

# primes and upper are not builtins — they are separate compiled-C programs
# the shell exec's as sandboxed children. primes generates; upper is a filter
# that reads its stdin.
echo -- primes up to 30 --
primes 30
echo -- upper (a stdin filter) --
echo shout this line | upper

# a file, then a concurrent ring pipeline: sort and dedupe run as separate
# stages, streaming over shared-memory rings with backpressure.
echo banana > fruits
echo apple >> fruits
echo banana >> fruits
echo cherry >> fruits
echo -- sorted, deduped --
cat fruits | sort | uniq

if test -f fruits; then echo fruits exists; fi
`,
  },
  'bash (real GNU bash 5.2, AOT-compiled)': {
    kind: 'bash',
    jit: false, // setjmp/longjmp + fork/exec run on the bytecode cooperative engine
    editable: true,
    lang: 'shell',
    url: './assets/bash.temen',
    // The /bin registry: the 13 repo-owned coreutils (chibicc-compiled command modules) bash
    // `fork → execve`s as separate sandboxed programs. Names are full paths — bash resolves
    // `seq` → `/bin/seq` through PATH=/bin.
    cmds: [
      { name: '/bin/true', url: './assets/bin_true.temen' },
      { name: '/bin/false', url: './assets/bin_false.temen' },
      { name: '/bin/echo', url: './assets/bin_echo.temen' },
      { name: '/bin/cat', url: './assets/bin_cat.temen' },
      { name: '/bin/seq', url: './assets/bin_seq.temen' },
      { name: '/bin/head', url: './assets/bin_head.temen' },
      { name: '/bin/wc', url: './assets/bin_wc.temen' },
      { name: '/bin/sort', url: './assets/bin_sort.temen' },
      { name: '/bin/uniq', url: './assets/bin_uniq.temen' },
      { name: '/bin/ls', url: './assets/bin_ls.temen' },
      { name: '/bin/pwd', url: './assets/bin_pwd.temen' },
      { name: '/bin/grep', url: './assets/bin_grep.temen' },
      { name: '/bin/tr', url: './assets/bin_tr.temen' },
    ],
    mode: 'io',
    desc: 'The real GNU bash 5.2 binary — compiled whole-program to LLVM bitcode, translated ' +
      'through the AOT on-ramp, and running client-side on the bytecode cooperative engine under ' +
      'the temen-posix personality (#802/#1080). Not a reimplementation: bash’s own parser, ' +
      'expansion, job control, setjmp/longjmp error paths, fork/exec/wait and pipes, with the ' +
      '13 coreutils in /bin run as separate compiled programs bash fork→execve’s in the ' +
      'sandbox. Edit the script and click Run: it executes as `bash -c ‘script’` and the ' +
      'captured stdout appears below. bash is GPLv3 and never committed — this card’s module ' +
      'is built at deploy (node build-bash-assets.mjs).',
    src: `# Real GNU bash, AOT-compiled, running in your browser's sandbox.
echo hello from bash $BASH_VERSION

# variables + arithmetic — bash's own expansion machinery
N=world
echo hi $N
echo $((6 * 7))

# external commands: each is a separate compiled program bash fork+execve's
# (seq, head, wc, sort, uniq live in /bin — type 'type seq' to see)
seq 5 | head -n 3

# pipelines run as real fork'd stages over capability pipes
printf 'banana\\napple\\nbanana\\n' | sort | uniq

for i in 1 2 3; do echo loop $i; done
if [ -n "$N" ]; then echo N is set; fi
type seq
`,
  },
  'SQLite (:memory: — write & run SQL)': {
    kind: 'module',
    jit: true, // _start is wasm-JIT-emittable (proven byte-identical by browser-jit-module-test)
    editable: true,
    lang: 'sql',
    url: './assets/sqlite_repl.temen',
    mode: 'io',
    desc: 'The unmodified SQLite 3.50.2 amalgamation (~257k lines of C), compiled through the LLVM ' +
      'on-ramp. Edit the SQL on the left and click Run: it executes against a fresh in-memory ' +
      'database (each Run starts clean) and prints result tables, change counts, and errors below. ' +
      'Real SQLite, running client-side in the sandbox. Toggle "wasm-JIT" to run the whole engine on ' +
      'emitted wasm (near-native); "Prove interp ≡ JIT" checks the stdout is byte-identical on both tiers.',
    src: `-- Write SQL here, then click Run. Each Run is a fresh :memory: database.
CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INT);
INSERT INTO users(name, age) VALUES ('Ada', 36), ('Alan', 41), ('Grace', 45), ('Edsger', 40);

SELECT name, age FROM users WHERE age >= 40 ORDER BY age DESC;

SELECT count(*) AS n, avg(age) AS avg_age, max(age) AS oldest FROM users;

-- a recursive CTE: the first 10 Fibonacci numbers
WITH RECURSIVE fib(n, a, b) AS (
  SELECT 1, 0, 1 UNION ALL SELECT n + 1, b, a + b FROM fib WHERE n < 10
)
SELECT n, a AS fib FROM fib;
`,
  },
  'JavaScript (QuickJS — write & run JS)': {
    kind: 'module',
    warm: true, // WASM_AOT.md warm-runtime snapshot: init the QuickJS runtime once, then restore that
    // warm image and eval-only per Run — the "trivial program takes >1s" fixed init is paid once, so
    // later Runs are ~milliseconds. Fresh-per-Run isolation is enforced in the engine (temen_warm_eval
    // restores the same post-warmup image each Run). Default path for this card.
    jit: true, // tick "wasm-JIT" for the **warm+JIT** tier (WASM_AOT.md): `eval_run` emitted to wasm and
    // run over the restored warm image — init stays paid-once, the eval runs near-native (the win is
    // compute-heavy JS; a trivial program is already ~instant on warm-interp). `runModule` routes a warm
    // card's JIT toggle to `runWarmJit` (not the cold `_start` path). See LLVM.md "Active target — QuickJS".
    editable: true,
    lang: 'js',
    url: './assets/qjs_snapshot.temen',
    mode: 'io',
    desc: 'Bellard\'s unmodified QuickJS 2024-01-13 — a full JavaScript engine (NaN-boxing, a bytecode ' +
      'VM with computed-goto dispatch, BigInt, regex, Unicode) compiled through the LLVM on-ramp. Edit ' +
      'the JS on the left and click Run: it evaluates in a fresh runtime (each Run starts clean), and ' +
      'prints anything you print()/console.log() plus the value of the last expression. Real QuickJS, ' +
      'running client-side in the sandbox — no ambient authority. By default it uses a warm-runtime ' +
      'snapshot: the first Run initializes the QuickJS runtime (~once), and every Run after restores ' +
      'that warm image and evaluates only your code — so a trivial program runs in milliseconds instead ' +
      'of rebuilding the whole engine each time. Tick "wasm-JIT" to evaluate on emitted wasm over that ' +
      'same warm image (warm+JIT — near-native eval, init still paid once); "Prove interp ≡ JIT" checks ' +
      'the stdout is byte-identical on both tiers.',
    src: `// Write JavaScript here, then click Run. Each Run is a fresh QuickJS runtime.
function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
console.log("fib(0..10):", Array.from({length: 11}, (_, i) => fib(i)).join(" "));

const xs = [5, 3, 8, 1, 9, 2, 7];
console.log("sorted:", xs.slice().sort((a, b) => a - b).join(","));

console.log("json:", JSON.stringify({ ok: true, nums: [1, 2, 3], nested: { pi: Math.PI } }));

// the completion value of the last expression is printed too:
"0.1 + 0.2 = " + (0.1 + 0.2);
`,
  },
  // Tcl — the reference Tcl 8.6 interpreter with FULL Tcl_Init: the standard script library (init.tcl,
  // clock, msgcat, …) is embedded and served through an in-guest Tcl_Filesystem VFS, so clock/file/
  // glob/auto_load/package all work with no filesystem capability. The `tcl_snapshot.temen` asset is built
  // by `build-onramp-assets.mjs`; runs byte-identical to native (`demo_tcl_init_stdin`).
  'Tcl (8.6 — write & run)': {
    kind: 'module',
    warm: true, // issue #805 follow-on: the two-phase `tcl_snapshot` driver (warmup = full Tcl_Init,
    // eval_run = eval-only). Runs on the snapshot worker (pre-warmed off the main thread), so the whole
    // Tcl_Init (sourcing init.tcl + the standard library) is paid once, not per Run.
    jit: true, // #865: warm+JIT drives the emitted `eval_run` (near-native eval over the restored warm
    // image) — tick "wasm-JIT" for it; init stays paid-once via the snapshot. (Previously declined: the
    // driver drove the cold `_start` export, whose Tcl_Init re-run trapped in encoding init — now fixed.)
    editable: true,
    lang: 'tcl',
    url: './assets/tcl_snapshot.temen',
    mode: 'io',
    desc: 'The reference Tcl 8.6.14 interpreter with the full standard library — its bytecode compiler ' +
      '+ execution engine, expr, string/list/dict, Henry Spencer regex, libtommath bignums, plus the ' +
      'script-library commands (clock, file, glob, auto_load, package) served from an embedded ' +
      'in-guest VFS. Edit the Tcl on the left and click Run: your script is piped to the guest as ' +
      'stdin, evaluated, and its output appears below. Real Tcl, running client-side in the sandbox. ' +
      'It uses a warm-runtime snapshot (pre-warmed on a worker at page load): Tcl_Init runs once, then ' +
      'every Run restores that warm image and evaluates only your script (each Run starts clean).',
    src: `# Write Tcl here, then click Run. The full standard library is available.
proc fib {n} { expr {$n < 2 ? $n : [fib [expr {$n-1}]] + [fib [expr {$n-2}]]} }
puts "fib(1..10): [lmap i {1 2 3 4 5 6 7 8 9 10} {fib $i}]"
puts "sorted:     [lsort -integer {5 3 8 1 9 2 7}]"

# clock, file, glob — the script library, from the embedded VFS (no real filesystem):
puts "clock:      [clock format 1000000000 -gmt 1 -format {%Y-%m-%d %H:%M:%S} ] UTC"
puts "file:       [file join /a b c]  ext=[file extension archive.tar.gz]"
puts [format "pi ~ %.4f, 255 = 0x%X, sqrt2 = %.6f" 3.14159265 255 [expr {sqrt(2)}]]

dict set d a 1; dict set d b 2
puts "dict:       $d"
puts "regexp:     [regexp -inline {(\\w+)@(\\w+)} user@host]"
puts [string toupper "tcl on temen"]
`,
  },
  'PostgreSQL (17.5 — write & run SQL)': {
    kind: 'pg',
    editable: true,
    lang: 'sql',
    url: './assets/postgres_resolved.temen',
    image: './assets/pgdata.img',
    mode: 'io',
    desc: 'A whole, unmodified PostgreSQL 17.5 --single backend — ~15,000 functions compiled LLVM → ' +
      'Temen IR, verified, and run on the bytecode interpreter inside wasm. Its data directory is an ' +
      'in-memory image mounted on a capability-scoped filesystem — no host filesystem, network, or ' +
      'ambient authority. It runs as a live **interactive session**: the first Run boots the backend ' +
      '(a few seconds), then each Run feeds your SQL to the *same* backend on its blocking stdin — so ' +
      'queries after the first are sub-second and state persists across them (a table you CREATE stays ' +
      'for the next query), exactly like psql. **Your database also survives a page reload:** after each ' +
      'query the data directory is snapshotted into your browser (IndexedDB), and the next visit boots ' +
      'from that snapshot — Postgres runs its own crash recovery over it. Run `\\reset` to wipe the ' +
      'saved database and start fresh; Stop just closes the live backend (Run reopens it). The two large ' +
      'artifacts (a ~20 MB module + a ~40 MB image) download once.',
    src: `-- Click Run to send this to the live backend. Run again with new SQL — the session persists
-- (the table below stays for later queries), and only the first Run pays the boot.
-- Your data survives a page reload too: reload, then Run to resume. Type \\reset + Run to wipe it.
CREATE TABLE t (x int, s text);
INSERT INTO t VALUES (1, 'one'), (2, 'two'), (3, 'three');
SELECT * FROM t WHERE x > 1 ORDER BY x DESC;
SELECT count(*), sum(x), avg(x) FROM t;
`,
  },
};

// Size the run's shared window from the source's `memory N` declaration (64 KiB minimum — the wasm
// page granularity §14 carves align to; 16 MiB cap keeps a typo from asking for the whole memory).
function winSizeOf(src) {
  const m = /^\s*memory\s+(\d+)/m.exec(src);
  const log2 = Math.min(Math.max(m ? Number(m[1]) : 16, 16), 24);
  return 1 << log2;
}

// ---- per-card run machinery ----------------------------------------------------------------------
// Each demo renders as a self-contained card (its own editor + controls + output). The run functions
// take that card's context `c`, so state never leaks between cards, and only one run is ever active at
// a time (a fresh Run supersedes any running reactor). `eng`/`run` are the shared wasm engine; `broken`
// latches when a threaded run is Stopped mid-flight (shared state may wedge → every card's Run disables).
let eng, run, aborter = null, broken = false;
let snapshotClient = null; // the snapshot worker (issue #804): owns warm-card sessions off the main thread
let engineReady = false; // set once the wasm engine loads; gates the per-card Debug/Run enablement
const cards = [];

const setState = (c, state, text) => { c.el.state.dataset.state = state; c.el.state.textContent = text; };
const logTo = (c, m) => { c.el.log.textContent += m + '\n'; };
const setEngineState = (state, text) => { const e = $('engine-state'); e.dataset.state = state; e.textContent = text; };

// Fetched `.temen` bytes, cached (a 6 MB SQLite module is worth not re-downloading on every Run).
const moduleCache = new Map();
async function fetchModule(url, onProgress) {
  if (moduleCache.has(url)) return moduleCache.get(url);
  // Resolve module URLs relative to this script (not the document), so they work under any base path
  // (origin root locally, `/<repo>/` on GitHub Pages).
  const resolved = new URL(url, import.meta.url);
  const r = await fetch(resolved);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  // Stream the body so the big downloads (SQLite ~6 MB, Postgres ~20 MB module + ~40 MB image) show
  // progress instead of a silent stall. Falls back to a one-shot read when there's no reader (or no
  // caller watching): Content-Length gives the percent, absent ⇒ a running byte count.
  if (onProgress && r.body && r.body.getReader) {
    const total = Number(r.headers.get('content-length')) || 0;
    const reader = r.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received, total);
    }
    const bytes = new Uint8Array(received);
    let off = 0;
    for (const ch of chunks) { bytes.set(ch, off); off += ch.length; }
    moduleCache.set(url, bytes);
    return bytes;
  }
  const bytes = new Uint8Array(await r.arrayBuffer());
  moduleCache.set(url, bytes);
  return bytes;
}

// A download-progress callback that reports into a card's status line. `label` is the file being
// fetched; `total` of 0 (no Content-Length) shows a running byte count instead of a percentage.
const fmtMB = (n) => (n / (1 << 20)).toFixed(1);
const onFetchProgress = (c, label) => (received, total) => {
  const pct = total ? ` ${Math.floor((received / total) * 100)}%` : '';
  const of = total ? ` (${fmtMB(received)}/${fmtMB(total)} MB)` : ` (${fmtMB(received)} MB)`;
  setState(c, 'running', `downloading ${label}…${pct}${of}`);
};
const baseName = (url) => url.split('/').pop();

// ---- run instrumentation: structured dev-console logging + stage timing ---------------------------
// Every Run threads a small *recorder* (`runStart` → `fetchTimed`/`runStage`/`runNote`/`runTier` →
// `runEnd`). It feeds two audiences: the human-facing status line + in-page log pane (unchanged), and a
// rich record in the browser dev console — a `▶ start` line naming the demo, its kind, and the tier it
// begins on (interpreter vs wasm-JIT), then a grouped summary with the per-stage split (fetch / compile
// / encode / run), byte sizes, what came from cache, the tier actually used (with any wasm-JIT →
// interpreter fallback called out), status, and result. This is pure UI — it touches no authority and no
// sandbox surface — so it stays well outside the TCB the verifier guards.
const clockNow = () => performance.now();

// Begin a run record. `fields` carries whatever is known up front (`tier`, `mode`). Logs the start line
// (so the console shows a run *beginning* even if it later hangs) and returns the recorder to thread on.
function runStart(c, fields = {}) {
  const rec = {
    demo: c.name,
    kind: c.ex.kind || 'temen-text',
    t0: clockNow(),
    last: clockNow(),
    stages: [],   // [{ stage, ms }] — the split we print so "where the time went" is visible
    assets: [],   // [{ name, bytes, cached }] — every fetched .temen + whether it was a cache hit
    notes: {},    // free-form extras merged into the summary (workers, frames, sizes, …)
    ...fields,
  };
  console.info(
    `▶ [Temen playground] ${rec.demo} — start · ${rec.kind}` +
    `${rec.mode ? ` · ${rec.mode}` : ''} · ${rec.tier || 'interpreter'}`);
  return rec;
}

// Record a stage: the ms since the previous mark, or an explicit duration when the caller already timed
// it. Returns the ms so callers can still fold it into the status line.
function runStage(rec, stage, ms) {
  if (!rec) return ms;
  const dur = ms == null ? clockNow() - rec.last : ms;
  rec.last = clockNow();
  rec.stages.push({ stage, ms: +Number(dur).toFixed(1) });
  return dur;
}

// Attach free-form key/values (result, worker count, frames, …) to the summary.
function runNote(rec, obj) { if (rec) Object.assign(rec.notes, obj); return rec; }

// Note the tier actually used; if it differs from the requested tier, record the fallback for the summary.
function runTier(rec, tier) {
  if (!rec) return rec;
  if (rec.tier && rec.tier !== tier) rec.notes.fallback = `${rec.tier} → ${tier}`;
  rec.tier = tier;
  return rec;
}

// Finish a run: emit the grouped console summary in one place — total time, the per-stage split, cache
// hits, the tier (+ any fallback), asset sizes, status, and result. `ok` picks the ✓/✗ glyph and whether
// the group starts collapsed (success) or open (failure, so it's visible without a click).
function runEnd(rec, { ok = true, status, result } = {}) {
  if (!rec) return 0;
  const total = clockNow() - rec.t0;
  const label =
    `${ok ? '✓' : '✗'} [Temen playground] ${rec.demo} — ${ok ? 'done' : 'FAILED'}` +
    ` · ${rec.tier || 'interpreter'}${rec.notes.fallback ? ' (fallback)' : ''} · ${total.toFixed(1)}ms`;
  (ok ? console.groupCollapsed : console.group).call(console, label);
  console.log('demo:', rec.demo, '· kind:', rec.kind, rec.mode ? `· mode: ${rec.mode}` : '');
  console.log('tier:', rec.tier || 'interpreter', rec.notes.fallback ? `· fallback: ${rec.notes.fallback}` : '');
  if (status !== undefined) console.log('status:', status);
  if (result !== undefined) console.log('result:', result);
  if (rec.stages.length) { console.log(`stage split (total ${total.toFixed(1)}ms):`); console.table(rec.stages); }
  else console.log('total:', `${total.toFixed(1)}ms`);
  if (rec.assets.length) { console.log('assets:'); console.table(rec.assets); }
  const extra = { ...rec.notes }; delete extra.fallback;
  if (Object.keys(extra).length) console.log('detail:', extra);
  console.groupEnd();
  return total;
}

// Fetch a module through the shared cache, timing the download and recording its size + whether it was
// already cached onto the recorder. Called inside each run's existing try/catch (it throws on a fetch
// failure, which the caller already surfaces). Centralizes the "what was from cache" logging the console
// summary reports.
async function fetchTimed(rec, c, url) {
  const cached = moduleCache.has(url);
  const t = clockNow();
  const bytes = await fetchModule(url, onFetchProgress(c, baseName(url)));
  if (rec) {
    runStage(rec, `fetch:${baseName(url)}`, clockNow() - t);
    rec.assets.push({ name: baseName(url), bytes: bytes.length, cached });
  }
  return bytes;
}

// Blit the framebuffer the last run presented (via the `display` capability) to this card's canvas.
// w/h of 0 ⇒ no frame: hide the canvas. Copies the RGBA out of wasm memory into a fresh
// Uint8ClampedArray (putImageData rejects a SharedArrayBuffer-backed view, and a later alloc could
// detach the buffer). The canvas' intrinsic size is the frame's; CSS scales it up (pixelated).
function presentFrame(c, w, h) {
  const canvas = c.el.canvas;
  if (!w || !h) { canvas.hidden = true; return; }
  const sp = eng.ex.temen_framebuffer_ptr();
  const sl = eng.ex.temen_framebuffer_len();
  const rgba = new Uint8ClampedArray(new Uint8Array(eng.memory.buffer).slice(sp, sp + sl));
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').putImageData(new ImageData(rgba, w, h), 0, 0);
  canvas.hidden = false;
}

// Read the captured stdout stash (a stable region, independent of the module buffer — safe to read
// after the module has been deallocated). Shared by the interpreter and wasm-JIT module paths.
const readModuleStdout = () =>
  new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(
    eng.ex.temen_stdout_ptr(), eng.ex.temen_stdout_ptr() + eng.ex.temen_stdout_len()));
const readModuleStderr = () =>
  new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(
    eng.ex.temen_stderr_ptr(), eng.ex.temen_stderr_ptr() + eng.ex.temen_stderr_len()));

// Inflate a gzip'd asset to a Uint8Array via the browser's built-in DecompressionStream (no library).
// Used by the nifler card, whose ~17.7 MB module ships gzipped (~3.8 MB) — see `runNifler`.
async function gunzip(bytes) {
  const ds = new DecompressionStream('gzip');
  const buf = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
  return new Uint8Array(buf);
}

// Run a pre-built on-ramp module single-shot on the interpreter: alloc a buffer, copy the module in
// (plus optional stdin), `temen_run_onramp` (the fixed §3e powerbox — stdout/stdin/exit/memory), read
// the captured stdout, free. Returns { rv, status, stdout }. No Workers (these guests are
// single-threaded), so it never touches the par.js shared-window path.
function moduleInterp(bytes, stdinBytes) {
  // Alloc both buffers *before* filling: temen_alloc may grow (detach) the linear memory, so take one
  // fresh view after all allocations and write into it.
  const p = eng.ex.temen_alloc(bytes.length);
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) stdinP = eng.ex.temen_alloc(stdinLen);
  const view = new Uint8Array(eng.memory.buffer);
  view.set(bytes, p);
  if (stdinP) view.set(stdinBytes, stdinP);
  const rv = eng.ex.temen_run_onramp(p, bytes.length, stdinP, stdinLen);
  const status = eng.ex.temen_status();
  const stdout = readModuleStdout();
  eng.ex.temen_dealloc(p, bytes.length);
  if (stdinP) eng.ex.temen_dealloc(stdinP, stdinLen);
  return { rv, status, stdout };
}

// ---- warm-runtime snapshot (WASM_AOT.md): init once, restore-per-Run for a two-phase on-ramp guest ----
// The engine holds ONE warm session (a Rust static: temen_warm_open/eval/close). `warmSessionUrl` tracks
// which module it's warmed for; a new temen_warm_open replaces any prior session, so we (re)open lazily
// only when the module URL changes. Fresh-per-Run isolation is enforced in the engine (each eval restores
// the same post-`warmup` image), so a `var` in one Run can't leak into the next — the card's
// "each Run starts clean" promise holds.
let warmSessionUrl = null;

// Ensure the warm session is open for `url`'s module `bytes` (runs the guest's `warmup` once and
// snapshots the post-init image). Returns true on success; false if the module isn't a warm-snapshot
// driver (no `warmup`/`eval_run` exports) or open traps — the caller then falls back to the cold path.
function ensureWarmSession(bytes, url) {
  if (warmSessionUrl === url) return true;
  const p = eng.ex.temen_alloc(bytes.length);
  new Uint8Array(eng.memory.buffer).set(bytes, p);
  const live = Number(eng.ex.temen_warm_open(p, bytes.length));
  eng.ex.temen_dealloc(p, bytes.length);
  if (live < 0 || eng.ex.temen_status() !== 0) {
    warmSessionUrl = null;
    return false;
  }
  warmSessionUrl = url;
  return true;
}

// Evaluate the user's source over the warm session — restore the snapshot + eval only, no runtime
// rebuild. Returns { rv, status, stdout }. Assumes ensureWarmSession succeeded for this module.
function warmEval(stdinBytes) {
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) {
    stdinP = eng.ex.temen_alloc(stdinLen);
    new Uint8Array(eng.memory.buffer).set(stdinBytes, stdinP);
  }
  const rv = Number(eng.ex.temen_warm_eval(stdinP, stdinLen));
  const status = eng.ex.temen_status();
  const stdout = readModuleStdout();
  if (stdinP) eng.ex.temen_dealloc(stdinP, stdinLen);
  return { rv, status, stdout };
}

// Pack a shell PATH registry — `[{ name, bytes }]` — into the blob `temen_run_shell` parses: a u32 entry
// count, then per entry u32 name-length + UTF-8 name + u32 module-length + module bytes (all
// little-endian). The `__stage` ring runner and every external command (`primes`, …) travel in one
// buffer. Returns null for an empty registry (the shell then runs bare).
function buildCmdsBlob(cmds) {
  if (!cmds || !cmds.length) return null;
  const enc = new TextEncoder();
  const parts = cmds.map((c) => ({ name: enc.encode(c.name), bytes: c.bytes }));
  let total = 4;
  for (const p of parts) total += 4 + p.name.length + 4 + p.bytes.length;
  const blob = new Uint8Array(total);
  const dv = new DataView(blob.buffer);
  let o = 0;
  dv.setUint32(o, parts.length, true); o += 4;
  for (const p of parts) {
    dv.setUint32(o, p.name.length, true); o += 4;
    blob.set(p.name, o); o += p.name.length;
    dv.setUint32(o, p.bytes.length, true); o += 4;
    blob.set(p.bytes, o); o += p.bytes.length;
  }
  return blob;
}

// Run the `temen-posix` **shell** single-shot on the bytecode cooperative engine, through the
// `temen_run_shell` entry (STAGE1.md) — it grants the POSIX personality and (when `cmdsBlob` is given)
// the shell's PATH registry: the `__stage` ring-filter runner and any external commands. The editor
// text feeds the shell's stdin as the script. With `__stage` registered, `cat f | sort | uniq`-style
// pipelines take the concurrent ring path (op 11 + SharedRegion + futex); external commands (`primes`)
// spawn as op-13 §14 children. Returns { rv, status, stdout }.
function shellInterp(bytes, stdinBytes, cmdsBlob) {
  const p = eng.ex.temen_alloc(bytes.length);
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) stdinP = eng.ex.temen_alloc(stdinLen);
  let cmdsP = 0;
  const cmdsLen = cmdsBlob ? cmdsBlob.length : 0;
  if (cmdsLen) cmdsP = eng.ex.temen_alloc(cmdsLen);
  const view = new Uint8Array(eng.memory.buffer);
  view.set(bytes, p);
  if (stdinP) view.set(stdinBytes, stdinP);
  if (cmdsP) view.set(cmdsBlob, cmdsP);
  const rv = eng.ex.temen_run_shell(p, bytes.length, stdinP, stdinLen, cmdsP, cmdsLen);
  const status = eng.ex.temen_status();
  const stdout = readModuleStdout();
  eng.ex.temen_dealloc(p, bytes.length);
  if (stdinP) eng.ex.temen_dealloc(stdinP, stdinLen);
  if (cmdsP) eng.ex.temen_dealloc(cmdsP, cmdsLen);
  return { rv, status, stdout };
}

// A card's Run for the shell: fetch the module (+ its PATH registry — the __stage ring runner and any
// external commands), feed the editor's script as stdin, run it, show the captured stdout. The shell
// runs on the bytecode cooperative engine (the wasm-safe interpreter tier that lowers its
// Instantiator/SharedRegion call.caps), so there is no JIT toggle.
async function runShell(c) {
  const ex = c.ex;
  // The shell carries Instantiator/SharedRegion call.caps, so it runs on the bytecode cooperative engine
  // (no wasm-JIT tier); the recorder still logs its fetch/run split + cache to the console.
  const rec = runStart(c, { tier: 'interpreter' });
  setState(c, 'running', 'fetching shell…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  let bytes;
  try {
    bytes = await fetchTimed(rec, c, ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    logTo(c, `fetch failed: ${e.message}`);
    runNote(rec, { fetchError: e.message });
    runEnd(rec, { ok: false });
    return;
  }
  logTo(c, `fetched ${ex.url}: ${bytes.length}B shell`);
  // The PATH registry (`ex.cmds`: `[{ name, url }]`) — the `__stage` ring runner (concurrent pipelines:
  // op 11 + SharedRegion + futex) and external commands like `primes`. Each is an optional companion
  // asset: a fetch failure drops just that command (pipelines fall back to memfs staging; a missing
  // external command is `not found`), so it is logged, not fatal.
  const cmds = [];
  for (const cmd of ex.cmds || []) {
    try {
      const cb = await fetchTimed(rec, c, cmd.url);
      cmds.push({ name: cmd.name, bytes: cb });
      logTo(c, `fetched ${cmd.url}: ${cb.length}B (${cmd.name})`);
    } catch (e) {
      logTo(c, `command '${cmd.name}' unavailable (${e.message})`);
    }
  }
  const cmdsBlob = buildCmdsBlob(cmds);
  // The editor holds the shell script — it is the shell's stdin. Ensure a trailing newline so the last
  // line runs (the read-eval loop acts on a completed line).
  let script = c.editor.getValue();
  if (!script.endsWith('\n')) script += '\n';
  const stdinBytes = new TextEncoder().encode(script);
  runNote(rec, { commands: cmds.map((x) => x.name), scriptBytes: stdinBytes.length });
  setState(c, 'running', 'running…');
  const t0 = performance.now();
  const { rv, status, stdout } = shellInterp(bytes, stdinBytes, cmdsBlob);
  const ms = runStage(rec, 'run:interpreter', performance.now() - t0).toFixed(0);
  c.el.stdout.textContent = stdout;
  c.el.result.textContent = `${rv}`;
  runNote(rec, { stdoutBytes: stdout.length });
  // 0 = OK, 5 = clean Exit (the `exit` builtin); anything else is a decode error / trap.
  if (status === 0 || status === 5) {
    setState(c, 'done', `done · status ${status} · ${ms}ms`);
    logTo(c, `shell run → ${rv} (status ${status}) in ${ms}ms`);
    runEnd(rec, { ok: true, status, result: rv });
  } else {
    setState(c, 'error', `run failed: status ${status} (1=decode 2=unsupported 3=trap)`);
    logTo(c, `shell run status ${status}`);
    runEnd(rec, { ok: false, status, result: rv });
  }
}

// Run **real GNU bash** single-shot on the bytecode cooperative engine, through the `temen_run_bash`
// entry (#1080) — it grants the POSIX personality (bash's fd 1/2, signals, fork/exec/wait) and, when
// `binsBlob` is given, registers each `/bin/<name>` module as a filesystem executable bash can
// `fork → execve`. The editor text runs as `bash -c '<script>'`. Returns { rv, status, stdout }.
function bashInterp(bytes, cmdBytes, binsBlob) {
  const p = eng.ex.temen_alloc(bytes.length);
  const cmdP = cmdBytes.length ? eng.ex.temen_alloc(cmdBytes.length) : 0;
  let binsP = 0;
  const binsLen = binsBlob ? binsBlob.length : 0;
  if (binsLen) binsP = eng.ex.temen_alloc(binsLen);
  const view = new Uint8Array(eng.memory.buffer);
  view.set(bytes, p);
  if (cmdP) view.set(cmdBytes, cmdP);
  if (binsP) view.set(binsBlob, binsP);
  const rv = eng.ex.temen_run_bash(p, bytes.length, cmdP, cmdBytes.length, 0, 0, binsP, binsLen);
  const status = eng.ex.temen_status();
  const stdout = readModuleStdout();
  eng.ex.temen_dealloc(p, bytes.length);
  if (cmdP) eng.ex.temen_dealloc(cmdP, cmdBytes.length);
  if (binsP) eng.ex.temen_dealloc(binsP, binsLen);
  return { rv, status, stdout };
}

// A card's Run for bash: fetch bash.temen (a deploy-built asset — GPLv3, never committed) + the /bin
// coreutils, run the editor's script as `bash -c`, show the captured stdout. Bytecode cooperative
// engine only (setjmp/longjmp + fork/exec are interpreter tiers), so no JIT toggle.
async function runBash(c) {
  const ex = c.ex;
  const rec = runStart(c, { tier: 'interpreter' });
  setState(c, 'running', 'fetching bash…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  let bytes;
  try {
    bytes = await fetchTimed(rec, c, ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message} — bash.temen is built at deploy (GPLv3, never committed): run \`node build-bash-assets.mjs\``);
    logTo(c, `fetch failed: ${e.message}`);
    runNote(rec, { fetchError: e.message });
    runEnd(rec, { ok: false });
    return;
  }
  logTo(c, `fetched ${ex.url}: ${bytes.length}B bash`);
  // The /bin registry — each coreutil is an optional companion asset: a fetch failure drops just
  // that command (bash then reports `not found` for it), so it is logged, not fatal.
  const bins = [];
  for (const cmd of ex.cmds || []) {
    try {
      const cb = await fetchTimed(rec, c, cmd.url);
      bins.push({ name: cmd.name, bytes: cb });
    } catch (e) {
      logTo(c, `command '${cmd.name}' unavailable (${e.message})`);
    }
  }
  logTo(c, `fetched /bin: ${bins.map((x) => x.name.slice(5)).join(' ')}`);
  const binsBlob = buildCmdsBlob(bins);
  const script = c.editor.getValue();
  const cmdBytes = new TextEncoder().encode(script);
  runNote(rec, { bins: bins.length, scriptBytes: cmdBytes.length });
  setState(c, 'running', 'running…');
  const t0 = performance.now();
  const { rv, status, stdout } = bashInterp(bytes, cmdBytes, binsBlob);
  const ms = runStage(rec, 'run:interpreter', performance.now() - t0).toFixed(0);
  const exitCode = eng.ex.temen_exit_code();
  c.el.stdout.textContent = stdout;
  c.el.result.textContent = `${exitCode}`;
  runNote(rec, { stdoutBytes: stdout.length, exitCode });
  // 0 = OK (a final external command exec'd on the root returns directly), 5 = clean Exit
  // (bash's exit_shell); anything else is a decode error / trap.
  if (status === 0 || status === 5) {
    setState(c, 'done', `done · exit ${exitCode} · ${ms}ms`);
    logTo(c, `bash run → exit ${exitCode} (status ${status}) in ${ms}ms`);
    runEnd(rec, { ok: true, status, result: rv });
  } else {
    setState(c, 'error', `run failed: status ${status} (1=decode 2=unsupported 3=trap)`);
    logTo(c, `bash run status ${status}`);
    runEnd(rec, { ok: false, status, result: rv });
  }
}

// A card's Run for an on-ramp module. The "wasm-JIT" toggle (offered on the emittable guests —
// hello_c/Lua/SQLite) emits the whole `_start` and runs it on wasm near-natively, servicing the ~7%
// cross-tier helpers through the interpreter; it falls back to the interpreter if the module isn't
// emittable (runJitModule throws). Both tiers share the fixed powerbox, so the stdout is identical.
async function runModule(c) {
  const ex = c.ex;
  setState(c, 'running', 'fetching module…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  const useJit = !!(ex.jit && c.el.jit && c.el.jit.checked);
  const rec = runStart(c, { tier: useJit ? 'wasm-JIT' : 'interpreter' });
  let bytes;
  try {
    bytes = await fetchTimed(rec, c, ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    logTo(c, `fetch failed: ${e.message}`);
    runNote(rec, { fetchError: e.message });
    runEnd(rec, { ok: false });
    return;
  }
  logTo(c, `fetched ${ex.url}: ${bytes.length}B module`);
  // An editable module reads the editor text as **stdin** (the guest evaluates it — e.g. Lua).
  let stdinBytes = null;
  if (ex.editable) {
    const enc = new TextEncoder().encode(c.editor.getValue());
    if (enc.length > 0) stdinBytes = enc;
  }
  runNote(rec, { moduleBytes: bytes.length, stdinBytes: stdinBytes ? stdinBytes.length : 0 });
  setState(c, 'running', `running…${useJit ? ' [wasm-JIT]' : ''}`);
  const t0 = performance.now();
  let rv = 0, status, tier = 'interpreter', stdout = '';
  // Warm cards run their snapshot on the dedicated **snapshot worker** (issue #804): the ~one-time warmup
  // and the eval happen off the main thread, so the UI never blocks. The worker was (usually) pre-warmed
  // on page load, so this is just an eval round-trip. A worker miss (not available, or the module isn't a
  // warm driver) leaves `status` undefined and falls through to the main-thread warm/JIT/interpreter path.
  if (ex.warm && snapshotClient) {
    try {
      const source = ex.editable ? c.editor.getValue() : '';
      if (!snapshotClient.isWarming(ex.url)) setState(c, 'running', 'warming up runtime (first Run)…');
      const r = await snapshotClient.evalWarm(ex.url, () => Promise.resolve(bytes), source, useJit);
      if (r.ok) {
        rv = r.value; status = r.status; stdout = r.stdout; tier = r.tier;
        // Observability hook (harmless): lets the browser test confirm the run actually went through the
        // worker rather than silently falling back to the main thread.
        globalThis.__snapshotWorkerRuns = (globalThis.__snapshotWorkerRuns || 0) + 1;
      } else {
        logTo(c, `snapshot worker: ${r.error}; falling back to the main thread`);
      }
    } catch (e) {
      logTo(c, `snapshot worker unavailable (${e.message}); falling back to the main thread`);
    }
  }
  if (status === undefined && useJit && ex.warm) {
    try {
      // Warm+JIT (WASM_AOT.md): evaluate the user's code on emitted wasm **over the restored warm image**
      // — the QuickJS runtime init stays paid-once (the snapshot), and the eval itself runs near-native.
      // The warm session must be open first (temen_warm_jit_open emits `eval_run` from it); a decline/trap
      // throws → we fall back to the interpreter warm path below. The compiled Module is cached under a
      // key distinct from the cold `_start` module (a different emit rooted at `eval_run`).
      const needOpen = warmSessionUrl !== ex.url;
      if (needOpen) setState(c, 'running', 'warming up runtime (first Run)…');
      if (!ensureWarmSession(bytes, ex.url)) throw new Error('warm session unavailable for this module');
      status = await runWarmJit(eng.ex, eng.memory, stdinBytes, `${ex.url}#eval`);
      rv = Number(eng.ex.temen_run_value());
      stdout = readModuleStdout();
      tier = 'warm+JIT';
    } catch (e) {
      logTo(c, `warm-JIT unavailable (${e.message}); falling back to the warm interpreter`);
      runNote(rec, { jitFallbackReason: e.message });
      status = undefined;
    }
  } else if (status === undefined && useJit) {
    try {
      // Emit `_start` and run it on wasm; temen_onramp_jit_run_finish captures stdout/exit/value into the
      // shared slots (read back via the usual accessors, exactly like the interpreter path). `temen_run_value`
      // is the guest's returned result — the same value `temen_run_onramp` returns on the interpreter, so the
      // result matches on both tiers (a trap throws → we fall back to the interpreter below).
      // Cache the compiled Module across Runs keyed by the module's content-addressed URL — the
      // emitted `_start` depends only on the module, not the editor `stdinBytes` (slice 1, WASM_AOT.md).
      status = await runJitModule(eng.ex, eng.memory, bytes, stdinBytes, ex.url);
      rv = Number(eng.ex.temen_run_value());
      stdout = readModuleStdout();
      tier = 'wasm-JIT';
    } catch (e) {
      logTo(c, `wasm-JIT module unavailable (${e.message}); falling back to the interpreter`);
      runNote(rec, { jitFallbackReason: e.message });
      status = undefined;
    }
  }
  if (status === undefined && ex.warm) {
    // Warm-runtime snapshot (the default for the QuickJS card): open the session once (the first Run
    // pays the ~one-time runtime init), then every Run restores the warm image and evaluates only.
    const needOpen = warmSessionUrl !== ex.url;
    if (needOpen) setState(c, 'running', 'warming up runtime (first Run)…');
    if (ensureWarmSession(bytes, ex.url)) {
      const r = warmEval(stdinBytes);
      rv = r.rv; status = r.status; stdout = r.stdout;
      tier = 'warm-snapshot';
    } else {
      logTo(c, 'warm-snapshot unavailable for this module; falling back to the interpreter');
      const r = moduleInterp(bytes, stdinBytes);
      rv = r.rv; status = r.status; stdout = r.stdout;
    }
  }
  if (status === undefined) {
    const r = moduleInterp(bytes, stdinBytes);
    rv = r.rv; status = r.status; stdout = r.stdout;
    // A framebuffer guest (gradient) presents through the interpreter path; the emittable JIT guests
    // above are stdout-only, so only the interpreter path blits a frame.
    presentFrame(c, eng.ex.temen_framebuffer_width(), eng.ex.temen_framebuffer_height());
  }
  runStage(rec, `run:${tier}`, performance.now() - t0);
  runTier(rec, tier);
  const ms = (performance.now() - t0).toFixed(0);
  c.el.stdout.textContent = stdout;
  c.el.result.textContent = `${rv}`;
  runNote(rec, { stdoutBytes: stdout.length });
  // 0 = OK, 5 = clean Exit; anything else is a decode error / trap / unsupported.
  if (status === 0 || status === 5) {
    setState(c, 'done', `done (${tier}) · status ${status} · ${ms}ms`);
    logTo(c, `module run (${tier}) → ${rv} (status ${status}) in ${ms}ms`);
    runEnd(rec, { ok: true, status, result: rv });
  } else {
    setState(c, 'error', `run failed: status ${status} (1=decode 2=unsupported 3=trap)`);
    logTo(c, `module run (${tier}) status ${status}`);
    runEnd(rec, { ok: false, status, result: rv });
  }
}

// The in-browser C compiler (SELFHOST_C.md §7 step 5) — two Temen passes in the sandbox:
//   1. compile: run `chibicc.temen` over the editor's C, seeded on an `fs` cap at `/in.c`
//      (`temen_run_onramp_fs`), and capture the emitted TEMEN-IR *text* on stdout;
//   2. encode + run: `temen_parse` that text into a module, then run it (`moduleInterp`) — the compiled
//      program's result is its `main` return value.
// Pass 1 (running chibicc) is the slow part, so it takes the "wasm-JIT" toggle: chibicc's whole
// `_start` emits to wasm (333 funcs; the cap-call/float helpers bounce cross-tier), running the compile
// several× faster than the bytecode interpreter — with a fallback to `temen_run_onramp_fs` if the emit is
// ever unavailable. Both tiers share the fixed powerbox + the same seeded memfs, so the emitted IR is
// byte-identical (gated by `chibicc_jit.rs`). Header-free sources compile with an empty header image;
// the demo corpus is return-value programs whose result is the value shown.
async function runChibicc(c) {
  const ex = c.ex;
  setState(c, 'running', 'fetching compiler…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  const useJit = !!(ex.jit && c.el.jit && c.el.jit.checked);
  const rec = runStart(c, { tier: useJit ? 'wasm-JIT' : 'interpreter' });
  let compiler;
  try {
    compiler = await fetchTimed(rec, c, ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    logTo(c, `fetch failed: ${e.message}`);
    runNote(rec, { fetchError: e.message });
    runEnd(rec, { ok: false });
    return;
  }
  const srcBytes = new TextEncoder().encode(c.editor.getValue());
  if (srcBytes.length === 0) { setState(c, 'error', 'empty source'); runEnd(rec, { ok: false }); return; }

  // Pass 1 — compile. On the wasm-JIT, hand the compiler + source to `runJitCompiler` (the cdylib seeds
  // the memfs + argv and emits `_start`); otherwise run chibicc on the bytecode interpreter. Both leave
  // the emitted IR text on the stdout stash. Alloc happens inside the JIT driver / just below.
  setState(c, 'running', `compiling…${useJit ? ' [wasm-JIT]' : ''}`);
  // `-g` iff the card's "debug info" checkbox is ticked (else clean, fast IR — see `temen_run_onramp_fs`).
  const gOn = c.el.gflag && c.el.gflag.checked ? 1 : 0;
  runNote(rec, { srcBytes: srcBytes.length, debugInfo: !!gOn });
  const tCompile = performance.now();
  let cstatus, compileTier = 'interpreter';
  if (useJit) {
    try {
      // The cdylib seeds the memfs + argv and emits `_start`; `gOn` selects the `-g` debug section.
      // chibicc's emitted `_start` is source-independent (the C source is fed via memfs, not baked
      // into the code), so cache it under a stable key — every compile reuses the compiled Module.
      cstatus = await runJitCompiler(eng.ex, eng.memory, compiler, srcBytes, gOn, 'chibicc-compiler');
      compileTier = 'wasm-JIT';
    } catch (e) {
      logTo(c, `wasm-JIT compile unavailable (${e.message}); falling back to the interpreter`);
      runNote(rec, { compileJitFallbackReason: e.message });
      cstatus = undefined;
    }
  }
  if (cstatus === undefined) {
    // Alloc both buffers before writing (temen_alloc may detach linear memory), pass an empty header
    // image (0,0), and run chibicc on the interpreter.
    const p = eng.ex.temen_alloc(compiler.length);
    const sp = eng.ex.temen_alloc(srcBytes.length);
    const view = new Uint8Array(eng.memory.buffer);
    view.set(compiler, p);
    view.set(srcBytes, sp);
    eng.ex.temen_run_onramp_fs(p, compiler.length, 0, 0, sp, srcBytes.length, gOn);
    cstatus = eng.ex.temen_status();
    eng.ex.temen_dealloc(p, compiler.length);
    eng.ex.temen_dealloc(sp, srcBytes.length);
  }
  const compileMs = runStage(rec, `compile:${compileTier}`, performance.now() - tCompile);
  // The headline tier is where the *compiler* ran (the wasm-JIT showcase); a wasm-JIT→interpreter note
  // here means the compile emit was unavailable. The compiled program always runs on the interpreter
  // oracle (pass 3), shown separately in the stage split.
  runTier(rec, compileTier);
  const ir = readModuleStdout();
  const cstderr = readModuleStderr();
  c.el.stdout.textContent = ir; // show the emitted Temen IR
  runNote(rec, { compileTier, irBytes: ir.length });
  logTo(c, `compiled (${compileTier}): ${srcBytes.length}B C → ${ir.length}B Temen IR in ${compileMs.toFixed(0)}ms (status ${cstatus})`);
  if ((cstatus !== 0 && cstatus !== 5) || ir.length === 0) {
    setState(c, 'error', `compile failed: status ${cstatus}${cstderr ? ` — ${cstderr.trim()}` : ''}`);
    runEnd(rec, { ok: false, status: cstatus });
    return;
  }

  // Pass 2 — encode the IR (temen_parse: parse + verify + encode) into a runnable module. Timed on its own
  // so the console split shows how much of "run" is really encode vs. execution.
  const tEncode = performance.now();
  const irBytes = new TextEncoder().encode(ir);
  const ip = eng.ex.temen_alloc(irBytes.length);
  new Uint8Array(eng.memory.buffer).set(irBytes, ip);
  const ok = eng.ex.temen_parse(ip, irBytes.length);
  const parsed = new Uint8Array(eng.memory.buffer).slice(
    eng.ex.temen_parse_ptr(), eng.ex.temen_parse_ptr() + eng.ex.temen_parse_len());
  eng.ex.temen_dealloc(ip, irBytes.length);
  if (ok !== 1) {
    setState(c, 'error', `encode failed: ${new TextDecoder().decode(parsed)}`);
    runEnd(rec, { ok: false });
    return;
  }
  runStage(rec, 'encode', performance.now() - tEncode);
  runNote(rec, { moduleBytes: parsed.length });

  // Pass 3 — run the compiled .temen artifact. It rides the wasm-JIT too (not just the compiler): the
  // runner now reports the guest's returned value (`temen_run_value`, matching the interpreter oracle) and
  // reports a trap as a trap — so `runJitModule` throws on a trap and we fall back to the interpreter,
  // which runs it correctly. A clean JIT run is byte-identical to the oracle (INVARIANT 9).
  const tRun = performance.now();
  let r, runTierName = 'interpreter';
  if (useJit) {
    try {
      const status = await runJitModule(eng.ex, eng.memory, parsed, null);
      r = { rv: Number(eng.ex.temen_run_value()), status, stdout: readModuleStdout() };
      runTierName = 'wasm-JIT';
    } catch (e) {
      logTo(c, `wasm-JIT run of compiled program declined (${e.message}); running it on the interpreter`);
      runNote(rec, { runJitDeclined: e.message });
    }
  }
  if (!r) r = moduleInterp(parsed, null);
  const runMs = runStage(rec, `run:${runTierName}`, performance.now() - tRun);
  runNote(rec, { runTier: runTierName, compileMs: +compileMs.toFixed(1), runMs: +runMs.toFixed(1), progStdoutBytes: (r.stdout || '').length });
  c.el.result.textContent = `${r.rv}`;
  // The stdout pane shows the compiled program's OUTPUT (what a `printf` writes through the
  // powerbox's ambient `write`), with the emitted Temen IR below it as a divider-separated section —
  // so both the payoff and "look, real IR" are visible. A pure return-value program shows just IR.
  const progOut = r.stdout || '';
  const irSection = `${'─'.repeat(18)} compiled to ${ir.length} B of Temen IR ${'─'.repeat(18)}\n${ir}`;
  c.el.stdout.textContent = progOut ? `${progOut}\n${irSection}` : irSection;
  // Status line now shows the compile/run split by tier, so "where the time went" is visible on-page.
  const split = `compile ${compileMs.toFixed(0)}ms (${compileTier}) · run ${runMs.toFixed(0)}ms (${runTierName})`;
  if (r.status === 0 || r.status === 5) {
    setState(c, 'done', `compiled & ran · returned ${r.rv} · ${split}`);
    logTo(c, `ran compiled program (${runTierName}) → ${r.rv} (status ${r.status}) in ${runMs.toFixed(0)}ms`);
    runEnd(rec, { ok: true, status: r.status, result: r.rv });
  } else {
    setState(c, 'error', `compiled program failed: status ${r.status} (1=decode 2=unsupported 3=trap)`);
    logTo(c, `compiled program status ${r.status}`);
    runEnd(rec, { ok: false, status: r.status, result: r.rv });
  }
}

// The **self-host** card (SELFHOST_C.md §7 step 5 — the capstone): chibicc compiles its *own* source in
// the browser. Fetch `chibicc.temen` + the committed closure image (its cc1 TU sources + the ~96-file
// glibc header closure `chibicc.h` pulls, + the self-host prelude), then run `chibicc.temen
// --emit-object <selected TU>` over that memfs — chibicc compiling its own tokenizer / parser / codegen
// into a linkable TEMEN-IR object, client-side. On the wasm-JIT every TU (giants included) compiles in a
// few hundred ms; a bytecode fallback covers a missing emit. The emitted object IR shows in the pane.
async function runSelfhost(c) {
  const ex = c.ex;
  setState(c, 'running', 'fetching compiler + sources…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  const useJit = !!(ex.jit && c.el.jit && c.el.jit.checked);
  const rec = runStart(c, { tier: useJit ? 'wasm-JIT' : 'interpreter' });
  let compiler, image;
  try {
    compiler = await fetchTimed(rec, c, ex.url);
    image = await fetchTimed(rec, c, ex.image);
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`node build-selfhost-assets.mjs\` to generate the closure image`);
    logTo(c, `fetch failed: ${e.message}`);
    runNote(rec, { fetchError: e.message });
    runEnd(rec, { ok: false });
    return;
  }
  // The chosen TU (its memfs-relative path) — the dropdown, or the first TU if the control is absent.
  const tu = c.el.tu ? c.el.tu.value : `frontend/chibicc/${ex.tus[0]}`;
  const tuBytes = new TextEncoder().encode(tu);
  const short = tu.split('/').pop();
  setState(c, 'running', `compiling ${short}…${useJit ? ' [wasm-JIT]' : ''}`);
  const gOn = c.el.gflag && c.el.gflag.checked ? 1 : 0;
  runNote(rec, { tu: short, debugInfo: !!gOn });
  const tCompile = performance.now();
  let cstatus, tier = 'interpreter';
  if (useJit) {
    try {
      cstatus = await runJitSelfhost(eng.ex, eng.memory, compiler, image, tuBytes, gOn, 'chibicc-selfhost');
      tier = 'wasm-JIT';
    } catch (e) {
      logTo(c, `wasm-JIT self-host unavailable (${e.message}); falling back to the interpreter`);
      runNote(rec, { jitFallbackReason: e.message });
      cstatus = undefined;
    }
  }
  if (cstatus === undefined) {
    // Alloc all three buffers before writing (temen_alloc may detach linear memory), then run on bytecode.
    const p = eng.ex.temen_alloc(compiler.length);
    const ip = eng.ex.temen_alloc(image.length);
    const tp = eng.ex.temen_alloc(tuBytes.length);
    const view = new Uint8Array(eng.memory.buffer);
    view.set(compiler, p);
    view.set(image, ip);
    view.set(tuBytes, tp);
    eng.ex.temen_selfhost_emit_object_fs(p, compiler.length, ip, image.length, tp, tuBytes.length, gOn);
    cstatus = eng.ex.temen_status();
    eng.ex.temen_dealloc(p, compiler.length);
    eng.ex.temen_dealloc(ip, image.length);
    eng.ex.temen_dealloc(tp, tuBytes.length);
  }
  const compileMs = runStage(rec, `compile:${tier}`, performance.now() - tCompile);
  runTier(rec, tier);
  const obj = readModuleStdout();
  const cstderr = readModuleStderr();
  const ms = compileMs.toFixed(0);
  runNote(rec, { objectBytes: obj.length });
  logTo(c, `self-host (${tier}): ${short} → ${obj.length}B object IR in ${ms}ms (status ${cstatus})`);
  if ((cstatus !== 0 && cstatus !== 5) || obj.length === 0) {
    c.el.stdout.textContent = obj;
    setState(c, 'error', `compile failed: status ${cstatus}${cstderr ? ` — ${cstderr.trim()}` : ''}`);
    runEnd(rec, { ok: false, status: cstatus });
    return;
  }
  const bar = '─'.repeat(12);
  c.el.stdout.textContent =
    `${bar} chibicc compiled its own ${short} → ${obj.length} B linkable TEMEN-IR object (${tier}) ${bar}\n${obj}`;
  c.el.result.textContent = `${obj.length} B`;
  setState(c, 'done', `compiled ${short} (${tier}) · ${obj.length} B object · ${ms}ms`);
  runEnd(rec, { ok: true, status: cstatus, result: `${obj.length} B object` });
}

// Compile Nim in the browser — the nimony **front-end** card (NIM.md §3c/§3e slice 4). Fetch
// `nifler.temen` (the first real nimony phase, Nim → parsed NIF, on-ramped to Temen), seed the editor's
// Nim as `/in.nim` on an in-memory `fs` cap, run `nifler p /in.nim /out.p.nif`, and show the `.p.nif`
// it emitted. Mirrors `runSelfhost` (memfs-seeded phase), but the output is a **file** nifler wrote (read
// back onto the stdout slot), not stdout text. Runs on the **wasm-JIT** first (#1011 slice 1 —
// `temen_run_nifler_jit_open` emits nifler's `_start`, its `.p.nif` read back after the run), falling back to
// the bytecode `temen_run_nifler_fs` when `_start` isn't wasm-drivable or the emitted run traps (INVARIANT 9).
async function runNifler(c) {
  const ex = c.ex;
  setState(c, 'running', 'fetching nifler…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  const rec = runStart(c, { tier: 'interpreter' });
  let compiler;
  try {
    // The asset ships **gzipped** (`nifler.temen.gz`, ~3.8 MB vs ~17.7 MB raw): fetch the compressed
    // bytes, then inflate them in the browser (DecompressionStream — no library) to the real module.
    const gz = await fetchTimed(rec, c, ex.url);
    compiler = await gunzip(gz);
    logTo(c, `nifler.temen.gz: ${gz.length}B → ${compiler.length}B module (inflated)`);
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`bash ../crates/temen-run/demos/nifler_temen/build_nifler_temen.sh\` to generate it`);
    logTo(c, `fetch/inflate failed: ${e.message}`);
    runNote(rec, { fetchError: e.message });
    runEnd(rec, { ok: false });
    return;
  }
  const srcBytes = new TextEncoder().encode(c.editor.getValue());
  runNote(rec, { moduleBytes: compiler.length, srcBytes: srcBytes.length });
  setState(c, 'running', 'parsing Nim…');
  const t0 = performance.now();
  // Try the **wasm-JIT** first (#1011 slice 1): emit nifler's `_start` and run the parse on emitted wasm,
  // with the produced `.p.nif` read back on the stdout slot (`temen_run_nifler_jit_open` → `driveJitRun`).
  // A decline (STATUS_UNSUPPORTED, e.g. `_start` not wasm-drivable) or an emitted-run trap throws → we
  // fall back to the bytecode `temen_run_nifler_fs` below, so the result matches on both tiers (INVARIANT 9).
  let status, tier;
  try {
    status = await runJitNifler(eng.ex, eng.memory, compiler, srcBytes, ex.url);
    tier = 'wasm-JIT';
  } catch (e) {
    logTo(c, `wasm-JIT nifler unavailable (${e.message}); falling back to the interpreter`);
    runNote(rec, { jitFallbackReason: e.message });
    status = undefined;
  }
  if (status === undefined) {
    // Alloc both buffers before writing (temen_alloc may detach linear memory), then run on bytecode.
    const p = eng.ex.temen_alloc(compiler.length);
    const sp = eng.ex.temen_alloc(srcBytes.length);
    const view = new Uint8Array(eng.memory.buffer);
    view.set(compiler, p);
    view.set(srcBytes, sp);
    Number(eng.ex.temen_run_nifler_fs(p, compiler.length, sp, srcBytes.length));
    status = eng.ex.temen_status();
    eng.ex.temen_dealloc(p, compiler.length);
    eng.ex.temen_dealloc(sp, srcBytes.length);
    tier = 'interpreter';
  }
  const ms = runStage(rec, `parse:${tier}`, performance.now() - t0).toFixed(0);
  runTier(rec, tier);
  const nif = readModuleStdout();
  const nstderr = readModuleStderr();
  runNote(rec, { nifBytes: nif.length });
  logTo(c, `nifler parse (${tier}) → ${nif.length}B .p.nif (status ${status}) in ${ms}ms`);
  // 0 = OK, 5 = clean Exit. A parse error (or a trap) leaves no `.p.nif`; show the guest's stderr.
  if ((status !== 0 && status !== 5) || nif.length === 0) {
    c.el.stdout.textContent = nstderr || nif;
    setState(c, 'error', `parse failed: status ${status}${nstderr ? ` — ${nstderr.trim().split('\n')[0]}` : ''}`);
    runEnd(rec, { ok: false, status });
    return;
  }
  const bar = '─'.repeat(12);
  c.el.stdout.textContent =
    `${bar} nifler parsed your Nim → ${nif.length} B .p.nif (nimony's NIF, on the Temen, ${tier}) ${bar}\n${nif}`;
  c.el.result.textContent = `${nif.length} B`;
  setState(c, 'done', `parsed Nim → ${nif.length} B .p.nif (${tier}) · ${ms}ms`);
  runEnd(rec, { ok: true, status, result: `${nif.length} B .p.nif` });
}

// Compile a **whole Nim program** in the browser — the nimony toolchain capstone (NIM.md §3c/§3e;
// #958). Fetch the three phase guests (`nifler`/`nimsem`/`hexer`, gzipped) + the stdlib image
// (gzipped), inflate them, then hand the editor's Nim as `<main>.nim` to `temen_compile_nim_fs`: the
// cdylib plays nifmake (computes stems, crawls the `import` graph with nifler), runs nimsem + hexer
// over the closure (nimsem spawning nifler through a wasm-native `exec` cap over the shared memfs),
// links through the nim→powerbox bridge, and runs `_start` under the powerbox. The program's real
// **stdout** comes back on the module stdout slot; a compile/link/run failure lands on stderr. All
// phases run on the bytecode engine (the ~hundreds-of-func Nim guests fold to the tree-walker); no JIT.
async function runNimc(c) {
  const ex = c.ex;
  setState(c, 'running', 'fetching toolchain…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  const rec = runStart(c, { tier: 'interpreter' });
  // Lazily fetch + inflate the four assets (each ships **gzipped**: phase `.temen` are ~3–17 MB raw, the
  // stdlib image ~2.4 MB; DecompressionStream inflates in-browser, no library). In the worker path this
  // runs **once** — the worker caches the guests — so re-Runs ship only the source, not ~28 MB again.
  const getAssets = async () => {
    const [gn, gs, gh, gl] = await Promise.all([
      fetchTimed(rec, c, ex.urls.nifler),
      fetchTimed(rec, c, ex.urls.nimsem),
      fetchTimed(rec, c, ex.urls.hexer),
      fetchTimed(rec, c, ex.urls.stdlib),
    ]);
    const [nifler, nimsem, hexer, stdlib] = await Promise.all([gunzip(gn), gunzip(gs), gunzip(gh), gunzip(gl)]);
    logTo(c, `inflated: nifler ${nifler.length}B · nimsem ${nimsem.length}B · hexer ${hexer.length}B · stdlib ${stdlib.length}B`);
    return { nifler, nimsem, hexer, stdlib };
  };
  const main = 'prog.nim';
  const source = c.editor.getValue();
  setState(c, 'running', 'compiling Nim (nifler → nimsem → hexer → link → run)…');
  const t0 = performance.now();
  let status, out, err;
  try {
    if (snapshotClient) {
      // Off the main thread (issue #1005): the whole toolchain runs on the snapshot worker's own engine,
      // so the ~1–3 min compile — or a runaway compiled guest that never returns — stalls only that
      // worker, never the page. A fresh Run first `cancelNim`s any still-running one (terminates the
      // stuck worker), so the card is never wedged by a previous hang.
      snapshotClient.cancelNim();
      const r = await snapshotClient.nimCompile(getAssets, source, main);
      if (!r.ok) throw new Error(r.error || 'nim worker unavailable');
      ({ status } = r);
      out = r.stdout;
      err = r.stderr;
    } else {
      // Fallback: no worker (e.g. the page lacks cross-origin isolation) — run on the main thread. This
      // freezes the tab for the duration, but keeps the card working where a worker can't be spawned.
      const { nifler, nimsem, hexer, stdlib } = await getAssets();
      const srcBytes = new TextEncoder().encode(source);
      const mainBytes = new TextEncoder().encode(main);
      // Alloc every buffer before writing any (temen_alloc may grow/detach linear memory), then take one
      // fresh view and fill them.
      const np = eng.ex.temen_alloc(nifler.length);
      const smp = eng.ex.temen_alloc(nimsem.length);
      const hp = eng.ex.temen_alloc(hexer.length);
      const ip = eng.ex.temen_alloc(stdlib.length);
      const sp = eng.ex.temen_alloc(srcBytes.length);
      const mp = eng.ex.temen_alloc(mainBytes.length);
      const view = new Uint8Array(eng.memory.buffer);
      view.set(nifler, np);
      view.set(nimsem, smp);
      view.set(hexer, hp);
      view.set(stdlib, ip);
      view.set(srcBytes, sp);
      view.set(mainBytes, mp);
      eng.ex.temen_compile_nim_fs(
        np, nifler.length, smp, nimsem.length, hp, hexer.length,
        ip, stdlib.length, sp, srcBytes.length, mp, mainBytes.length);
      status = eng.ex.temen_status();
      eng.ex.temen_dealloc(np, nifler.length);
      eng.ex.temen_dealloc(smp, nimsem.length);
      eng.ex.temen_dealloc(hp, hexer.length);
      eng.ex.temen_dealloc(ip, stdlib.length);
      eng.ex.temen_dealloc(sp, srcBytes.length);
      eng.ex.temen_dealloc(mp, mainBytes.length);
      out = readModuleStdout();
      err = readModuleStderr();
    }
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`bash ../crates/temen-run/demos/nim_e2e_chain/build_e2e_chain.sh\` to build the phase guests + stdlib image`);
    logTo(c, `compile failed: ${e.message}`);
    runNote(rec, { fetchError: e.message });
    runEnd(rec, { ok: false });
    return;
  }
  const ms = runStage(rec, 'compile+run:interpreter', performance.now() - t0).toFixed(0);
  runTier(rec, 'interpreter');
  logTo(c, `compile+run → status ${status}, ${out.length}B stdout in ${ms}ms`);
  // 0 = OK, 5 = clean Exit. Any other status: a phase/link/run failure — show the diagnostic (stderr).
  if (status !== 0 && status !== 5) {
    c.el.stdout.textContent = err || out;
    setState(c, 'error', `compile failed: status ${status}${err ? ` — ${err.trim().split('\n')[0]}` : ''}`);
    runEnd(rec, { ok: false, status });
    return;
  }
  const bar = '─'.repeat(10);
  c.el.stdout.textContent =
    `${bar} your Nim, compiled by the Temen (nifler → nimsem → hexer → temen-leng) and run — stdout ${bar}\n${out}`;
  c.el.result.textContent = `${out.length} B stdout`;
  setState(c, 'done', `compiled + ran your Nim · ${out.length} B stdout · ${ms}ms`);
  runEnd(rec, { ok: true, status, result: `${out.length} B stdout` });
}

// Boot PostgreSQL `--single` single-shot on the main engine (the `temen_run_pg` entry): fetch the
// pre-translated+resolved module + the data image, feed the editor's SQL as stdin, mount the image on
// an in-memory `fs` cap, run to a queried backend, read the captured stdout. A fresh boot per Run.
// Read the engine's captured stdout buffer (the `temen_pg_*` delta, or a `temen_run_pg` full capture).
function readEngineStdout() {
  const p = eng.ex.temen_stdout_ptr();
  const l = eng.ex.temen_stdout_len();
  return new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + l));
}

// The Postgres session's stdout, with `postgres --single`'s raw debug-tuple result blocks reformatted
// into psql-style aligned tables (`pg-format.js`). Prompts/banner/notices pass through untouched.
function readPgStdout() {
  return formatPgOutput(readEngineStdout());
}

// ---- persistent Postgres storage (IndexedDB) -----------------------------------------------------
// The live backend's data dir is an in-memory `mem_fs`; on its own it evaporates when the page unloads.
// After each query we snapshot that fs (`temen_pg_snapshot` → an `temen_fs` data image) and stash the image
// in IndexedDB; the next session boots from the saved image instead of the pristine one — so a table you
// CREATE (and its rows) survive a full page reload, recovered by Postgres' own startup recovery over the
// snapshot. Keyed per module URL so distinct builds don't collide. All best-effort: any storage failure
// just logs and the session keeps running in memory.
const PG_DB = 'temen-pg';
const PG_STORE = 'sessions';
const pgKey = (c) => c.ex.url;
function pgIdb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(PG_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(PG_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function pgLoad(key) {
  try {
    const db = await pgIdb();
    return await new Promise((resolve, reject) => {
      const r = db.transaction(PG_STORE, 'readonly').objectStore(PG_STORE).get(key);
      r.onsuccess = () => resolve(r.result || null);
      r.onerror = () => reject(r.error);
    });
  } catch {
    return null; // no IndexedDB (private mode, etc.) ⇒ fall back to the pristine image
  }
}
async function pgSave(key, bytes) {
  const db = await pgIdb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(PG_STORE, 'readwrite');
    tx.objectStore(PG_STORE).put(bytes, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
async function pgClear(key) {
  try {
    const db = await pgIdb();
    await new Promise((resolve) => {
      const tx = db.transaction(PG_STORE, 'readwrite');
      tx.objectStore(PG_STORE).delete(key);
      tx.oncomplete = resolve;
      tx.onerror = resolve;
    });
  } catch {
    /* nothing to clear */
  }
}
// Snapshot the live session's data dir and persist it, **coalescing** concurrent saves: at most one IDB
// write is in flight; a query that lands mid-write just marks the card dirty and re-saves once it drains
// (so a burst of queries collapses to one trailing write of the latest state). The snapshot bytes are
// copied straight out of wasm memory before the async write, so a later `memory.grow` can't detach them.
function persistPg(c) {
  if (!c.pgSession) return;
  let bytes;
  try {
    if (eng.ex.temen_pg_snapshot() !== 0) return;
    const p = eng.ex.temen_pg_snapshot_ptr();
    const l = eng.ex.temen_pg_snapshot_len();
    if (!p || !l) return;
    bytes = new Uint8Array(eng.memory.buffer, p, l).slice(); // detach from the wasm buffer
  } catch (e) {
    logTo(c, `snapshot failed: ${e.message}`);
    return;
  }
  if (c.pgSaving) {
    c.pgDirty = true;
    return;
  }
  c.pgSaving = true;
  pgSave(pgKey(c), bytes)
    .catch((e) => logTo(c, `session save failed: ${e.message}`))
    .finally(() => {
      c.pgSaving = false;
      if (c.pgDirty) {
        c.pgDirty = false;
        persistPg(c);
      }
    });
}

// PostgreSQL as a **live interactive session** (the `temen_pg_open`/`_query`/`_close` path): the first Run
// boots one `postgres --single` backend to the `backend>` prompt (a few seconds) and leaves it suspended
// on its blocking stdin; every Run after feeds the editor's SQL to that *same* backend and resumes it to
// the next prompt — so queries are sub-second and state persists across them. The output pane is a
// running transcript; Stop closes the session (`stopDemo`), and the next Run boots fresh.
async function runPg(c) {
  const ex = c.ex;
  // Postgres boots + queries on the main engine (no wasm-JIT tier); the recorder logs the boot vs. query
  // split, whether the session was restored from IndexedDB, and each query's timing to the console.
  const rec = runStart(c, { tier: 'interpreter' });
  c.el.result.textContent = '';
  c.el.canvas.hidden = true;
  // `\reset` (a bare meta-command in the editor): drop the saved database and close any live session, so
  // the next Run boots from the pristine image. The way back to a clean slate once a session persists.
  if (c.editor.getValue().trim() === '\\reset') {
    await pgClear(pgKey(c));
    if (c.pgSession) {
      eng.ex.temen_pg_close();
      c.pgSession = false;
      c.el.stop.disabled = true;
    }
    setState(c, 'done', 'saved database cleared — the next Run boots a fresh cluster');
    logTo(c, 'reset: cleared the saved session');
    c.el.run.disabled = broken;
    runNote(rec, { action: 'reset' });
    runEnd(rec, { ok: true });
    return;
  }
  // 1) Open the session on the first Run. Prefer a **saved** snapshot (a prior session's data dir,
  //    persisted in IndexedDB) over the pristine image, so a page reload resumes where you left off.
  if (!c.pgSession) {
    setState(c, 'running', 'fetching module + image…');
    c.el.stdout.textContent = '';
    let modBytes, imgBytes, restored = false;
    try {
      // Sequential (not Promise.all) so the two large downloads report progress one at a time.
      modBytes = await fetchTimed(rec, c, ex.url);
      const saved = await pgLoad(pgKey(c));
      if (saved) {
        imgBytes = saved instanceof Uint8Array ? saved : new Uint8Array(saved);
        restored = true;
        rec.assets.push({ name: 'saved-image (IndexedDB)', bytes: imgBytes.length, cached: true });
      } else {
        imgBytes = await fetchTimed(rec, c, ex.image);
      }
    } catch (e) {
      setState(c, 'error', `${e.message} — run \`node build-pg-assets.mjs\` to stage the Postgres artifacts`);
      logTo(c, `fetch failed: ${e.message}`);
      runNote(rec, { fetchError: e.message });
      runEnd(rec, { ok: false });
      return;
    }
    runNote(rec, { restored });
    setState(c, 'running', restored
      ? 'restoring your saved database… (first Run only — a few seconds)'
      : 'booting postgres… (first Run only — a few seconds; later queries are instant)');
    c.el.run.disabled = true;
    await new Promise((r) => setTimeout(r, 30)); // let the status paint before the synchronous boot
    try {
      const modP = eng.ex.temen_alloc(modBytes.length);
      const imgP = eng.ex.temen_alloc(imgBytes.length);
      const view = new Uint8Array(eng.memory.buffer);
      view.set(modBytes, modP);
      view.set(imgBytes, imgP);
      const t0 = performance.now();
      const rc = eng.ex.temen_pg_open(modP, modBytes.length, imgP, imgBytes.length);
      const ms = runStage(rec, restored ? 'restore' : 'boot', performance.now() - t0).toFixed(0);
      eng.ex.temen_dealloc(modP, modBytes.length);
      eng.ex.temen_dealloc(imgP, imgBytes.length);
      c.el.stdout.textContent += readPgStdout(); // the banner + first prompt
      if (rc !== 0) {
        // A saved image that won't boot is likely corrupt — drop it so the next Run starts clean.
        if (restored) {
          await pgClear(pgKey(c));
          logTo(c, 'saved session failed to boot — cleared it; Run again for a fresh database');
        }
        setState(c, 'error', `boot failed: status ${eng.ex.temen_status()} (1=decode 3=trap 6=verify)`);
        c.el.run.disabled = broken;
        runNote(rec, { bootStatus: eng.ex.temen_status() });
        runEnd(rec, { ok: false });
        return;
      }
      c.pgSession = true;
      c.el.stop.disabled = false;
      logTo(c, restored ? `temen_pg_open: restored saved session in ${ms}ms` : `temen_pg_open: backend booted in ${ms}ms`);
    } catch (e) {
      setState(c, 'error', `boot error: ${e.message}`);
      c.el.run.disabled = broken;
      runNote(rec, { bootError: e.message });
      runEnd(rec, { ok: false });
      return;
    }
  }
  // 2) Send the editor's SQL to the live backend as one query.
  const sql = c.editor.getValue();
  if (!sql.trim()) {
    setState(c, 'done', 'session live — type SQL and Run (state persists across reloads · `\\reset` clears it)');
    c.el.run.disabled = broken;
    runNote(rec, { action: 'open-only (no query)' });
    runEnd(rec, { ok: true });
    return;
  }
  try {
    const text = sql.endsWith('\n') ? sql : sql + '\n';
    const b = new TextEncoder().encode(text);
    const p = eng.ex.temen_alloc(b.length);
    new Uint8Array(eng.memory.buffer).set(b, p);
    const t0 = performance.now();
    const rc = eng.ex.temen_pg_query(p, b.length);
    const ms = runStage(rec, 'query', performance.now() - t0).toFixed(0);
    eng.ex.temen_dealloc(p, b.length);
    // Append this query's output delta to the running transcript (result blocks → psql-style tables).
    c.el.stdout.textContent += readPgStdout();
    c.el.stdout.scrollTop = c.el.stdout.scrollHeight;
    const status = eng.ex.temen_status();
    runNote(rec, { sqlBytes: b.length });
    if (rc === 0) {
      setState(c, 'done', `query ran in ${ms}ms · session live · saved (reload to resume)`);
      logTo(c, `temen_pg_query in ${ms}ms`);
      persistPg(c); // snapshot the (possibly mutated) data dir so it survives a reload
      runEnd(rec, { ok: true, status });
    } else if (status === 5) {
      // The backend exited (e.g. the SQL issued a shutdown) — the session is over. Persist its final
      // state first, so even a clean shutdown is resumable.
      persistPg(c);
      c.pgSession = false;
      c.el.stop.disabled = true;
      setState(c, 'done', 'backend exited — Run reopens your saved database');
      runNote(rec, { backendExited: true });
      runEnd(rec, { ok: true, status });
    } else {
      setState(c, 'error', `query failed: status ${status}`);
      logTo(c, `temen_pg_query status ${status}`);
      runEnd(rec, { ok: false, status });
    }
  } catch (e) {
    setState(c, 'error', `query error: ${e.message}`);
    logTo(c, `query error: ${e.message}`);
    runNote(rec, { queryError: e.message });
    runEnd(rec, { ok: false });
  } finally {
    c.el.run.disabled = broken;
  }
}

// ---- the reactor run model (interactive per-frame guests: bounce, life, Doom) --------------------
// Open a reactor module once, then drive it one `tick` per requestAnimationFrame. Only one reactor
// runs at a time; `activeReactorCard` is the card it belongs to (for teardown + the GPU canvas).
let reactorRAF = null; // the pending requestAnimationFrame id while a reactor loop runs (else null)
let jitReactor = null; // the wasm-JIT reactor driver while a JIT loop runs (else null → interpreter)
let activeReactorCard = null;
// Instrumentation for the live reactor loop: the run recorder plus a live frame count / start time, so a
// user Stop (which cancels the loop out-of-band) can still emit the console summary the loop's own
// terminal path would. `finalizeReactor` is the single close-out; it no-ops once consumed.
let reactorRun = null; // { rec, t0, frames } while a reactor is live (else null)
function finalizeReactor(how) {
  if (!reactorRun) return;
  const { rec, t0, frames } = reactorRun;
  reactorRun = null;
  const secs = (clockNow() - t0) / 1000;
  const fps = secs > 0 ? frames / secs : 0;
  runStage(rec, `run:${rec.tier || 'interpreter'}`, clockNow() - t0);
  runNote(rec, { frames, seconds: +secs.toFixed(2), avgFps: +fps.toFixed(1), ...how });
  runEnd(rec, { ok: how.ok !== false, status: how.status });
}

// Feed one key event to the running reactor guest through the `keyboard` capability (JS keyCode +
// pressed flag). Shared by the physical-keyboard handler and the on-screen touch dpad; a no-op when no
// reactor loop is running, and routed to whichever tier (interpreter / wasm-JIT) is live.
function sendReactorKey(keyCode, pressed) {
  if (reactorRAF === null) return;
  if (jitReactor) eng.ex.temen_onramp_jit_key(keyCode, pressed);
  else eng.ex.temen_onramp_key(keyCode, pressed);
}

// Cancel any running reactor loop and free the guest instance. Safe to call when none is running.
function stopReactor() {
  teardownWebGPU(); // drop any GPU device + the servicer (no-op for non-webgpu reactors)
  if (activeReactorCard) activeReactorCard.el.gpucanvas.hidden = true;
  if (reactorRAF === null) { finalizeReactor({ ended: 'stopped', ok: true }); activeReactorCard = null; return; }
  cancelAnimationFrame(reactorRAF);
  reactorRAF = null;
  finalizeReactor({ ended: 'stopped', ok: true }); // a live loop was cancelled by the user / a superseding Run
  if (jitReactor) {
    jitReactor.close();
    jitReactor = null;
  } else {
    eng.ex.temen_onramp_close();
  }
  activeReactorCard = null;
}

async function runReactor(c) {
  const ex = c.ex;
  stopReactor();
  activeReactorCard = c;
  setState(c, 'running', 'fetching module…');
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;
  // The "wasm-JIT" toggle runs an emittable reactor's whole tick() on emitted wasm (near-native) rather
  // than the interpreter. Only offered for JIT-capable examples (Doom); falls back if the emit fails.
  const useJit = !!(ex.jit && c.el.jit && c.el.jit.checked);
  const rec = runStart(c, { tier: useJit ? 'wasm-JIT' : 'interpreter' });
  let bytes;
  try {
    bytes = await fetchTimed(rec, c, ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate it`);
    runNote(rec, { fetchError: e.message });
    runEnd(rec, { ok: false });
    return;
  }
  // A GPU demo: bring up a `navigator.gpu` device + the WebGPU canvas and install the servicer BEFORE
  // the reactor's first `tick`, so the guest's `webgpu` capability calls (set_shader/present) render.
  if (ex.webgpu) {
    if (!webgpuAvailable()) {
      setState(c, 'error', 'no WebGPU in this browser — the GPU demo needs it (try Chrome/Edge)');
      runNote(rec, { webgpu: 'unavailable' });
      runEnd(rec, { ok: false });
      return;
    }
    try {
      c.el.gpucanvas.hidden = false;
      await initWebGPU(c.el.gpucanvas);
      logTo(c, 'WebGPU device ready — the guest ships one WGSL shader; the GPU renders every frame');
    } catch (e) {
      setState(c, 'error', `WebGPU init failed: ${e.message}`);
      c.el.gpucanvas.hidden = true;
      runNote(rec, { webgpuError: e.message });
      runEnd(rec, { ok: false });
      return;
    }
  }
  // Open the reactor: alloc, copy the module in, run _start (decode + grant powerbox). A guest that
  // needs a served file (Doom reads its WAD at _start) is opened with temen_onramp_open_fs, which grants
  // the `fs` capability over the fetched blob; every other reactor guest uses plain temen_onramp_open.
  let wad = null;
  if (ex.wad) {
    try {
      wad = await fetchTimed(rec, c, ex.wad);
    } catch (e) {
      setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate the WAD`);
      runNote(rec, { fetchError: e.message });
      runEnd(rec, { ok: false });
      return;
    }
    logTo(c, `fetched ${ex.wad}: ${wad.length}B file (served through the fs capability)`);
  }
  setState(c, 'running',
    ex.wad ? `booting DOOM… (reading the WAD, building the renderer — a few seconds)${useJit ? ' [wasm-JIT]' : ''}`
      : 'running…');
  const tOpen = performance.now();
  if (useJit) {
    try {
      jitReactor = await openJitReactor(eng.ex, eng.memory, bytes, 'doom1.wad', wad);
      logTo(c, `wasm-JIT reactor opened: ${ex.url} (${bytes.length}B) — tick() runs on emitted wasm`);
    } catch (e) {
      jitReactor = null;
      logTo(c, `wasm-JIT reactor unavailable (${e.message}); falling back to the interpreter`);
      runNote(rec, { jitFallbackReason: e.message });
    }
  }
  if (!jitReactor) {
    let opened;
    if (ex.wad) {
      const nameBytes = new TextEncoder().encode('doom1.wad');
      const modP = eng.ex.temen_alloc(bytes.length);
      const nameP = eng.ex.temen_alloc(nameBytes.length);
      const wadP = eng.ex.temen_alloc(wad.length);
      const view = new Uint8Array(eng.memory.buffer);
      view.set(bytes, modP);
      view.set(nameBytes, nameP);
      view.set(wad, wadP);
      opened = eng.ex.temen_onramp_open_fs(modP, bytes.length, nameP, nameBytes.length, wadP, wad.length);
      eng.ex.temen_dealloc(modP, bytes.length);
      eng.ex.temen_dealloc(nameP, nameBytes.length);
      eng.ex.temen_dealloc(wadP, wad.length);
    } else {
      const p = eng.ex.temen_alloc(bytes.length);
      new Uint8Array(eng.memory.buffer).set(bytes, p);
      opened = eng.ex.temen_onramp_open(p, bytes.length);
      eng.ex.temen_dealloc(p, bytes.length);
    }
    if (opened !== 0) {
      setState(c, 'error', `reactor open failed: status ${eng.ex.temen_status()} (2=unsupported 3=trap)`);
      logTo(c, `temen_onramp_open failed: ${opened}`);
      activeReactorCard = null;
      runNote(rec, { openStatus: eng.ex.temen_status() });
      runEnd(rec, { ok: false });
      return;
    }
    logTo(c, `reactor opened: ${ex.url} (${bytes.length}B) — arrow keys steer, Stop ends`);
  }
  const tier = jitReactor ? 'wasm-JIT' : 'interpreter';
  runStage(rec, `open:${tier}`, performance.now() - tOpen);
  runTier(rec, tier);
  setState(c, 'running', `running (${tier}) — arrow keys to steer, Stop to end`);
  c.el.run.disabled = true;
  c.el.stop.disabled = false;
  let frames = 0;
  const t0 = performance.now();
  let fpsFrames = 0;
  let fpsT0 = t0;
  // Publish the live loop's stats so a user Stop (out-of-band cancel) can still emit the console summary.
  reactorRun = { rec, t0, frames: 0 };
  const loop = () => {
    const status = jitReactor ? jitReactor.frame() : eng.ex.temen_onramp_frame();
    presentFrame(c, eng.ex.temen_framebuffer_width(), eng.ex.temen_framebuffer_height());
    frames++;
    if (reactorRun) reactorRun.frames = frames;
    fpsFrames++;
    const now = performance.now();
    if (now - fpsT0 >= 1000) {
      const fps = (fpsFrames * 1000 / (now - fpsT0)).toFixed(1);
      setState(c, 'running', `running (${tier}) — ${fps} fps · arrow keys to steer, Stop to end`);
      fpsFrames = 0;
      fpsT0 = now;
    }
    if (status === 0) {
      reactorRAF = requestAnimationFrame(loop);
      return;
    }
    reactorRAF = null;
    let trapDetail = '';
    if (status !== 0 && status !== 5) {
      const n = jitReactor ? eng.ex.temen_onramp_jit_trap_len() : eng.ex.temen_onramp_trap_len();
      if (n > 0) {
        trapDetail = new TextDecoder().decode(
          new Uint8Array(eng.memory.buffer).slice(eng.ex.temen_stdout_ptr(), eng.ex.temen_stdout_ptr() + n));
      }
    }
    if (jitReactor) {
      jitReactor.close();
      jitReactor = null;
    } else {
      eng.ex.temen_onramp_close();
    }
    activeReactorCard = null;
    c.el.run.disabled = broken;
    c.el.stop.disabled = true;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setState(c, status === 5 ? 'done' : 'error',
      status === 5 ? `guest exited after ${frames} frames · ${secs}s`
        : `reactor trapped: status ${status}${trapDetail ? ` (${trapDetail})` : ''}`);
    logTo(c, `reactor stopped (${tier}): status ${status}${trapDetail ? ` ${trapDetail}` : ''} after ${frames} frames in ${secs}s`);
    finalizeReactor({ ended: status === 5 ? 'exit' : 'trap', ok: status === 5, status, trap: trapDetail || undefined });
  };
  reactorRAF = requestAnimationFrame(loop);
}

// ---- "prove it": the interpreter ≡ wasm-JIT differential, in the page --------------------------------
// The project's core claim is "verified ⇒ the same result on both tiers." For a JIT-emittable reactor,
// prove it live: open the SAME guest on the interpreter and on the wasm-JIT tier, run N frames on each,
// and compare the presented framebuffer byte-for-byte. (This is exactly what browser-jit-reactor-test.mjs
// asserts in CI — surfaced here as a button.)

// FNV-1a over the presented framebuffer, tagged with its dimensions (so a size divergence also shows).
// Copied out of shared memory — a plain view would be a live alias.
function hashFB() {
  const w = eng.ex.temen_framebuffer_width();
  const h = eng.ex.temen_framebuffer_height();
  const p = Number(eng.ex.temen_framebuffer_ptr());
  const px = new Uint8Array(eng.memory.buffer).slice(p, p + w * h * 4);
  let hsh = 0x811c9dc5;
  for (let i = 0; i < px.length; i++) { hsh ^= px[i]; hsh = Math.imul(hsh, 0x01000193) >>> 0; }
  return `${w}x${h}:${(hsh >>> 0).toString(16)}`;
}

// Open the interpreter reactor, run up to `n` frames, hashing each presented frame; close. Synchronous.
function framesInterp(bytes, wad, n) {
  let opened;
  if (wad) {
    const nameBytes = new TextEncoder().encode('doom1.wad');
    const modP = eng.ex.temen_alloc(bytes.length);
    const nameP = eng.ex.temen_alloc(nameBytes.length);
    const wadP = eng.ex.temen_alloc(wad.length);
    const view = new Uint8Array(eng.memory.buffer);
    view.set(bytes, modP);
    view.set(nameBytes, nameP);
    view.set(wad, wadP);
    opened = eng.ex.temen_onramp_open_fs(modP, bytes.length, nameP, nameBytes.length, wadP, wad.length);
    eng.ex.temen_dealloc(modP, bytes.length);
    eng.ex.temen_dealloc(nameP, nameBytes.length);
    eng.ex.temen_dealloc(wadP, wad.length);
  } else {
    const p = eng.ex.temen_alloc(bytes.length);
    new Uint8Array(eng.memory.buffer).set(bytes, p);
    opened = eng.ex.temen_onramp_open(p, bytes.length);
    eng.ex.temen_dealloc(p, bytes.length);
  }
  if (opened !== 0) throw new Error(`interpreter open failed: status ${eng.ex.temen_status()}`);
  const hs = [];
  for (let i = 0; i < n; i++) { if (eng.ex.temen_onramp_frame() !== 0) break; hs.push(hashFB()); }
  eng.ex.temen_onramp_close();
  return hs;
}

// Open the wasm-JIT reactor (throws if the tick isn't emittable), run up to `n` frames, hashing each.
async function framesJit(bytes, wad, n) {
  const r = await openJitReactor(eng.ex, eng.memory, bytes, 'doom1.wad', wad);
  const hs = [];
  for (let i = 0; i < n; i++) { if (r.frame() !== 0) break; hs.push(hashFB()); }
  r.close();
  return hs;
}

async function proveParity(c) {
  if (broken) return;
  stopReactor(); // a parity run supersedes any running reactor loop
  const ex = c.ex;
  setState(c, 'running', 'proving interpreter ≡ wasm-JIT…');
  c.el.run.disabled = true;
  c.el.prove.disabled = true;
  let bytes, wad = null;
  try {
    bytes = await fetchModule(ex.url);
    if (ex.wad) wad = await fetchModule(ex.wad);
  } catch (e) {
    setState(c, 'error', `${e.message}`);
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
    return;
  }
  const N = 30;
  try {
    // Yield a paint so "proving…" lands before the synchronous interpreter frames block the thread.
    await new Promise((r) => setTimeout(r, 30));
    const interpH = framesInterp(bytes, wad, N);
    const jitH = await framesJit(bytes, wad, N);
    const n = Math.min(interpH.length, jitH.length);
    let mismatch = -1;
    for (let i = 0; i < n; i++) if (interpH[i] !== jitH[i]) { mismatch = i; break; }
    const identical = mismatch === -1 && interpH.length === jitH.length && n > 0;
    if (identical) {
      setState(c, 'done', `✓ interpreter ≡ wasm-JIT — byte-identical framebuffer across ${n} frames`);
      logTo(c, `parity: ${n} frames byte-identical on both tiers`);
    } else {
      setState(c, 'error', `✗ tiers diverged at frame ${mismatch} (interp ${interpH.length} / jit ${jitH.length} frames)`);
      logTo(c, `parity: diverged at frame ${mismatch}`);
    }
  } catch (e) {
    setState(c, 'error', `parity run failed: ${e.message}`);
    logTo(c, `parity run failed: ${e.message}`);
  } finally {
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
  }
}

// The module twin of proveParity: run the SAME on-ramp module (with the same editor stdin) on the
// interpreter and on the wasm-JIT tier and assert the captured **stdout** is byte-identical — the
// "verified ⇒ same result on both tiers" claim for a run-to-completion guest (framebuffer demos prove
// it per-frame instead). This is exactly what browser-jit-module-test.mjs asserts in CI.
async function proveModuleParity(c) {
  if (broken) return;
  stopReactor();
  const ex = c.ex;
  setState(c, 'running', 'proving interpreter ≡ wasm-JIT…');
  c.el.run.disabled = true;
  c.el.prove.disabled = true;
  let bytes;
  try {
    bytes = await fetchModule(ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message}`);
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
    return;
  }
  let stdinBytes = null;
  if (ex.editable) {
    const enc = new TextEncoder().encode(c.editor.getValue());
    if (enc.length > 0) stdinBytes = enc;
  }
  try {
    // Yield a paint so "proving…" lands before the synchronous interpreter run blocks the thread.
    await new Promise((r) => setTimeout(r, 30));
    // A warm card runs the two **warm** tiers, so prove those agree: warm-interp (`temen_warm_eval`) ≡
    // warm+JIT (`runWarmJit`), both evaluating over the same restored snapshot.
    if (ex.warm) {
      if (!ensureWarmSession(bytes, ex.url)) throw new Error('warm session unavailable for this module');
      const interpOut = warmEval(stdinBytes).stdout;
      let warmJitOut;
      try {
        await runWarmJit(eng.ex, eng.memory, stdinBytes, `${ex.url}#eval`);
        warmJitOut = readModuleStdout();
      } catch (e) {
        // A setjmp-rooted guest (e.g. Lua's `lua_pcall`) routes `eval_run` to InterpDriven (#1081), so
        // `temen_warm_jit_open` declines with STATUS_UNSUPPORTED (2) and this guest has only one warm
        // tier — the warm interpreter, which already ran above. That's the documented fallback, not a
        // failure: report the single-tier result rather than erroring the card.
        if (eng.ex.temen_status() === 2) {
          setState(c, 'done', `✓ warm-interp only — warm+JIT declined for this guest (setjmp-rooted, ${interpOut.length}B stdout)`);
          logTo(c, `parity: warm+JIT declined (eval_run → InterpDriven, #1081); warm interpreter carries this guest`);
          return;
        }
        setState(c, 'error', `✗ warm+JIT unavailable: ${e.message}`);
        logTo(c, `parity: warm-JIT emit failed: ${e.message}`);
        return;
      }
      if (interpOut === warmJitOut) {
        setState(c, 'done', `✓ warm-interp ≡ warm+JIT — byte-identical stdout (${warmJitOut.length}B)`);
        logTo(c, `parity: ${warmJitOut.length}B stdout byte-identical on both warm tiers`);
      } else {
        setState(c, 'error', `✗ warm tiers diverged (interp ${interpOut.length}B / jit ${warmJitOut.length}B stdout)`);
        logTo(c, `parity: warm stdout diverged (interp ${interpOut.length}B vs jit ${warmJitOut.length}B)`);
      }
      return;
    }
    const interp = moduleInterp(bytes, stdinBytes);
    let jitOut;
    try {
      await runJitModule(eng.ex, eng.memory, bytes, stdinBytes);
      jitOut = readModuleStdout();
    } catch (e) {
      setState(c, 'error', `✗ wasm-JIT unavailable: ${e.message}`);
      logTo(c, `parity: JIT emit failed: ${e.message}`);
      return;
    }
    if (interp.stdout === jitOut) {
      setState(c, 'done', `✓ interpreter ≡ wasm-JIT — byte-identical stdout (${jitOut.length}B)`);
      logTo(c, `parity: ${jitOut.length}B stdout byte-identical on both tiers`);
    } else {
      setState(c, 'error', `✗ tiers diverged (interp ${interp.stdout.length}B / jit ${jitOut.length}B stdout)`);
      logTo(c, `parity: stdout diverged (interp ${interp.stdout.length}B vs jit ${jitOut.length}B)`);
    }
  } catch (e) {
    setState(c, 'error', `parity run failed: ${e.message}`);
    logTo(c, `parity run failed: ${e.message}`);
  } finally {
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
  }
}

// The chibicc-card twin of `proveModuleParity`: run **pass 1** (chibicc compiling the editor's C) on both
// tiers and assert the emitted TEMEN-IR text is byte-identical — the compiler is the run-to-completion
// guest, its stdout is the IR. This is exactly what `chibicc_jit.rs` asserts natively.
async function proveChibiccParity(c) {
  if (broken) return;
  stopReactor();
  const ex = c.ex;
  setState(c, 'running', 'proving interpreter ≡ wasm-JIT…');
  c.el.run.disabled = true;
  c.el.prove.disabled = true;
  let compiler;
  try {
    compiler = await fetchModule(ex.url);
  } catch (e) {
    setState(c, 'error', `${e.message}`);
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
    return;
  }
  const srcBytes = new TextEncoder().encode(c.editor.getValue());
  if (srcBytes.length === 0) {
    setState(c, 'error', 'empty source');
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
    return;
  }
  // Both tiers must compile with the same `-g` setting for the IR to match (it's a byte differential).
  const gOn = c.el.gflag && c.el.gflag.checked ? 1 : 0;
  try {
    // Yield a paint so "proving…" lands before the synchronous interpreter compile blocks the thread.
    await new Promise((r) => setTimeout(r, 30));
    // Interpreter pass 1.
    const p = eng.ex.temen_alloc(compiler.length);
    const sp = eng.ex.temen_alloc(srcBytes.length);
    const view = new Uint8Array(eng.memory.buffer);
    view.set(compiler, p);
    view.set(srcBytes, sp);
    eng.ex.temen_run_onramp_fs(p, compiler.length, 0, 0, sp, srcBytes.length, gOn);
    eng.ex.temen_dealloc(p, compiler.length);
    eng.ex.temen_dealloc(sp, srcBytes.length);
    const interpIr = readModuleStdout();
    // wasm-JIT pass 1.
    let jitIr;
    try {
      await runJitCompiler(eng.ex, eng.memory, compiler, srcBytes, gOn);
      jitIr = readModuleStdout();
    } catch (e) {
      setState(c, 'error', `✗ wasm-JIT unavailable: ${e.message}`);
      logTo(c, `parity: JIT compile failed: ${e.message}`);
      return;
    }
    if (interpIr === jitIr) {
      setState(c, 'done', `✓ interpreter ≡ wasm-JIT — byte-identical Temen IR (${jitIr.length}B)`);
      logTo(c, `parity: ${jitIr.length}B emitted IR byte-identical on both tiers`);
    } else {
      setState(c, 'error', `✗ tiers diverged (interp ${interpIr.length}B / jit ${jitIr.length}B IR)`);
      logTo(c, `parity: emitted IR diverged (interp ${interpIr.length}B vs jit ${jitIr.length}B)`);
    }
  } catch (e) {
    setState(c, 'error', `parity run failed: ${e.message}`);
    logTo(c, `parity run failed: ${e.message}`);
  } finally {
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
  }
}

// The self-host twin of `proveChibiccParity`: compile the selected chibicc TU to an object on both the
// bytecode interpreter and the wasm-JIT and assert the emitted object IR is byte-identical — the same
// interp≡JIT guarantee `chibicc_jit.rs` proves natively, now over chibicc's own source.
async function proveSelfhostParity(c) {
  if (broken) return;
  stopReactor();
  const ex = c.ex;
  setState(c, 'running', 'proving interpreter ≡ wasm-JIT…');
  c.el.run.disabled = true;
  c.el.prove.disabled = true;
  let compiler, image;
  try {
    compiler = await fetchModule(ex.url);
    image = await fetchModule(ex.image);
  } catch (e) {
    setState(c, 'error', `${e.message}`);
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
    return;
  }
  const tu = c.el.tu ? c.el.tu.value : `frontend/chibicc/${ex.tus[0]}`;
  const short = tu.split('/').pop();
  const tuBytes = new TextEncoder().encode(tu);
  const gOn = c.el.gflag && c.el.gflag.checked ? 1 : 0;
  try {
    // Yield a paint so "proving…" lands before the synchronous interpreter compile blocks the thread.
    await new Promise((r) => setTimeout(r, 30));
    // Interpreter emit-object.
    const p = eng.ex.temen_alloc(compiler.length);
    const ip = eng.ex.temen_alloc(image.length);
    const tp = eng.ex.temen_alloc(tuBytes.length);
    const view = new Uint8Array(eng.memory.buffer);
    view.set(compiler, p);
    view.set(image, ip);
    view.set(tuBytes, tp);
    eng.ex.temen_selfhost_emit_object_fs(p, compiler.length, ip, image.length, tp, tuBytes.length, gOn);
    eng.ex.temen_dealloc(p, compiler.length);
    eng.ex.temen_dealloc(ip, image.length);
    eng.ex.temen_dealloc(tp, tuBytes.length);
    const interpObj = readModuleStdout();
    // wasm-JIT emit-object.
    let jitObj;
    try {
      await runJitSelfhost(eng.ex, eng.memory, compiler, image, tuBytes, gOn);
      jitObj = readModuleStdout();
    } catch (e) {
      setState(c, 'error', `✗ wasm-JIT unavailable: ${e.message}`);
      logTo(c, `parity: JIT self-host failed: ${e.message}`);
      return;
    }
    if (interpObj === jitObj && interpObj.length > 0) {
      setState(c, 'done', `✓ interpreter ≡ wasm-JIT — byte-identical object for ${short} (${jitObj.length}B)`);
      logTo(c, `parity: ${short} → ${jitObj.length}B object byte-identical on both tiers`);
    } else {
      setState(c, 'error', `✗ tiers diverged (interp ${interpObj.length}B / jit ${jitObj.length}B)`);
      logTo(c, `parity: ${short} object diverged (interp ${interpObj.length}B vs jit ${jitObj.length}B)`);
    }
  } catch (e) {
    setState(c, 'error', `parity run failed: ${e.message}`);
    logTo(c, `parity run failed: ${e.message}`);
  } finally {
    c.el.run.disabled = broken;
    c.el.prove.disabled = false;
  }
}

// ---- the DAP debugger (DEBUGGING.md): breakpoints · stepping · variables, on the bytecode engine --
// One debug session at a time. The panel drives the `temen-dap` server (bytecode backend) through the
// `dap.js` client over the wasm FFI: launch the Temen text, run to a breakpoint, highlight the stopped
// source line, and show the paused frame's named locals; Step/Continue advance it. This is the same
// DAP an editor speaks — the playground is just another DAP frontend.
let dapClient = null; // the active DAP client while a session runs (else null)
let dapCard = null; // the card the session belongs to
let dapWatch = new Set(); // source-variable names armed as data breakpoints (watchpoints) this session
let dapScopeRef = 0; // the paused frame's Locals `variablesReference` (what `dataBreakpointInfo` scopes to)
let dapStopped = 1; // the DAP threadId that hit the current stop (stepping always drives this thread)
let dapThread = 1; // the DAP threadId currently being inspected (select a thread to view its stack)
let dapSource = 'source.temt'; // the DAP source path breakpoints target this session (the launched
                              // program's `debug.file`; for chibicc the compiled IR's `/in.c`, not the C editor)

// The DAP source a breakpoint request targets — the program's own `debug.file 0 "…"` if it declares
// one, else the name the engine's auto debug info uses (temen-text's AUTO_DEBUG_FILE = "source.temt"), so
// breakpoints bind for a hand-written program with no explicit `debug` section. For chibicc this reads
// the *compiled IR*'s `debug.file` (`/in.c`) — the C editor lines still map through chibicc's debug.loc.
function dapSourceName(src) {
  const m = /debug\.file\s+0\s+"([^"]+)"/.exec(src);
  return m ? m[1] : 'source.temt';
}

// Push the card's current breakpoint lines (editor 0-based → DAP 1-based) to the server, against the
// session's source (`dapSource` — set at launch to the launched program's `debug.file`).
function dapSyncBreakpoints(c) {
  const breakpoints = c.editor.breakpointLines().map((l) => ({ line: l + 1 }));
  dapClient.send('setBreakpoints', {
    source: { path: dapSource },
    breakpoints,
  });
}

// Render the inspected thread's paused frame + its named locals into the card's Variables pane, and
// highlight its source line (frame.line is 1-based; 0 ⇒ an unmapped op, so no highlight). For a
// multithreaded (`thread.spawn`) guest a thread selector lists every live vCPU — clicking one focuses
// its stack (`select_task`, via a per-thread `stackTrace`) without resuming; the thread that hit the
// stop is marked ●, and stepping always drives *it*. Each memory-located variable gets a ● watch
// toggle (a data breakpoint); a promoted SSA scalar has no window address, so its toggle is disabled.
function dapShowStop(c) {
  // Live threads (one per vCPU). A single-vCPU guest reports just thread 1, so the bar is hidden.
  const threads = dapClient.send('threads', {}).response.body.threads;
  if (!threads.some((t) => t.id === dapThread)) dapThread = dapStopped; // inspected thread went away
  const frame = dapClient.send('stackTrace', { threadId: dapThread }).response.body.stackFrames[0];
  if (!frame) return;
  if (frame.line > 0) c.editor.setStopLine(frame.line - 1);
  else c.editor.clearStopLine();
  const scope = dapClient.send('scopes', { frameId: frame.id }).response.body.scopes[0];
  dapScopeRef = scope.variablesReference;
  const vars = dapClient.send('variables', { variablesReference: dapScopeRef }).response.body.variables;
  const rows = vars
    .map((v) => {
      // A `null` dataId ⇒ no watchable window address here (an SSA-promoted scalar) → disabled toggle.
      const dataId = dapClient.send('dataBreakpointInfo', { variablesReference: dapScopeRef, name: v.name })
        .response.body.dataId;
      const on = dapWatch.has(v.name);
      const cls = `wp${dataId == null ? ' off' : ''}${on ? ' on' : ''}`;
      const title = dataId == null ? 'no watchable address here'
        : (on ? 'Remove this data breakpoint' : 'Break when this value changes');
      const toggle = `<button class="${cls}" data-watch="${v.name}"${dataId == null ? ' disabled' : ''} title="${title}">●</button>`;
      return `<div>${toggle} <span class="bpname">${v.name}</span> = ${v.value}${v.type ? ` <em>${v.type}</em>` : ''}</div>`;
    })
    .join('');
  const bar = threads.length > 1
    ? `<div class="dbg-threads">${threads
        .map((t) => {
          const sel = t.id === dapThread ? ' sel' : '';
          const mark = t.id === dapStopped ? ' ●' : '';
          return `<button class="thr${sel}" data-thread="${t.id}" title="Inspect ${t.name} (● = stopped here)">${t.name}${mark}</button>`;
        })
        .join('')}</div>`
    : '';
  c.el.dbgVars.innerHTML = `${bar}<div>${frame.name} · line ${frame.line}</div>${rows}`;
  dapArmWatches(); // (re)arm the watched set against this stop's addresses
}

// Focus a different thread's stack (multithreaded sessions) and re-render — no resume; stepping still
// drives the stopped thread.
function dapSelectThread(c, id) {
  if (dapCard !== c || !dapClient) return;
  dapThread = id;
  dapShowStop(c);
}

// Arm the current watch set as DAP data breakpoints: mint a fresh `dataId` for each watched variable
// (`dataBreakpointInfo`, scoped to the paused frame) and replace the server's set (`setDataBreakpoints`
// takes the full list each call). A name that no longer resolves to an address is silently dropped.
function dapArmWatches() {
  if (!dapClient) return;
  const breakpoints = [];
  for (const name of dapWatch) {
    const dataId = dapClient.send('dataBreakpointInfo', { variablesReference: dapScopeRef, name })
      .response.body.dataId;
    if (dataId != null) breakpoints.push({ dataId, accessType: 'write' });
  }
  dapClient.send('setDataBreakpoints', { breakpoints });
}

// Toggle a data breakpoint on a source variable, then re-render (which re-arms + refreshes the ● state).
function dapToggleWatch(c, name) {
  if (dapCard !== c || !dapClient) return;
  dapWatch.has(name) ? dapWatch.delete(name) : dapWatch.add(name);
  dapShowStop(c);
}

// Handle a resume reply: a `terminated` event ends the session; a `stopped` event pauses (show it).
function dapHandle(c, reply) {
  // A powerbox session (chibicc under the on-ramp I/O powerbox) streams the guest's captured stdout as
  // `output` events. The event carries the *full* current output (it rewinds on a reverse step), so
  // replace the pane with the latest one — the program's own output shows as you step / reverse.
  const output = reply.events.filter((e) => e.event === 'output').pop();
  if (output) c.el.stdout.textContent = output.body.output;
  if (reply.events.some((e) => e.event === 'terminated')) {
    endDebug(c, 'program finished');
    return;
  }
  const stopped = reply.events.find((e) => e.event === 'stopped');
  if (stopped) {
    // Focus follows the thread that hit the stop (stepping drives it); the user can then select another.
    dapStopped = stopped.body.threadId || 1;
    dapThread = dapStopped;
    dapShowStop(c);
    const where = dapClient.send('threads', {}).response.body.threads.length > 1
      ? `paused (${stopped.body.reason}) in thread-${dapStopped - 1} — Step / Continue, Stop to end`
      : `paused (${stopped.body.reason}) — Step / Continue, Stop to end`;
    setState(c, 'running', where);
  }
}

// Fetch chibicc and compile the card's current C source to TEMEN-IR text with `-g` (the debug waist:
// source lines + variable names), for a source-level debug session. Returns `{ ir, status, stderr }`;
// the caller reports a failed compile. Mirrors `runChibicc`'s pass 1, always debug-on.
async function chibiccCompileIR(c) {
  const compiler = await fetchModule(c.ex.url, onFetchProgress(c, baseName(c.ex.url)));
  const srcBytes = new TextEncoder().encode(c.editor.getValue());
  if (srcBytes.length === 0) return { ir: '', status: -1, stderr: 'empty source' };
  const p = eng.ex.temen_alloc(compiler.length);
  const sp = eng.ex.temen_alloc(srcBytes.length);
  const view = new Uint8Array(eng.memory.buffer);
  view.set(compiler, p);
  view.set(srcBytes, sp);
  eng.ex.temen_run_onramp_fs(p, compiler.length, 0, 0, sp, srcBytes.length, 1); // 1 = -g
  const status = eng.ex.temen_status();
  const ir = readModuleStdout();
  const stderr = readModuleStderr();
  eng.ex.temen_dealloc(p, compiler.length);
  eng.ex.temen_dealloc(sp, srcBytes.length);
  return { ir, status, stderr };
}

// Start a debug session on the bytecode engine. For an TEMEN-text card the editor content *is* the
// program. For the chibicc C card, compile the C with `-g` first and debug the emitted IR at **C source
// level**: breakpoints on C lines + C locals by name bind through chibicc's `debug.loc`/`debug.var`,
// while the editor keeps showing C. (The DAP backend runs deny-all, so a program that calls a host
// capability — e.g. `printf` → `write` — CapFaults; compute-only programs debug cleanly.)
async function startDebug(c) {
  if (broken) return;
  stopReactor();
  if (dapCard) endDebug(dapCard, null); // supersede any running session
  let programText;
  if (c.ex.kind === 'chibicc') {
    setState(c, 'running', 'compiling with -g…');
    let compiled;
    try {
      compiled = await chibiccCompileIR(c);
    } catch (e) {
      setState(c, 'error', `${e.message} — run \`node build-onramp-assets.mjs\` to generate the compiler`);
      return;
    }
    if ((compiled.status !== 0 && compiled.status !== 5) || compiled.ir.length === 0) {
      setState(c, 'error', `compile failed: status ${compiled.status}${compiled.stderr ? ` — ${compiled.stderr.trim()}` : ''}`);
      return;
    }
    programText = compiled.ir;
    logTo(c, `compiled with -g: ${compiled.ir.length}B debuggable Temen IR`);
  } else {
    programText = c.editor.getValue();
  }
  dapClient = createDapClient(eng.ex, eng.memory);
  dapCard = c;
  dapWatch = new Set();
  dapStopped = 1;
  dapThread = 1;
  dapSource = dapSourceName(programText); // breakpoints target the launched program's `debug.file`
  c.el.result.textContent = '';
  c.el.dbgVars.innerHTML = '';
  // Clear the output pane for a chibicc session — the guest's own stdout (its `printf`s under the I/O
  // powerbox) streams in here as it steps, rather than the compile's emitted IR.
  if (c.ex.kind === 'chibicc') c.el.stdout.textContent = '';
  dapClient.send('initialize', {});
  // A chibicc C program runs under the on-ramp I/O powerbox, so a `printf` (a `write` cap) runs and its
  // output streams back as `output` events instead of trapping; a hand-written Temen card stays deny-all.
  const launchArgs = { programText, function: 0, args: [], engine: 'bytecode' };
  if (c.ex.kind === 'chibicc') launchArgs.powerbox = 'onramp';
  const launch = dapClient.send('launch', launchArgs);
  if (!launch.response.success) {
    endDebug(c, null);
    setState(c, 'error', 'debug launch failed — does the program parse and verify?');
    return;
  }
  dapSyncBreakpoints(c);
  c.editor.setReadOnly(true);
  c.el.dbg.classList.add('active');
  c.el.run.disabled = true;
  logTo(c, 'debug session started (bytecode engine) — running to the first breakpoint');
  dapHandle(c, dapClient.send('configurationDone', {}));
}

// A step verb (continue / next / stepIn / stepOut) on the active session.
function debugStep(c, command) {
  if (dapCard !== c || !dapClient) return;
  c.editor.clearStopLine();
  dapHandle(c, dapClient.send(command, {}));
}

// End the session: disconnect, clear the stop highlight, restore the editor.
function endDebug(c, message) {
  if (!dapClient || dapCard !== c) {
    if (message && c) setState(c, 'done', message);
    return;
  }
  dapClient.send('disconnect', {});
  dapClient = null;
  dapCard = null;
  dapWatch = new Set();
  c.editor.clearStopLine();
  c.editor.setReadOnly(false);
  c.el.dbg.classList.remove('active');
  c.el.run.disabled = broken;
  if (message) setState(c, 'done', message);
}

// Temen **text** guests: parse+verify inside the sandbox (`temen_parse`), then run across Workers under the
// card's selected powerbox recipe.
async function runText(c) {
  const mode = c.el.mode.value;
  // The `jit` powerbox recipe runs the guest on the §22 guest-JIT (host-compiled units in the shared
  // Domain) — the wasm-JIT tier for TEMEN-text demos; every other recipe runs on the bytecode engine.
  const rec = runStart(c, { tier: mode === 'jit' ? 'wasm-JIT' : 'interpreter', mode });
  setState(c, 'running', 'parsing…');
  const src = c.editor.getValue();
  c.el.result.textContent = '';
  c.el.stdout.textContent = '';
  c.el.canvas.hidden = true;

  const u8 = () => new Uint8Array(eng.memory.buffer);
  const srcBytes = new TextEncoder().encode(src);
  let guest;
  if (srcBytes.length === 0) {
    setState(c, 'error', 'parse error: empty source');
    runEnd(rec, { ok: false });
    return;
  }
  {
    const tParse = performance.now();
    const p = eng.ex.temen_alloc(srcBytes.length);
    u8().set(srcBytes, p);
    const ok = eng.ex.temen_parse(p, srcBytes.length);
    eng.ex.temen_dealloc(p, srcBytes.length);
    const out = u8().slice(eng.ex.temen_parse_ptr(), eng.ex.temen_parse_ptr() + eng.ex.temen_parse_len());
    if (ok !== 1) {
      const msg = new TextDecoder().decode(out);
      setState(c, 'error', msg);
      c.editor.markError(msg); // pin the offending line in the editor when we can locate it
      runNote(rec, { parseError: msg });
      runEnd(rec, { ok: false });
      return;
    }
    guest = out;
    runStage(rec, 'parse', performance.now() - tParse);
  }
  logTo(c, `parsed: ${srcBytes.length}B text → ${guest.length}B module`);
  runNote(rec, { srcBytes: srcBytes.length, moduleBytes: guest.length });

  aborter = new AbortController();
  c.el.run.disabled = true;
  c.el.stop.disabled = false;
  setState(c, 'running', 'running…');
  const opts = {
    jit: mode === 'jit',
    inst: mode === 'inst',
    io: mode === 'io',
    // Slice 2 (WASM_AOT.md): the compute-only recipe defaults to the wasm-JIT tier-up path — the
    // interpreter drives, hot in-subset functions run on emitted wasm over the same live window
    // (fail-closed per-function; validated interp≡tier-up in `browser-tierup-mainline-test.mjs`). The
    // §22-`jit`/§14-`inst` recipes have their own JIT; `io` stays on the interpreter for now.
    tierup: mode === 'plain',
    winSize: winSizeOf(src),
    signal: aborter.signal,
  };
  const t0 = performance.now();
  try {
    const { value, started, tierups } = await run(guest, opts);
    const tiered = opts.tierup && tierups > 0;
    const label = tiered ? 'interpreter+wasm-JIT' : mode === 'jit' ? 'wasm-JIT' : 'interpreter';
    const ms = runStage(rec, `run:${label}`, performance.now() - t0).toFixed(0);
    c.el.result.textContent = `${value}`;
    if (mode === 'io') c.el.stdout.textContent = readParStdout(eng);
    const tierNote = tiered ? ` · ${tierups} region${tierups === 1 ? '' : 's'} on emitted wasm` : '';
    setState(c, 'done', `done: ${started} Worker${started === 1 ? '' : 's'} · ${ms}ms${tierNote}`);
    logTo(c, `run → ${value} across ${started} Workers in ${ms}ms${tierNote}`);
    runNote(rec, { workers: started, tierups });
    runEnd(rec, { ok: true, result: value });
  } catch (e) {
    if (e.message === 'stopped') {
      // Workers were torn down mid-run; shared state (locks, the live-vCPU counter) may be wedged.
      broken = true;
      setState(c, 'stopped', 'stopped — reload the page to run again');
      logTo(c, 'stopped by user');
      for (const card of cards) { card.el.run.disabled = true; if (card.el.prove) card.el.prove.disabled = true; }
      runNote(rec, { stopped: true });
      runEnd(rec, { ok: false });
    } else {
      setState(c, 'error', `run error: ${e.message}`);
      logTo(c, `run error: ${e.message}`);
      runNote(rec, { runError: e.message });
      runEnd(rec, { ok: false });
    }
  } finally {
    aborter = null;
    c.el.run.disabled = broken;
    c.el.stop.disabled = true;
  }
}

// A card's Run: supersede any running reactor, then dispatch by kind.
async function runDemo(c) {
  if (broken) return;
  if (c.editor) c.editor.clearError();
  if (dapCard) endDebug(dapCard, null); // a fresh Run supersedes any debug session
  stopReactor(); // a fresh Run supersedes any running reactor loop
  const ex = c.ex;
  if (ex.kind === 'reactor') return runReactor(c);
  if (ex.kind === 'pg') return runPg(c);
  if (ex.kind === 'chibicc') return runChibicc(c);
  if (ex.kind === 'selfhost') return runSelfhost(c);
  if (ex.kind === 'nifler') return runNifler(c);
  if (ex.kind === 'nimc') return runNimc(c);
  if (ex.kind === 'shell') return runShell(c);
  if (ex.kind === 'bash') return runBash(c);
  if (ex.kind === 'module') return runModule(c);
  return runText(c);
}

// A card's Stop: close a live Postgres session, end a running reactor, or abort a threaded text run.
function stopDemo(c) {
  if (c.pgSession) {
    eng.ex.temen_pg_close();
    c.pgSession = false;
    c.el.run.disabled = broken;
    c.el.stop.disabled = true;
    // Stop closes the *live* backend but keeps the saved snapshot, so Run reopens the same database.
    setState(c, 'stopped', 'session closed — Run reopens your saved database (`\\reset` for a clean one)');
    return;
  }
  if (reactorRAF !== null) {
    stopReactor();
    c.el.run.disabled = broken;
    c.el.stop.disabled = true;
    setState(c, 'stopped', 'stopped');
  } else {
    aborter?.abort();
  }
}

// ---- DOM: build one card per demo + the sidebar --------------------------------------------------
const slug = (name) => name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

// ---- editor state: persistence + shareable permalinks --------------------------------------------
// Each editable card's source is persisted under its slug so edits survive a reload; "Reset" restores
// the demo's default and drops the saved copy. localStorage is best-effort — a private-mode/quota
// error must never break the page, so every access is guarded.
const STORE_PREFIX = 'temen-play:src:';
const loadSaved = (id) => { try { return localStorage.getItem(STORE_PREFIX + id); } catch { return null; } };
const saveSrc = (id, value, dflt) => {
  try {
    if (value === dflt) localStorage.removeItem(STORE_PREFIX + id); // back to default ⇒ forget it
    else localStorage.setItem(STORE_PREFIX + id, value);
  } catch { /* private mode / quota — persistence is best-effort */ }
};
const clearSaved = (id) => { try { localStorage.removeItem(STORE_PREFIX + id); } catch { /* ignore */ } };

// URL-safe base64 of a UTF-8 string (for the `#src=` permalink payload). Byte-by-byte, not a spread,
// so a large source can't blow the call stack.
function toB64Url(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
function fromB64Url(b64) {
  const bin = atob(b64.replace(/-/g, '+').replace(/_/g, '/'));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

// Build a link that reproduces a card's current editor contents: `…/play.html#demo=<slug>&src=<b64url>`.
const buildShareURL = (id, src) =>
  `${location.origin}${location.pathname}#${new URLSearchParams({ demo: id, src: toB64Url(src) }).toString()}`;

// Copy a permalink for this card to the clipboard (falling back to the address bar if the clipboard is
// blocked — e.g. an insecure context or a denied permission).
async function shareCard(c) {
  const url = buildShareURL(c.id, c.editor.getValue());
  try {
    await navigator.clipboard.writeText(url);
    setState(c, 'done', 'link copied to clipboard');
  } catch {
    location.hash = url.slice(url.indexOf('#') + 1);
    setState(c, 'done', 'link in the address bar — copy it');
  }
  logTo(c, url);
}

// Apply a shared editor state from the URL hash (`#demo=<slug>&src=<b64url>`) once at startup: seed the
// target card's editor and scroll it into view. A bare `#demo=<slug>` (no src) just scrolls to it.
function applyHash() {
  if (!location.hash) return;
  let params;
  try { params = new URLSearchParams(location.hash.slice(1)); } catch { return; }
  const id = params.get('demo');
  if (!id) return;
  const c = cards.find((card) => card.id === id);
  if (!c) return;
  const src = params.get('src');
  if (src != null && c.editor) {
    try { c.editor.setValue(fromB64Url(src)); } catch { /* malformed payload — leave the default */ }
  }
  c.el.section.scrollIntoView({ block: 'start' });
}

const POWERBOX_MODES = [
  ['plain', 'none (compute only)'],
  ['io', 'host I/O (stdout)'],
  ['jit', 'guest JIT (§22)'],
  ['inst', 'instantiator (§14)'],
];

function buildCard(name, ex) {
  const id = slug(name);
  const section = el('section', 'demo');
  section.id = 'demo-' + id;
  section.dataset.demo = name; // stable hook for tests
  section.append(el('h2', 'demo-title', name));
  section.append(el('p', 'desc', ex.desc || ''));

  // Temen text (no `kind`) and editable modules (Lua/SQL/Postgres) get an editor; a fixed C guest or a
  // reactor gets a lightweight read-only note (its "source" is a pre-built binary).
  const editable = !ex.kind || !!ex.editable;
  const dflt = ex.src || '';
  let editor = null;
  if (editable) {
    const ta = el('textarea');
    ta.value = dflt;
    const wrap = el('div', 'editor');
    wrap.appendChild(ta);
    section.appendChild(wrap);
    editor = createEditor(ta, ex.lang || 'temen');
    // Restore a previously edited source, then persist every edit under this card's slug.
    const saved = loadSaved(id);
    if (saved != null && saved !== dflt) editor.setValue(saved);
    editor.onChange(() => saveSrc(id, editor.getValue(), dflt));
    // Debug-capable cards: a gutter click toggles a breakpoint (live-synced to an active session), and
    // the demo may pre-place one. (`c` is referenced by the click closure, which only fires post-build.)
    if (ex.debug) {
      if (ex.bp != null) editor.toggleBreakpoint(ex.bp);
      editor.onGutterClick((line) => {
        editor.toggleBreakpoint(line);
        if (dapCard === c) dapSyncBreakpoints(c);
      });
    }
  } else {
    section.appendChild(el('pre', 'note',
      ex.kind === 'reactor'
        ? `Pre-built on-ramp reactor module (${ex.url}). Click Run — the page calls tick() once per animation frame; the arrow keys steer it through the keyboard capability.`
        : ex.kind === 'selfhost'
        ? `chibicc compiling its own source. Pick one of chibicc’s cc1 translation units and click Run — chibicc.temen compiles that file to a linkable TEMEN-IR object, reading its ~96-file glibc header closure from the seeded in-memory filesystem, entirely in your browser. The emitted object appears below; "Prove interp ≡ JIT" recompiles it on both engines and checks they’re byte-identical.`
        : `Pre-built on-ramp module (${ex.url}). Click Run — it executes as a real C/C++ guest via temen_run_onramp; its stdout appears below.`));
  }

  const controls = el('div', 'controls');
  let modeSel = null;
  if (!ex.kind) {
    modeSel = el('select');
    for (const [v, label] of POWERBOX_MODES) {
      const o = el('option', null, label);
      o.value = v;
      modeSel.appendChild(o);
    }
    modeSel.value = ex.mode;
    const l = el('label', null, 'powerbox ');
    l.appendChild(modeSel);
    controls.appendChild(l);
  }
  // The self-host card picks which of chibicc's own cc1 TUs to compile; the option value is the TU's
  // memfs-relative path (its key in the seeded closure image), read by `runSelfhost`.
  let tuSel = null;
  if (ex.kind === 'selfhost') {
    tuSel = el('select');
    for (const tu of ex.tus || []) {
      const o = el('option', null, tu);
      o.value = `frontend/chibicc/${tu}`;
      tuSel.appendChild(o);
    }
    const l = el('label', null, 'translation unit ');
    l.appendChild(tuSel);
    controls.appendChild(l);
  }
  const runBtn = el('button', 'run', 'Run');
  runBtn.disabled = true;
  const stopBtn = el('button', 'stop', 'Stop');
  stopBtn.disabled = true;
  controls.append(runBtn, stopBtn);
  // Editable cards get Reset (restore the demo's default source) + Share (copy a permalink of the
  // current editor contents). A fixed/reactor card has no editable source, so neither applies.
  let resetBtn = null, shareBtn = null;
  if (editable) {
    resetBtn = el('button', 'reset', 'Reset');
    resetBtn.title = 'Restore this demo’s original source';
    shareBtn = el('button', 'share', 'Share');
    shareBtn.title = 'Copy a link that reproduces the current editor contents';
    controls.append(resetBtn, shareBtn);
  }
  // The chibicc C card gets a "debug info (-g)" checkbox: off by default (chibicc then compiles clean,
  // fast IR — the `debug.*` waist is ~a third of the output). Ticking it makes chibicc emit the source
  // waist (so Run shows the `debug.*` sections) and enables source-level C debugging via the Debug
  // button below. Default-on for a card that carries `gOn` (the ready-to-debug demo).
  let gflag = null;
  if (ex.kind === 'chibicc') {
    const l = el('label', 'jit-label');
    l.title = 'Emit chibicc’s debug info (source lines + variable names). Off = clean, fast IR; on = debuggable.';
    gflag = el('input');
    gflag.type = 'checkbox';
    gflag.checked = !!ex.gOn;
    l.append(gflag, ' debug info (-g)');
    controls.appendChild(l);
  }
  // A debug-capable card gets a Debug button (starts a DAP session on the bytecode engine).
  let debugBtn = null;
  if (ex.debug) {
    debugBtn = el('button', 'debug', 'Debug');
    debugBtn.title = ex.kind === 'chibicc'
      ? 'Compile with -g and debug the C at source level — breakpoints on C lines, C locals by name (tick "debug info (-g)" first)'
      : 'Debug this Temen program on the bytecode engine — breakpoints, stepping, variables';
    debugBtn.disabled = true;
    controls.appendChild(debugBtn);
  }
  let jit = null;
  let proveBtn = null;
  if (ex.jit) {
    // A reactor emits its per-frame tick(); a module (and the chibicc compiler) emits the whole _start.
    // The parity check compares the framebuffer (reactor, per frame) or the stdout / emitted IR
    // (module / chibicc, run-to-completion) accordingly.
    const isModule = ex.kind === 'module' || ex.kind === 'chibicc' || ex.kind === 'selfhost';
    const l = el('label', 'jit-label');
    l.title = isModule
      ? 'Run the whole guest (_start) on emitted wasm (wasm-JIT tier) instead of the interpreter'
      : 'Run the reactor’s tick() on emitted wasm (wasm-JIT tier) instead of the interpreter';
    jit = el('input');
    jit.type = 'checkbox';
    // A warm-snapshot card (QuickJS) defaults to the warm path (checkbox off); ticking it opts into the
    // cold wasm-JIT tier. Every other jit card defaults to the JIT tier on.
    jit.checked = !ex.warm;
    l.append(jit, ' wasm-JIT');
    controls.appendChild(l);
    // "Prove it": run the guest on both tiers and assert the result is byte-identical.
    proveBtn = el('button', 'prove', 'Prove interp ≡ JIT');
    proveBtn.title = isModule
      ? 'Run the guest on the interpreter and the wasm-JIT tier and check stdout is byte-identical'
      : 'Run 30 frames on the interpreter and the wasm-JIT tier and check the framebuffer is byte-identical';
    proveBtn.disabled = true;
    controls.appendChild(proveBtn);
  }
  const state = el('span', 'state', 'ready');
  state.dataset.state = 'ready';
  controls.appendChild(state);
  section.appendChild(controls);

  const out = el('div', 'output');
  const result = el('pre', 'result');
  const canvas = el('canvas', 'canvas');
  canvas.hidden = true;
  const gpucanvas = el('canvas', 'gpucanvas');
  gpucanvas.hidden = true;
  const stdout = el('pre', 'stdout');
  const logEl = el('pre', 'log');
  out.append(el('strong', null, 'result'), result, canvas, gpucanvas,
    el('strong', null, 'stdout'), stdout, el('strong', null, 'log'), logEl);
  section.appendChild(out);

  // On-screen dpad for the interactive reactors: arrows steer, plus the action keys Doom's menus/play
  // use. Only rendered for reactor cards; CSS shows it on touch / narrow screens. Each button dispatches
  // the same keyboard-cap event as the physical key (pressed on pointerdown, released on up/leave).
  if (ex.kind === 'reactor') {
    const pad = el('div', 'dpad');
    // [label, JS keyCode] — arrows (37/38/40/39), fire (Ctrl 17), use (Space 32), enter (13), esc (27).
    for (const [label, code] of [['←', 37], ['↑', 38], ['↓', 40], ['→', 39], ['fire', 17], ['use', 32], ['↵', 13], ['esc', 27]]) {
      const b = el('button', 'dkey', label);
      b.type = 'button';
      b.dataset.key = String(code);
      const press = (down) => (ev) => { ev.preventDefault(); sendReactorKey(code, down ? 1 : 0); };
      b.addEventListener('pointerdown', press(true));
      b.addEventListener('pointerup', press(false));
      b.addEventListener('pointerleave', press(false));
      b.addEventListener('pointercancel', press(false));
      pad.appendChild(b);
    }
    section.appendChild(pad);
  }

  // Debugger panel (DAP over the bytecode engine): step controls + a live Variables pane. Hidden until
  // a session pauses (`.dbg.active`). Only built for debug-capable cards.
  let dbg = null, dbgVars = null;
  if (ex.debug) {
    dbg = el('div', 'dbg');
    const dc = el('div', 'dbg-controls');
    const mk = (label, title, cmd) => {
      const b = el('button', null, label);
      b.title = title;
      b.dataset.cmd = cmd || 'stop';
      b.addEventListener('click', () => (cmd ? debugStep(c, cmd) : endDebug(c, 'debug session ended')));
      return b;
    };
    dc.append(
      mk('▶ Continue', 'Run to the next breakpoint', 'continue'),
      mk('⤼ Step Over', 'Step over the next source line', 'next'),
      mk('↳ Step In', 'Step into a call', 'stepIn'),
      mk('↰ Step Out', 'Run to the caller', 'stepOut'),
      // Reverse debugging (bytecode backend, deterministic replay): step one op back, or run backward
      // to the previous breakpoint.
      mk('⤺ Step Back', 'Re-execute to one step earlier', 'stepBack'),
      mk('◀◀ Reverse', 'Run backward to the previous breakpoint', 'reverseContinue'),
      mk('■ Stop', 'End the debug session', null),
    );
    dbgVars = el('pre', 'dbg-vars');
    dbg.append(dc, dbgVars);
    section.appendChild(dbg);
  }

  const c = {
    name, ex, editor, id,
    el: { section, state, result, stdout, log: logEl, canvas, gpucanvas, run: runBtn, stop: stopBtn, mode: modeSel, tu: tuSel, jit, gflag, prove: proveBtn, reset: resetBtn, share: shareBtn, debug: debugBtn, dbg, dbgVars },
  };
  runBtn.addEventListener('click', () => runDemo(c));
  if (debugBtn) debugBtn.addEventListener('click', () => startDebug(c));
  // chibicc: the Debug button needs debug info, so it tracks the "-g" checkbox (source-level C debugging
  // is only possible with the emitted `debug.*` waist). Non-chibicc debug cards leave it engine-gated.
  if (gflag && debugBtn) {
    const syncDbg = () => { debugBtn.disabled = broken || !engineReady || !gflag.checked; };
    gflag.addEventListener('change', syncDbg);
    c._syncDbgBtn = syncDbg;
  }
  // Clicking a variable's ● toggle arms/clears a data breakpoint; clicking a thread button focuses that
  // thread's stack (delegated: the Variables pane is re-rendered on every stop, so the listener lives
  // on the stable container).
  if (dbgVars) dbgVars.addEventListener('click', (ev) => {
    const w = ev.target.closest('button[data-watch]');
    if (w && !w.disabled) { dapToggleWatch(c, w.dataset.watch); return; }
    const t = ev.target.closest('button[data-thread]');
    if (t) dapSelectThread(c, Number(t.dataset.thread));
  });
  stopBtn.addEventListener('click', () => stopDemo(c));
  if (proveBtn) {
    const prove = c.ex.kind === 'chibicc' ? proveChibiccParity : c.ex.kind === 'selfhost' ? proveSelfhostParity : c.ex.kind === 'module' ? proveModuleParity : proveParity;
    proveBtn.addEventListener('click', () => prove(c));
  }
  if (resetBtn) resetBtn.addEventListener('click', () => {
    editor.setValue(dflt);
    clearSaved(id);
    editor.clearError();
    setState(c, 'ready', 'reset to the original source');
  });
  if (shareBtn) shareBtn.addEventListener('click', () => shareCard(c));
  return c;
}

// The sidebar: one link per demo, scroll-spied so the in-view demo is highlighted, and a global Vim
// toggle. Clicking a link scrolls its card into view.
function buildSidebar() {
  const nav = $('nav-list');
  for (const c of cards) {
    const a = el('a', 'nav-link', c.name);
    a.href = '#' + c.el.section.id;
    a.dataset.target = c.el.section.id;
    nav.appendChild(a);
  }
  // Scroll-spy: highlight the link whose card is nearest the top of the viewport.
  const links = new Map([...nav.querySelectorAll('.nav-link')].map((a) => [a.dataset.target, a]));
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const a = links.get(entry.target.id);
      if (a) a.classList.toggle('active', entry.isIntersecting);
    }
  }, { rootMargin: '-45% 0px -45% 0px' }); // a thin band across the vertical middle
  for (const c of cards) observer.observe(c.el.section);
}

// Theme picker: the head script already resolved the initial `data-theme` from the stored preference;
// here we seed the sidebar select and keep it live — persisting the choice and re-resolving `auto`
// against the OS as it changes.
function setupTheme() {
  const sel = $('theme');
  let stored = 'auto';
  try { stored = localStorage.getItem('temen-play:theme') || 'auto'; } catch { /* private mode */ }
  sel.value = stored;
  const mq = matchMedia('(prefers-color-scheme: dark)');
  const apply = (pref) => {
    const dark = pref === 'dark' || (pref === 'auto' && mq.matches);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  };
  sel.addEventListener('change', () => {
    try { localStorage.setItem('temen-play:theme', sel.value); } catch { /* best-effort */ }
    apply(sel.value);
  });
  mq.addEventListener('change', () => { if (sel.value === 'auto') apply('auto'); }); // follow the OS live
}

async function main() {
  const demosEl = $('demos');
  for (const [name, ex] of Object.entries(EXAMPLES)) {
    const c = buildCard(name, ex);
    cards.push(c);
    demosEl.appendChild(c.el.section);
  }
  buildSidebar();
  refreshAll(); // lay the editors out now they're in the DOM
  applyHash();  // seed a card's editor from a shared #demo=…&src=… permalink, if present
  setupTheme();
  $('vim').addEventListener('change', (e) => setVimAll(e.target.checked));

  // Forward keys to the running reactor guest through the `keyboard` capability (as JS keyCodes — the
  // guest maps them: bounce steers on the arrows; Doom adds Ctrl fire / Space use / Enter·Esc·Tab
  // menus / Shift run / the letter keys). Only while a loop is running. `preventDefault` is limited to
  // the keys whose default would disrupt play (arrows/Space/Tab scroll or move focus), and never fires
  // for a browser shortcut (Ctrl/Meta + a letter — e.g. Ctrl+R), so reload etc. still work.
  const REACTOR_KEYS = new Set([37, 38, 39, 40, 17, 32, 13, 27, 9, 16]);
  for (let k = 65; k <= 90; k++) REACTOR_KEYS.add(k);
  const SWALLOW = new Set([37, 38, 39, 40, 32, 9]);
  const forward = (pressed) => (e) => {
    if (reactorRAF === null || !REACTOR_KEYS.has(e.keyCode)) return;
    sendReactorKey(e.keyCode, pressed);
    const shortcut = (e.ctrlKey || e.metaKey) && e.keyCode !== 17;
    if (SWALLOW.has(e.keyCode) && !shortcut) e.preventDefault();
  };
  window.addEventListener('keydown', forward(1));
  window.addEventListener('keyup', forward(0));

  if (!self.crossOriginIsolated) {
    setEngineState('error', 'no cross-origin isolation (SharedArrayBuffer unavailable) — serve via serve.mjs');
    return;
  }
  try {
    eng = await loadEngine();
    run = makeRunner(eng);
  } catch (e) {
    setEngineState('error', `engine load failed: ${e.message}`);
    return;
  }
  engineReady = true;
  for (const c of cards) {
    c.el.run.disabled = false;
    if (c.el.prove) c.el.prove.disabled = false;
    // chibicc's Debug tracks its "-g" checkbox (source-level debugging needs the emitted debug info);
    // every other debug card enables unconditionally now the engine is up.
    if (c._syncDbgBtn) c._syncDbgBtn();
    else if (c.el.debug) c.el.debug.disabled = false;
  }
  setEngineState('ready', 'engine ready');

  // Spin up the snapshot worker (issue #804) and pre-warm every warm card off the main thread, so the
  // ~one-time QuickJS warmup is done before the user's first Run and never blocks the UI. Best-effort: if
  // the worker fails to start, warm cards silently fall back to the main-thread warm path on Run. Deferred
  // to idle so it doesn't compete with initial page setup, and the 4 MiB snapshot fetch stays off the
  // critical path. Each pre-warming card shows a "warming up…" indicator until its session is ready.
  try {
    snapshotClient = new SnapshotClient(eng.module);
    globalThis.__snapshotClient = snapshotClient; // test/telemetry hook (harmless): inspect prewarm state
    const warmCards = cards.filter((c) => c.ex.warm);
    if (warmCards.length) {
      const prewarmAll = () => {
        for (const c of warmCards) {
          setState(c, 'warming', 'warming up runtime…');
          snapshotClient
            // `primeJit` (the card's `jit` flag) pre-compiles the warm+JIT tier during pre-warm — but only
            // for cards that actually use it (QuickJS/Lua); Tcl's warm+JIT declines, so we skip it there.
            .prewarm(c.ex.url, () => fetchModule(c.ex.url), !!c.ex.jit)
            .then((r) => setState(c, 'ready', r.ok ? 'runtime warm — Run is instant' : 'ready'))
            .catch(() => setState(c, 'ready', 'ready'));
        }
      };
      if (typeof requestIdleCallback === 'function') requestIdleCallback(prewarmAll, { timeout: 3000 });
      else setTimeout(prewarmAll, 0);
    }
  } catch (e) {
    snapshotClient = null;
    console.warn('[Temen playground] snapshot worker unavailable; warm cards use the main thread:', e.message);
  }
}

main().catch((e) => setEngineState('error', `fatal: ${e.message}`));
