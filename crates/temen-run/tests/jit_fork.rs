//! #1768 — a personality `fork` (and the blocking `waitpid` that reaps it) answered **identically on
//! every engine**, the Cranelift JIT included (FORK.md §9.5).
//!
//! The JIT forks by durable unwind: the forking caller's frames spill into its window's shadow stack,
//! the window is duplicated, and both copies rewind past the fork call with their own reply. So what
//! these guests pin is what that mechanism could get wrong and the interpreters' vCPU clone cannot:
//! values live across the fork call in the frames below it, a fork inside a loop, the twin's window
//! being a *copy* (its writes never reach the parent), a twin forking again, the pids both copies
//! see, and a twin that crashes. Each guest runs on the tree-walk oracle, the bytecode engine and the
//! JIT, and every engine must produce the same outcome.
//!
//! `caller_request_parity.rs` covers the call forms and grant shapes of the same requests; this file
//! covers the shapes of the program around them.

use temen_run::{
    instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig, SharedHostProc,
};
use temen_text::parse_module;

/// The personality slots every guest declares, and the window: the shadow arena is where the JIT
/// unwinds a forking caller to — its placement is the module's (INVARIANTS.md #16), clear of the data
/// cells the guests use below it.
const HEAD: &str = "memory 17 shadow 65536 69632\n\
import 0 \"fork\" () -> (i64)\n\
import 1 \"wait4\" (i64, i64, i64, i64) -> (i64)\n\
import 2 \"exit\" (i32) -> ()\n";

/// Run `src` (after [`HEAD`], or bare when `head` is false) on `backend` over a POSIX personality.
fn run_on(src: &str, head: bool, backend: Backend) -> Outcome {
    let text = if head {
        format!("{HEAD}{src}")
    } else {
        src.to_string()
    };
    let module = parse_module(&text).expect("parse guest");
    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let slot = SharedHostProc::new(make, temen_posix::cap_fork_factory(&posix));
    let imports = Imports::new()
        .provide(
            "fork",
            HostCap::host_proc_shared(temen_posix::OP_FORK, &slot),
        )
        .provide(
            "wait4",
            HostCap::host_proc_shared(temen_posix::OP_WAIT4, &slot),
        )
        .provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(module, imports).expect("instantiate");
    let p = posix.clone();
    let mut setup = move |host: &mut temen_interp::Host| {
        let handle = slot.install(host);
        let (door, armed) = temen_posix::cap_signal_source(&p);
        host.set_signal_source(door, armed);
        let (names, sigs) = temen_posix::cap_vtable();
        host.set_host_proc_vtable(handle, names, sigs);
    };
    inst.run_with_caps_and_host(backend, &RunConfig::default(), &[], Some(&mut setup))
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"))
        .outcome
}

/// Every engine must run `src` to `want`.
fn assert_every_engine(src: &str, want: Outcome) {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        assert_eq!(run_on(src, true, backend), want, "{backend:?}");
    }
}

/// Three children forked in a loop, reaped by `wait4(-1)`. Before each fork the parent writes `10*i`
/// to a cell; the child reads it (its window is a copy), overwrites it with `999`, and exits
/// `11*i + 1` (`cell + i + 1`). After each fork the parent checks the cell still holds `10*i` — a
/// twin's write reaching its parent exits `1`. The parent then reaps all three, and exits with the
/// sum of the statuses (`1 + 12 + 23`), of the pids `fork` returned (`2 + 3 + 4`), and of the pids
/// `wait4` returned (the same three): `36 + 9 + 9 = 54`.
///
/// The loop counter and the running pid sum are live across the fork call, so each fork's unwind
/// carries them through the shadow stack; and `wait4` usually blocks, since each child is still
/// starting when its parent reaches it.
const FORK_LOOP: &str = "\
func () -> () {
block 0 () {
  vi0 = i64.const 0
  vs0 = i64.const 0
  br 1(vi0, vs0)
  }
block 1 (li: i64, lsum: i64) {
  la = i64.const 40000
  lten = i64.const 10
  lx = i64.mul li lten
  i64.store la lx
  lpid = call.import 0 ()
  lz = i64.const 0
  lchild = i64.eq lpid lz
  br_if lchild 2(li) 3(li, lsum, lpid)
  }
block 2 (ci: i64) {
  ca = i64.const 40000
  cv = i64.load ca
  c999 = i64.const 999
  i64.store ca c999
  cone = i64.const 1
  cs = i64.add cv ci
  cs2 = i64.add cs cone
  cc = i32.wrap_i64 cs2
  call.import 2 (cc)
  unreachable
  }
block 3 (pi: i64, psum: i64, ppid: i64) {
  pa = i64.const 40000
  pv = i64.load pa
  pten = i64.const 10
  pwant = i64.mul pi pten
  pok = i64.eq pv pwant
  br_if pok 4(pi, psum, ppid) 7()
  }
block 4 (qi: i64, qsum: i64, qpid: i64) {
  qs = i64.add qsum qpid
  qone = i64.const 1
  qn = i64.add qi qone
  qthree = i64.const 3
  qmore = i64.lt_s qn qthree
  qk = i64.const 0
  br_if qmore 1(qn, qs) 5(qs, qk)
  }
block 5 (rsum: i64, rk: i64) {
  rany = i64.const -1
  rst = i64.const 41000
  rz = i64.const 0
  rw = call.import 1 (rany, rst, rz, rz)
  rhi = i64.const 41001
  rcode8 = i32.load8_u rhi
  rcode = i64.extend_i32_u rcode8
  rs1 = i64.add rsum rcode
  rs2 = i64.add rs1 rw
  rone = i64.const 1
  rk1 = i64.add rk rone
  rthree = i64.const 3
  rmore = i64.lt_s rk1 rthree
  br_if rmore 5(rs2, rk1) 6(rs2)
  }
block 6 (dsum: i64) {
  dc = i32.wrap_i64 dsum
  call.import 2 (dc)
  unreachable
  }
block 7 () {
  xone = i32.const 1
  call.import 2 (xone)
  unreachable
  }
}
export 0 func \"_start\" 0
";

#[test]
fn forks_in_a_loop_copy_the_window_and_reap_identically_on_every_engine() {
    assert_every_engine(FORK_LOOP, Outcome::Exited(54));
}

/// The fork three calls deep, with values live across it in every frame: `f1(7)` computes
/// `y = 21` and calls `f2(y)`, which computes `z = y + 5` and forks; `f2` returns `pid*100 + z` and
/// `f1` adds its `y`, so each copy's `_start` sees `pid*100 + 47`. The child exits `47`; the parent
/// (pid `2` ⇒ `247`) reaps it and exits `status + (r - 200)` = `47 + 47 = 94`.
const FORK_DEEP: &str = "\
func () -> () {
block 0 () {
  a = i64.const 7
  r = call 1 (a)
  h = i64.const 100
  lt = i64.lt_s r h
  br_if lt 1(r) 2(r)
  }
block 1 (cr: i64) {
  cc = i32.wrap_i64 cr
  call.import 2 (cc)
  unreachable
  }
block 2 (pr: i64) {
  pb = i64.const 247
  pok = i64.eq pr pb
  br_if pok 3(pr) 4()
  }
block 3 (qr: i64) {
  qkid = i64.const 2
  qst = i64.const 41000
  qz = i64.const 0
  qw = call.import 1 (qkid, qst, qz, qz)
  qhi = i64.const 41001
  qcode8 = i32.load8_u qhi
  qcode = i64.extend_i32_u qcode8
  q200 = i64.const 200
  qd = i64.sub qr q200
  qs = i64.add qcode qd
  qc = i32.wrap_i64 qs
  call.import 2 (qc)
  unreachable
  }
block 4 () {
  xone = i32.const 1
  call.import 2 (xone)
  unreachable
  }
}
func (i64) -> (i64) {
block 0 (x: i64) {
  three = i64.const 3
  y = i64.mul x three
  r = call 2 (y)
  s = i64.add r y
  return s
  }
}
func (i64) -> (i64) {
block 0 (y: i64) {
  five = i64.const 5
  z = i64.add y five
  p = call.import 0 ()
  h = i64.const 100
  ph = i64.mul p h
  s = i64.add ph z
  return s
  }
}
export 0 func \"_start\" 0
";

#[test]
fn a_fork_deep_in_the_call_stack_resumes_every_frame_identically_on_every_engine() {
    assert_every_engine(FORK_DEEP, Outcome::Exited(94));
}

/// A twin forks again. The grandchild exits `5`; the child (which saw pid `3` for it — the next twin
/// pid, whoever forks) reaps it and exits `10 + 5 + 3 = 18`; the root reaps the child and exits
/// `100 + 18 = 118`.
const FORK_NESTED: &str = "\
func () -> () {
block 0 () {
  p = call.import 0 ()
  z = i64.const 0
  child = i64.eq p z
  br_if child 1() 3(p)
  }
block 1 () {
  q = call.import 0 ()
  z1 = i64.const 0
  gchild = i64.eq q z1
  br_if gchild 4() 2(q)
  }
block 2 (cq: i64) {
  cst = i64.const 41000
  cz = i64.const 0
  cw = call.import 1 (cq, cst, cz, cz)
  chi = i64.const 41001
  ccode8 = i32.load8_u chi
  ccode = i64.extend_i32_u ccode8
  cten = i64.const 10
  cs1 = i64.add cten ccode
  cs2 = i64.add cs1 cq
  cc = i32.wrap_i64 cs2
  call.import 2 (cc)
  unreachable
  }
block 3 (pp: i64) {
  pst = i64.const 42000
  pz = i64.const 0
  pw = call.import 1 (pp, pst, pz, pz)
  phi = i64.const 42001
  pcode8 = i32.load8_u phi
  pcode = i64.extend_i32_u pcode8
  p100 = i64.const 100
  ps = i64.add p100 pcode
  pc = i32.wrap_i64 ps
  call.import 2 (pc)
  unreachable
  }
block 4 () {
  gfive = i32.const 5
  call.import 2 (gfive)
  unreachable
  }
}
export 0 func \"_start\" 0
";

#[test]
fn a_twin_forks_its_own_twin_identically_on_every_engine() {
    assert_every_engine(FORK_NESTED, Outcome::Exited(118));
}

/// The child traps (`unreachable`); its parent reaps the crash status, `128`, and exits with it. A
/// crashing child never crashes the parent (STAGE1.md), on any engine — and the engines that record
/// twin traps name it.
const FORK_CRASH: &str = "\
func () -> () {
block 0 () {
  p = call.import 0 ()
  z = i64.const 0
  child = i64.eq p z
  br_if child 1() 2(p)
  }
block 1 () {
  unreachable
  }
block 2 (pp: i64) {
  pst = i64.const 41000
  pz = i64.const 0
  pw = call.import 1 (pp, pst, pz, pz)
  phi = i64.const 41001
  pcode8 = i32.load8_u phi
  call.import 2 (pcode8)
  unreachable
  }
}
export 0 func \"_start\" 0
";

#[test]
fn a_crashing_twin_reaps_as_128_identically_on_every_engine() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        assert_eq!(
            run_on(FORK_CRASH, true, backend),
            Outcome::Exited(128),
            "{backend:?}"
        );
        // The bytecode engine keeps no twin-trap record; the other two name the trap.
        if backend != Backend::Bytecode {
            let traps = temen_interp::last_twin_traps();
            assert_eq!(traps.len(), 1, "{backend:?}: {traps:?}");
            assert_eq!(traps[0].task, 2, "{backend:?}");
            assert_eq!(
                traps[0].trap,
                temen_interp::Trap::Unreachable,
                "{backend:?}"
            );
        }
    }
}

/// The twin inherits the forking thread's `vcpu.tls` register, as POSIX's child inherits the rest
/// of the thread — a guest that keeps its TLS block's address there (the fs-base recipe) finds the
/// same block in its copy of the window. The parent sets the register to `50000` and forks; the
/// child exits `7` if it reads `50000` back (`1` otherwise); the parent reaps it and exits with its
/// status, or `2` if its own register moved.
const FORK_TLS: &str = "\
func () -> () {
block 0 () {
  t = i64.const 50000
  vcpu.tls.set t
  p = call.import 0 ()
  z = i64.const 0
  child = i64.eq p z
  br_if child 1() 2(p)
  }
block 1 () {
  ct = vcpu.tls.get
  cw = i64.const 50000
  cok = i64.eq ct cw
  br_if cok 3() 4()
  }
block 2 (pp: i64) {
  pt = vcpu.tls.get
  pw = i64.const 50000
  pok = i64.eq pt pw
  br_if pok 5(pp) 6()
  }
block 3 () {
  seven = i32.const 7
  call.import 2 (seven)
  unreachable
  }
block 4 () {
  one = i32.const 1
  call.import 2 (one)
  unreachable
  }
block 5 (qp: i64) {
  qst = i64.const 41000
  qz = i64.const 0
  qw = call.import 1 (qp, qst, qz, qz)
  qhi = i64.const 41001
  qcode8 = i32.load8_u qhi
  call.import 2 (qcode8)
  unreachable
  }
block 6 () {
  two = i32.const 2
  call.import 2 (two)
  unreachable
  }
}
export 0 func \"_start\" 0
";

#[test]
fn a_twin_inherits_the_forking_threads_tls_register_on_every_engine() {
    assert_every_engine(FORK_TLS, Outcome::Exited(7));
}

/// FORK.md §9.5 — the JIT forks a program only where it can unwind the forking caller: into a shadow
/// arena the module declares (INVARIANTS.md #16 — there is no default placement). A program without
/// one gets `-ENOSYS` ("unavailable on this tier") from its `fork` — a probeable refusal, never a
/// wrong answer — where the interpreters fork it. Here the parent exits `100 + errno` on a refusal
/// and `pid` otherwise.
const FORK_NO_ARENA: &str = "memory 17\n\
import 0 \"fork\" () -> (i64)\n\
import 1 \"wait4\" (i64, i64, i64, i64) -> (i64)\n\
import 2 \"exit\" (i32) -> ()\n\
func () -> () {
block 0 () {
  p = call.import 0 ()
  z = i64.const 0
  child = i64.eq p z
  neg = i64.lt_s p z
  br_if child 1() 2(p, neg)
  }
block 1 () {
  czero = i32.const 0
  call.import 2 (czero)
  unreachable
  }
block 2 (pp: i64, pneg: i32) {
  br_if pneg 3(pp) 4(pp)
  }
block 3 (ep: i64) {
  e100 = i64.const 100
  e = i64.sub e100 ep
  ec = i32.wrap_i64 e
  call.import 2 (ec)
  unreachable
  }
block 4 (op: i64) {
  oc = i32.wrap_i64 op
  call.import 2 (oc)
  unreachable
  }
}
export 0 func \"_start\" 0
";

#[test]
fn without_a_shadow_arena_the_jit_refuses_fork_probeably() {
    for backend in [Backend::TreeWalk, Backend::Bytecode] {
        assert_eq!(
            run_on(FORK_NO_ARENA, false, backend),
            Outcome::Exited(2),
            "{backend:?}"
        );
    }
    assert_eq!(
        run_on(FORK_NO_ARENA, false, Backend::Jit),
        Outcome::Exited(100 + 38),
        "the JIT answers -ENOSYS"
    );
}
