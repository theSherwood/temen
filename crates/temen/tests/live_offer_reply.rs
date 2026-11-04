//! #2738, INVARIANTS #8 — a call through a **live offer** (a call queued into another running
//! domain) replies with exactly the op's declared results, its handles translated into the caller's
//! table, as a call through an instanced offer does. Each path the call can take is pinned: served
//! inline by direct handoff, a parked root, and a parked fiber.
//!
//! A transport failure reaches the caller as its errno in the first result when that result is an
//! integer, the rest zero; when it is not, the call traps.
//!
//! The caller spawns the serving child detached (op 17), mints a live offer over its export (op 14),
//! makes the call, joins the child, and returns `join · 1_000_000 + call`: the child's entry serves
//! one dispatch (`svc.wait`) and returns how many it served. Each case runs on the tree-walker, the
//! bytecode engine's pump and executor 2, and the Cranelift JIT, which must agree (INVARIANTS #9). On the JIT a caller
//! without a fiber holds its powerbox outright, and one with a fiber shares it in a cell.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{bytecode, run_with_host, run_with_host_fast, Host, MemLayout, Trap, Value};
use temen_jit::JitOutcome;

/// `CAP_REVOKED`: the errno a call through a dead callee answers.
const CAP_REVOKED: i64 = -9;

/// The caller around `call`, a snippet that defines `vcall: i64` from the offer's handle `v7` and the
/// caller's clock `v3`. Func 1 is the fiber [`IN_A_FIBER`] drives, when a test supplies one.
fn caller(call: &str, fiber: &str) -> String {
    format!(
        r#"memory 17

func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {{
  vrm = i64.const 17416
  i32.store vrm v1
  vrb = i64.const 17420
  i32.store vrb v2
  vrp = i64.const 17408
  v5 = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  v6 = i64.const 0
  v7 = call.cap 6 14 (i32, i64) -> (i32) v0 (v5, v6)
{call}
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) v0 (v5)
  vk = i64.const 1000000
  vm = i64.mul vj vk
  vs = i64.add vm vcall
  return vs
  }}
}}
{fiber}{rec}"#,
        rec = rec::segment(17408, &temen_ir::SpawnRec::default())
    )
}

/// A child that serves one dispatch of its offer, whose types and handler `serve` gives.
fn server(types: &str, handler: &str) -> String {
    format!(
        r#"memory 12
{types}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }}
}}

{handler}"#
    )
}

/// The caller's entry drives fiber func 1 with `cont.resume.block`, passing the offer's handle, until
/// it returns; its return is the call's value.
const IN_A_FIBER: &str = r#"
  vf = ref.func 1
  vfsp = i64.const 24576
  vfk = cont.new vf vfsp
  voff = i64.extend_i32_u v7
  vfs, vfv = cont.resume.block vfk voff
  br 1(v0, v5, vfk, voff, vfs, vfv)
}
block 1 (w0: i32, w5: i32, wk: i64, woff: i64, ws: i32, wv: i64) {
  vone = i32.const 1
  vdone = i32.eq ws vone
  br_if vdone 3(w0, w5, wv) 2(w0, w5, wk, woff)
}
block 2 (x0: i32, x5: i32, xk: i64, xoff: i64) {
  xs, xv = cont.resume.block xk xoff
  br 1(x0, x5, xk, xoff, xs, xv)
}
block 3 (v0: i32, v5: i32, vcall: i64) {"#;

/// The calls of the two-result tests: `addmul(40, 2)` is `(42, 80)`, folded to `42_080`.
const ADDMUL_CALL: &str = r#"
  va = i64.const 40
  vb = i64.const 2
  vsum, vprod = call.cap 268435456 0 (i64, i64) -> (i64, i64) v7 (va, vb)
  vthousand = i64.const 1000
  vsh = i64.mul vsum vthousand
  vcall = i64.add vsh vprod"#;

/// Fiber func 1 of the two-result fiber test: the same call, from inside the fiber.
const ADDMUL_FIBER: &str = r#"
func (i64, i64) -> (i64) {
block 0 (vsp: i64, voff: i64) {
  v7 = i32.wrap_i64 voff
  va = i64.const 40
  vb = i64.const 2
  vsum, vprod = call.cap 268435456 0 (i64, i64) -> (i64, i64) v7 (va, vb)
  vthousand = i64.const 1000
  vsh = i64.mul vsum vthousand
  vr = i64.add vsh vprod
  return vr
  }
}
"#;

const ADDMUL_TYPES: &str = "type 0 func (i64, i64) -> (i64, i64)
type 1 interface { addmul: 0 }
export 0 interface \"addmul\" 1 { addmul: 1 }";

const ADDMUL: &str = r#"func (i64, i64) -> (i64, i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  vm = i64.mul va vb
  return vs, vm
  }
}
"#;

fn module(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    m
}

/// An engine's run of the caller's entry over `host`.
type Entry = fn(&temen_ir::Module, &[i32; 4], &mut Host) -> Result<Vec<Value>, Trap>;

fn tree_walker(m: &temen_ir::Module, args: &[i32; 4], host: &mut Host) -> Result<Vec<Value>, Trap> {
    let mut fuel = 5_000_000u64;
    run_with_host(m, 0, &args.map(Value::I32), &mut fuel, host)
}

fn bytecode_engine(
    m: &temen_ir::Module,
    args: &[i32; 4],
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    let mut fuel = 5_000_000u64;
    run_with_host_fast(m, 0, &args.map(Value::I32), &mut fuel, host)
}

/// Executor 2, the bytecode engine's parallel driver: a thread per task.
fn parallel(m: &temen_ir::Module, args: &[i32; 4], host: &mut Host) -> Result<Vec<Value>, Trap> {
    let mut fuel = 5_000_000u64;
    let args = args.map(Value::I32);
    let run = bytecode::compile_and_run_capture_over_parallel_with_host;
    let (out, _) =
        run(m, 0, &args, &mut fuel, &[], None, host).expect("executor 2 runs the caller");
    out
}

fn jit(m: &temen_ir::Module, args: &[i32; 4], host: &mut Host) -> Result<Vec<Value>, Trap> {
    let image = MemLayout::image(vec![0u8; 128 << 10]);
    let (out, _) = temen_run::jit_cap_run(m, 0, &args.map(i64::from), &image, 0, 0, host, None)
        .expect("the JIT compiles the caller");
    match out {
        JitOutcome::Returned(slots) => Ok(slots.into_iter().map(Value::I64).collect()),
        JitOutcome::Trapped(k) => Err(Trap::from_code(k.code()).expect("a trap")),
        other => panic!("the JIT run ended otherwise: {other:?}"),
    }
}

/// Run `caller` with the serving child `server` on `entry`, direct handoff on or off. The run is on
/// its own thread, so a call no engine answers fails the test instead of stalling the suite.
fn run_on(entry: Entry, caller: &str, server: &str, handoff: bool) -> Result<Vec<Value>, Trap> {
    let (a, b) = (module(caller), module(server));
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut host = Host::new();
        host.set_handoff(handoff);
        let args = [
            host.grant_instantiator(),
            host.grant_module(&b),
            host.grant_budget(-1, 1 << 20, -1),
            host.grant_clock(),
        ];
        let _ = tx.send(entry(&a, &args, &mut host));
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .expect("the run hung")
}

/// Run `caller` with the serving child `server` on the tree-walker, natively on the bytecode engine's
/// two executors, and on the Cranelift JIT, and answer their one result.
fn run(caller: &str, server: &str, handoff: bool) -> Result<Vec<Value>, Trap> {
    let oracle = interpreted(caller, server, handoff);
    let jitted = run_on(jit, caller, server, handoff);
    assert_eq!(jitted, oracle, "the JIT answers as the tree-walker does");
    oracle
}

/// [`run`] on the tree-walker and the bytecode engine's two executors alone.
fn interpreted(caller: &str, server: &str, handoff: bool) -> Result<Vec<Value>, Trap> {
    assert!(
        bytecode::admits(&module(caller)),
        "the bytecode engine runs the caller"
    );
    assert!(bytecode::admits(&module(server)), "and the serving child");
    let oracle = run_on(tree_walker, caller, server, handoff);
    let fast = run_on(bytecode_engine, caller, server, handoff);
    assert_eq!(
        fast, oracle,
        "the bytecode engine answers as the tree-walker does"
    );
    let par = run_on(parallel, caller, server, handoff);
    assert_eq!(par, oracle, "executor 2 answers as the tree-walker does");
    oracle
}

/// One served dispatch, then the call's value.
fn served(call: i64) -> Result<Vec<Value>, Trap> {
    Ok(vec![Value::I64(1_000_000 + call)])
}

#[test]
fn a_parked_root_gets_both_results() {
    let r = run(
        &caller(ADDMUL_CALL, ""),
        &server(ADDMUL_TYPES, ADDMUL),
        false,
    );
    assert_eq!(r, served(42_080));
}

#[test]
fn a_call_served_by_handoff_gets_both_results() {
    let r = run(
        &caller(ADDMUL_CALL, ""),
        &server(ADDMUL_TYPES, ADDMUL),
        true,
    );
    assert_eq!(r, served(42_080));
}

#[test]
fn a_parked_fiber_gets_both_results() {
    let r = run(
        &caller(IN_A_FIBER, ADDMUL_FIBER),
        &server(ADDMUL_TYPES, ADDMUL),
        false,
    );
    assert_eq!(r, served(42_080));
}

/// A handle passed to the child is granted into its table: the child reads its own clock through
/// it, `1000` past what it reads.
#[test]
fn a_handle_argument_is_granted_into_the_callee() {
    let types = "type 0 func (cap) -> (i64)
type 1 interface { now: 0 }
export 0 interface \"clock\" 1 { now: 1 }";
    let handler = r#"func (cap) -> (i64) {
block 0 (vc: cap) {
  vt = call.cap 2 0 () -> (i64) vc ()
  vk = i64.const 1000
  vr = i64.add vt vk
  return vr
  }
}
"#;
    let call = r#"
  vcall = call.cap 268435456 0 (cap) -> (i64) v7 (v3)"#;
    let r = run(&caller(call, ""), &server(types, handler), false);
    assert_eq!(r, served(1000), "the child read a clock of its own");
}

/// A handle the child returns is granted into the caller's table: a fresh handle, not the number the
/// caller passed, and the caller reads its clock through it.
#[test]
fn a_handle_result_is_granted_back_into_the_caller() {
    let types = "type 0 func (cap) -> (cap)
type 1 interface { echo: 0 }
export 0 interface \"echo\" 1 { echo: 1 }";
    let handler = r#"func (cap) -> (cap) {
block 0 (vc: cap) {
  return vc
  }
}
"#;
    // `vcall`: 1 when the returned handle is fresh, plus 10 when a clock read through it succeeds.
    let call = r#"
  vh = call.cap 268435456 0 (cap) -> (cap) v7 (v3)
  vsame = i32.eq vh v3
  vfresh = i32.eqz vsame
  vfresh64 = i64.extend_i32_u vfresh
  vt = call.cap 2 0 () -> (i64) vh ()
  vok = i64.ge_s vt v6
  vok64 = i64.extend_i32_u vok
  vten = i64.const 10
  vokten = i64.mul vok64 vten
  vcall = i64.add vfresh64 vokten"#;
    let r = run(&caller(call, ""), &server(types, handler), false);
    assert_eq!(r, served(11), "a fresh caller-side clock handle");
}

/// The callee dies before it replies: a two-result call answers `CAP_REVOKED` in its first result
/// and zero in its second.
#[test]
fn a_failed_call_answers_its_errno_in_the_first_result() {
    let handler = r#"func (i64, i64) -> (i64, i64) {
block 0 (va: i64, vb: i64) {
  unreachable
  }
}
"#;
    let r = run(
        &caller(ADDMUL_CALL, ""),
        &server(ADDMUL_TYPES, handler),
        false,
    );
    // The child trapped, so its join answers its trap, not a count.
    let Ok(v) = r else {
        panic!("the caller survives a dead callee: {r:?}")
    };
    let [Value::I64(x)] = v[..] else {
        panic!("{v:?}")
    };
    assert_eq!(
        x.rem_euclid(1_000_000),
        (CAP_REVOKED * 1000).rem_euclid(1_000_000),
        "(CAP_REVOKED, 0) in the call's two results: {x}"
    );
}

/// #2742 — the same from a fiber, which the call parked alone: the callee's death answers it
/// `(CAP_REVOKED, 0)` when it next resumes.
#[test]
fn a_parked_fiber_whose_callee_dies_answers_its_errno() {
    let handler = r#"func (i64, i64) -> (i64, i64) {
block 0 (va: i64, vb: i64) {
  unreachable
  }
}
"#;
    let r = run(
        &caller(IN_A_FIBER, ADDMUL_FIBER),
        &server(ADDMUL_TYPES, handler),
        false,
    );
    let Ok(v) = r else {
        panic!("the caller survives a dead callee: {r:?}")
    };
    let [Value::I64(x)] = v[..] else {
        panic!("{v:?}")
    };
    assert_eq!(
        x.rem_euclid(1_000_000),
        (CAP_REVOKED * 1000).rem_euclid(1_000_000),
        "(CAP_REVOKED, 0) in the call's two results: {x}"
    );
}

/// A failed call whose first result is not an integer cannot carry its errno, so it traps.
#[test]
fn a_failed_call_without_an_integer_first_result_traps() {
    let types = "type 0 func (i64) -> (f64, i64)
type 1 interface { f: 0 }
export 0 interface \"f\" 1 { f: 1 }";
    let handler = r#"func (i64) -> (f64, i64) {
block 0 (va: i64) {
  unreachable
  }
}
"#;
    let call = r#"
  va = i64.const 1
  vf, vi = call.cap 268435456 0 (i64) -> (f64, i64) v7 (va)
  vcall = i64.const 0"#;
    let r = run(&caller(call, ""), &server(types, handler), false);
    assert_eq!(r, Err(Trap::CapFault));
}

/// A caller that declares a result a handle where the op answers an integer gets the integer: the
/// op's own signature decides which slots cross as handles, never the call site's. Were the call
/// site to decide, the child's `5` would be translated from the child's table, into a fresh handle
/// to whatever the child holds there or a fault, and a caller could take a callee's handle by naming
/// its number.
#[test]
fn a_caller_cannot_declare_an_integer_result_a_handle() {
    let types = "type 0 func (i64) -> (i64)
type 1 interface { five: 0 }
export 0 interface \"five\" 1 { five: 1 }";
    let handler = r#"func (i64) -> (i64) {
block 0 (va: i64) {
  v5 = i64.const 5
  return v5
  }
}
"#;
    let call = r#"
  va = i64.const 0
  vh = call.cap 268435456 0 (i64) -> (cap) v7 (va)
  vfive = i32.const 5
  vsame = i32.eq vh vfive
  vcall = i64.extend_i32_u vsame"#;
    let r = run(&caller(call, ""), &server(types, handler), false);
    assert_eq!(r, served(1), "the number, untranslated");
}

/// A call whose arity is not the op's faults, as at the instanced boundary: here it declares one
/// result of an op that answers two.
#[test]
fn a_call_of_another_arity_faults() {
    let call = r#"
  va = i64.const 40
  vb = i64.const 2
  vcall = call.cap 268435456 0 (i64, i64) -> (i64) v7 (va, vb)"#;
    let r = run(&caller(call, ""), &server(ADDMUL_TYPES, ADDMUL), false);
    assert_eq!(r, Err(Trap::CapFault));
}

/// A domain can hold a live offer into itself: here its parent grants the child the offer over the
/// child (`Instantiator.grant`, op 19) and hands it the index (`keep`), and the child calls through
/// it once its serve loop has returned. The call queues on the child's own queue, which nothing
/// serves: every engine reaches the deadlock verdict, and none hangs on a powerbox lock it holds.
#[test]
fn a_domain_calling_a_live_offer_into_itself_deadlocks_on_every_engine() {
    let server = r#"memory 12
type 0 func (i64) -> (i64)
type 1 interface { keep: 0 }
export 0 interface "keep" 1 { keep: 1 }
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  va = i64.const 64
  vh64 = i64.load va
  vh = i32.wrap_i64 vh64
  vr = call.cap 268435456 0 (i64) -> (i64) vh (vh64)
  return vr
  }
}

func (i64) -> (i64) {
block 0 (vh: i64) {
  va = i64.const 64
  i64.store va vh
  vzero = i64.const 0
  return vzero
  }
}
"#;
    let call = r#"
  v764 = i64.extend_i32_u v7
  vin = call.cap 6 19 (i32, i64) -> (i32) v0 (v5, v764)
  vin64 = i64.extend_i32_s vin
  vcall = call.cap 268435456 0 (i64) -> (i64) v7 (vin64)"#;
    let r = run(&caller(call, ""), server, false);
    assert_eq!(r, Err(Trap::ThreadFault), "the deadlock verdict");
}

/// #2742 — a handler that calls its own domain waits while its serve loop serves that call. The
/// parent grants the child the offer over the child (op 19) and calls `again(h, 1)` with the child's
/// index of it. That handler calls `again(h, 0)` through `h`, which queues on the child's own queue,
/// and answers its reply plus one. The child's one `svc.wait` serves both: the handler parked on
/// its own domain's call parks alone, as a fiber of the serve loop, and takes the reply at once.
/// The Cranelift JIT runs a handler on its serve loop's stack, so there its park parks the whole
/// loop and the call deadlocks: this case leaves it out.
#[test]
fn a_handler_calling_its_own_domain_is_served_meanwhile() {
    let r = interpreted(&caller(AGAIN_CALL, ""), &server(AGAIN_TYPES, AGAIN), false);
    // Two dispatches served, and the call answers 7 + 1.
    assert_eq!(r, Ok(vec![Value::I64(2_000_008)]));
}

const AGAIN_TYPES: &str = "type 0 func (i64, i64) -> (i64)
type 1 interface { again: 0 }
export 0 interface \"again\" 1 { again: 1 }";

/// `again(h, n)`: 7 at `n = 0`, else one more than its call of `again(h, n - 1)` through `h`.
const AGAIN: &str = r#"func (i64, i64) -> (i64) {
block 0 (vh: i64, vn: i64) {
  vzero = i64.const 0
  vlast = i64.eq vn vzero
  br_if vlast 1() 2(vh, vn)
}
block 1 () {
  vseven = i64.const 7
  return vseven
}
block 2 (wh: i64, wn: i64) {
  vone = i64.const 1
  wn1 = i64.sub wn vone
  vself = i32.wrap_i64 wh
  vr = call.cap 268435456 0 (i64, i64) -> (i64) vself (wh, wn1)
  vr1 = i64.add vr vone
  return vr1
  }
}
"#;

/// The parent's call of `again(h, 1)`, `h` the child's index of the offer over itself.
const AGAIN_CALL: &str = r#"
  v764 = i64.extend_i32_u v7
  vin = call.cap 6 19 (i32, i64) -> (i32) v0 (v5, v764)
  vin64 = i64.extend_i32_s vin
  vone = i64.const 1
  vcall = call.cap 268435456 0 (i64, i64) -> (i64) v7 (vin64, vone)"#;

/// The caller's entry resumes fiber func 1 once with a plain `cont.resume`, then with
/// `cont.resume.block` until it returns. `vcall` is the first resume's status · 100_000 plus the
/// fiber's value.
const POLLED_FIBER: &str = r#"
  vf = ref.func 1
  vfsp = i64.const 24576
  vfk = cont.new vf vfsp
  voff = i64.extend_i32_u v7
  vs1, vv1 = cont.resume vfk voff
  vone1 = i32.const 1
  vdone1 = i32.eq vs1 vone1
  br_if vdone1 2(v0, v5, vs1, vv1) 1(v0, v5, vfk, voff, vs1)
}
block 1 (x0: i32, x5: i32, xk: i64, xoff: i64, xs1: i32) {
  xs, xv = cont.resume.block xk xoff
  xone = i32.const 1
  xdone = i32.eq xs xone
  br_if xdone 2(x0, x5, xs1, xv) 1(x0, x5, xk, xoff, xs1)
}
block 2 (v0: i32, v5: i32, ws1: i32, wv: i64) {
  ws164 = i64.extend_i32_u ws1
  vhk = i64.const 100000
  vsh1 = i64.mul ws164 vhk
  vcall = i64.add vsh1 wv"#;

/// Fiber func 1 of [`a_fiber_s_thread_local_word_survives_its_park`]: [`ADDMUL_FIBER`]'s call, after
/// it sets the thread-local word to 42.
const TLS_FIBER: &str = r#"
func (i64, i64) -> (i64) {
block 0 (vsp: i64, voff: i64) {
  vtl = i64.const 42
  vcpu.tls.set vtl
  v7 = i32.wrap_i64 voff
  va = i64.const 40
  vb = i64.const 2
  vsum, vprod = call.cap 268435456 0 (i64, i64) -> (i64, i64) v7 (va, vb)
  vthousand = i64.const 1000
  vsh = i64.mul vsum vthousand
  vr = i64.add vsh vprod
  return vr
  }
}
"#;

/// The caller's entry resumes fiber func 1 once with a plain `cont.resume`, reads the thread-local
/// word, then resumes it with `cont.resume.block` until it returns. `vcall` is the word · 100_000
/// plus the fiber's value.
const TLS_READER: &str = r#"
  vf = ref.func 1
  vfsp = i64.const 24576
  vfk = cont.new vf vfsp
  voff = i64.extend_i32_u v7
  vs1, vv1 = cont.resume vfk voff
  vtl = vcpu.tls.get
  br 1(v0, v5, vfk, voff, vtl)
}
block 1 (x0: i32, x5: i32, xk: i64, xoff: i64, xtl: i64) {
  xs, xv = cont.resume.block xk xoff
  xone = i32.const 1
  xdone = i32.eq xs xone
  br_if xdone 2(x0, x5, xtl, xv) 1(x0, x5, xk, xoff, xtl)
}
block 2 (v0: i32, v5: i32, wtl: i64, wv: i64) {
  vhk = i64.const 100000
  vsh1 = i64.mul wtl vhk
  vcall = i64.add vsh1 wv"#;

/// A fiber parked on its call hands the thread-local word back with execution: the word is the
/// vCPU's, so its resumer reads what the fiber set (#2742, `park_running_fiber`).
#[test]
fn a_fiber_s_thread_local_word_survives_its_park() {
    let r = run(
        &caller(TLS_READER, TLS_FIBER),
        &server(ADDMUL_TYPES, ADDMUL),
        false,
    );
    assert_eq!(r, served(42 * 100_000 + 42_080));
}

/// #2742 — a fiber's call parks the fiber, not its task: the plain resume that ran it into the call
/// answers `FIBER_PARKED` (3), and the blocking resumes after it take the call's value once the
/// reply comes.
#[test]
fn a_polled_fiber_making_a_call_parks_alone() {
    let r = run(
        &caller(POLLED_FIBER, ADDMUL_FIBER),
        &server(ADDMUL_TYPES, ADDMUL),
        false,
    );
    assert_eq!(r, served(3 * 100_000 + 42_080));
}

/// #2742 — a handler parked on a call into another domain: its serve loop waits in `svc.wait` with
/// nothing queued, and the reply, when it comes, wakes the loop to resume the handler. The parent
/// spawns two children from [`AGAIN`]'s module, grants the first an offer over the second (op 19),
/// and calls `again(h, 1)` on the first, whose handler calls `again(h, 0)` on the second. Each child
/// serves one dispatch. The parent answers `joinA · 1_000_000 + joinB · 1_000 + call`.
#[test]
fn a_handler_waiting_on_another_domain_wakes_its_serve_loop() {
    let caller = format!(
        r#"memory 17

func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {{
  vrm = i64.const 17416
  i32.store vrm v1
  vrb = i64.const 17420
  i32.store vrb v2
  vrp = i64.const 17408
  va = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vb = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vz = i64.const 0
  vtob = call.cap 6 14 (i32, i64) -> (i32) v0 (vb, vz)
  vtob64 = i64.extend_i32_u vtob
  vin = call.cap 6 19 (i32, i64) -> (i32) v0 (va, vtob64)
  vin64 = i64.extend_i32_s vin
  vtoa = call.cap 6 14 (i32, i64) -> (i32) v0 (va, vz)
  vone = i64.const 1
  vcall = call.cap 268435456 0 (i64, i64) -> (i64) vtoa (vin64, vone)
  vja_st, vja = call.cap 6 1 (i32) -> (i64, i64) v0 (va)
  vjb_st, vjb = call.cap 6 1 (i32) -> (i64, i64) v0 (vb)
  vmil = i64.const 1000000
  vk = i64.const 1000
  vma = i64.mul vja vmil
  vmb = i64.mul vjb vk
  vs1 = i64.add vma vmb
  vs = i64.add vs1 vcall
  return vs
  }}
}}
{rec}"#,
        rec = rec::segment(17408, &temen_ir::SpawnRec::default())
    );
    let r = run(&caller, &server(AGAIN_TYPES, AGAIN), false);
    assert_eq!(r, Ok(vec![Value::I64(1_001_008)]));
}
