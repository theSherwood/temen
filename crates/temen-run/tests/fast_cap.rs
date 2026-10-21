//! Regression for the §9/D45 **allocation-free clock fast path** (ISSUES.md I12). `run_powerbox`
//! wires `fast_cap_resolver`, which devirtualizes `Clock.now()` to `Host::fast_clock_now` — an
//! authority-checked, `Vec`-free inline read. This pins that the fast path produces the **same**
//! results as the generic `cap_thunk` path and the tree-walker (so interp == JIT still holds), and
//! that a forged handle still faults.

use temen_interp::{cap_id, run_with_host, Host, Value};
use temen_jit::{
    compile_and_run_with_host, compile_and_run_with_host_fast, InstanceAddrs, JitOutcome, TrapKind,
    VmCtx,
};
use temen_run::{cap_thunk, fast_cap_resolver};

// `(i32 clk) -> i64`: call `Clock.now()` (0-arg, so the fast resolver `(CLOCK,0,0,1)` engages) twice
// and return the delta. The mock clock starts at 0 and increments once per read, so this is 1.
const SRC: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = call.cap 2 0 () -> (i64) v0 ()
  v2 = call.cap 2 0 () -> (i64) v0 ()
  v3 = i64.sub v2 v1
  return v3
  }
}
"#;

fn module() -> temen_ir::Module {
    let m = temen_text::parse_module(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

#[test]
fn clock_fast_path_matches_generic_and_interp() {
    let m = module();

    // Tree-walker (full Host dispatch).
    let mut host = Host::new();
    let clk = host.grant_clock();
    let mut fuel = u64::MAX;
    let tw = run_with_host(&m, 0, &[Value::I32(clk)], &mut fuel, &mut host).expect("interp");
    assert_eq!(tw, vec![Value::I64(1)], "tree-walk clock delta");

    // JIT, generic cap_thunk.
    let mut host = Host::new();
    let clk = host.grant_clock();
    let ctx = &mut host as *mut Host as *mut core::ffi::c_void;
    let gen = compile_and_run_with_host(&m, 0, &[clk as i64], cap_thunk, ctx).expect("jit");
    assert_eq!(
        gen,
        JitOutcome::Returned(vec![1]),
        "jit generic clock delta"
    );

    // JIT, D45 fast resolver — the inline allocation-free path.
    let mut host = Host::new();
    let clk = host.grant_clock();
    let ctx = &mut host as *mut Host as *mut core::ffi::c_void;
    let fast = compile_and_run_with_host_fast(
        &m,
        0,
        &[clk as i64],
        cap_thunk,
        ctx,
        fast_cap_resolver,
        None,
    )
    .expect("jit fast");
    assert_eq!(fast, JitOutcome::Returned(vec![1]), "jit fast clock delta");
}

#[test]
fn clock_fast_path_faults_on_forged_handle() {
    // A handle that was never granted must fault on the fast path exactly like the generic one
    // (authority is preserved: `resolve` rejects a wrong generation/type/closed slot).
    let m = module();
    let mut host = Host::new();
    let _real = host.grant_clock();
    let forged = 0x7fff_fffe; // not a valid (slot, generation) for a CLOCK binding
    let ctx = &mut host as *mut Host as *mut core::ffi::c_void;
    let r =
        compile_and_run_with_host_fast(&m, 0, &[forged], cap_thunk, ctx, fast_cap_resolver, None);
    // A CapFault surfaces as a trap (Err) or a Trapped outcome — either way, not a clean return.
    let faulted = !matches!(r, Ok(JitOutcome::Returned(_)));
    assert!(
        faulted,
        "forged clock handle must not return cleanly: {r:?}"
    );
}

#[test]
fn resolver_never_claims_parkable_ifaces() {
    // DESIGN.md §12 parking pin: a parkable registration is never fast-path-claimed. The D45
    // resolver bakes a register-ABI address at compile time; a handler that can answer `Pending`
    // needs the generic thunk (completion mint + park), so `HOST_PROC` and the test-only
    // `BLOCKING` iface must resolve to null for every plausible arity.
    use temen_interp::cap_id;
    for iface in [cap_id::HOST_PROC, cap_id::BLOCKING] {
        for op in 0..4u32 {
            for n_args in 0..5u32 {
                for n_res in 0..2u32 {
                    let p = unsafe { fast_cap_resolver(iface, op, n_args, n_res) };
                    assert!(
                        p.is_null(),
                        "fast_cap_resolver claimed parkable iface {iface} op {op} \
                         ({n_args} args, {n_res} res)"
                    );
                }
            }
        }
    }
}

/// #2499 — a call that succeeds leaves its domain's end alone: a sibling vCPU may have raised it
/// during the call, and the thunks used to answer success by writing `0` over it, which let the
/// domain run on past a trap or an `exit`. Both thunks, called as compiled code calls them, with an
/// end already raised.
#[test]
fn a_successful_call_leaves_the_domains_end() {
    let mut host = Host::new();
    let clk = host.grant_clock();
    let ctx = &mut host as *mut Host as *mut core::ffi::c_void;
    let end = TrapKind::Unreachable.code();
    let vm = VmCtx::new(InstanceAddrs::NONE);
    vm.raise(end);

    let mut now = 0i64;
    // SAFETY: `ctx` is a live `*mut Host`; no args, one result slot, no window.
    unsafe {
        cap_thunk(
            ctx,
            core::ptr::null_mut(),
            0,
            0,
            cap_id::CLOCK,
            0,
            clk,
            core::ptr::null(),
            0,
            &mut now,
            1,
            &vm,
        );
    }
    assert_eq!(vm.ended(), Some(end), "the generic thunk kept the end");

    // SAFETY: a pure lookup; a claimed op answers with the fast ABI for its arity.
    let fast = unsafe { fast_cap_resolver(cap_id::CLOCK, 0, 0, 1) };
    assert!(!fast.is_null(), "the clock read is fast-path claimed");
    // SAFETY: the `FastCapResolver` ABI of a 0-arg, 1-result op.
    let fast: unsafe extern "C" fn(*mut core::ffi::c_void, *mut u8, u64, i32, &VmCtx) -> i64 =
        unsafe { core::mem::transmute(fast) };
    // SAFETY: as for the generic call.
    unsafe { fast(ctx, core::ptr::null_mut(), 0, clk, &vm) };
    assert_eq!(vm.ended(), Some(end), "the fast path kept the end");
}
