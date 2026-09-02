//! **Accesses past a caller-provided backing never reach host memory** (#1191, found by the #1151
//! `nested_paged` differential). Every Region-backed vCPU path (`Vcpu::new_root_with_powerbox`, the
//! browser's `Region::shared` reactor / tier-up / threads windows) asks for the 1-TiB
//! `DEFAULT_RESERVED_LOG2` reservation over a backing of the window's size — deliberately (#1153: a
//! guest's high `vm_map` is admitted so the emitted tier's bounds check turns the first access into a
//! decline). The confinement bound is the reservation plus the page map; the flat backings' word and
//! atomic fast paths were bounded by nothing else, so a `map`/`protect` past the backing's end followed
//! by a load/store read or wrote **host memory** behind the window. The `Region` accessors now bound
//! every access to the backing: such a load reads zero and such a store is dropped — the documented
//! "reserved-tail accesses beyond the backing read as zero" contract, enforced at the seam. The canary
//! byte placed just past the backing pins the store side directly; the page op itself still succeeds
//! (the oracle's engine-`mmap`ed window would too — the divergence is the tier's to decline).

use std::sync::Arc;
use temen_interp::{bytecode, Host, Region, Trap, Value};

/// The declared window and the backing: both 128 KiB (`memory 17`) — no room to grow.
const WIN_LOG2: u8 = 17;
const WIN: u64 = 1 << WIN_LOG2;

/// The guest for page op `op` (0 = `map` RW, 2 = `protect` RO). `f0(as, off, len)` runs the op and
/// returns its result (`0` ok, negative errno). `f1(as, off, len)` runs the op, then `i64.store`s a
/// marker at `off + len - 8` (the range's last word) and `i64.load`s it back — the access an admitted page past the backing would
/// turn into a host-memory read/write.
fn src(op: u32) -> String {
    let call = if op == 0 {
        "  vprot = i32.const 3\n  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)"
    } else {
        "  vro = i32.const 1\n  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vro)"
    };
    format!(
        r#"memory 17
func (i32, i64, i64) -> (i64) {{
block 0 (vas: i32, voff: i64, vlen: i64) {{
{call}
  return vr
  }}
}}
func (i32, i64, i64) -> (i64) {{
block 0 (vas: i32, voff: i64, vlen: i64) {{
{call}
  veight = i64.const 8
  vend = i64.add voff vlen
  vaddr = i64.sub vend veight
  vmark = i64.const 424242
  i64.store vaddr vmark
  vld = i64.load vaddr
  return vld
  }}
}}
"#
    )
}

/// A zeroed buffer twice the window, the `Region::shared` covering only the first half, and a canary
/// right after the backing's end — the byte a host-memory escape would clobber.
fn run(op: u32, func: u32, off: u64, len: u64) -> (Result<Vec<Value>, Trap>, u8) {
    let m = temen_text::parse_module(&src(op)).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let layout = std::alloc::Layout::from_size_align(2 * WIN as usize, 8).unwrap();
    // SAFETY: non-zero layout; the buffer is owned here and freed after the vCPU is dropped.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    const CANARY: u8 = 0x5A;
    // SAFETY: `WIN + 8` is inside the 2*WIN buffer.
    unsafe { base.add(WIN as usize + 8).write(CANARY) };
    // SAFETY: `[base, base+WIN)` is valid, 8-aligned, exclusively this window's.
    let back = Arc::new(unsafe { Region::shared(base, WIN) });
    let mut host = Host::new();
    let asl = host.grant_memory();
    let args = [
        Value::I32(asl),
        Value::I64(off as i64),
        Value::I64(len as i64),
    ];
    let mut vcpu = bytecode::Vcpu::new_root_with_powerbox(&prog, func, &args, back, &[], host)
        .expect("root vcpu");
    let res = match vcpu.run() {
        bytecode::VcpuEvent::Done(v) => Ok(v),
        bytecode::VcpuEvent::Trapped(t) => Err(t),
        _ => panic!("unexpected event"),
    };
    let reserved = vcpu.mem_map_info().expect("window").2;
    assert!(
        reserved > WIN,
        "the reservation stays wider than the backing (#1153)"
    );
    drop(vcpu);
    // SAFETY: the vCPU (and its `Mem` aliasing the region) is dropped.
    let canary = unsafe { base.add(WIN as usize + 8).read() };
    unsafe { std::alloc::dealloc(base, layout) };
    (res, canary)
}

fn errno_of(r: &Result<Vec<Value>, Trap>) -> i64 {
    match r {
        Ok(v) => match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("i64 result expected, got {other:?}"),
        },
        Err(t) => panic!("expected a value, got trap {t:?}"),
    }
}

#[test]
fn map_past_the_backing_never_writes_host_memory() {
    let page = temen_interp::host_page_size();
    // A grow landing exactly past the backing is admitted by the (wider) reservation — the page op
    // itself succeeds, as it would over the engine's own reservation.
    let (r, canary) = run(0, 0, WIN, page);
    assert_eq!(errno_of(&r), 0, "map inside the reservation succeeds");
    assert_eq!(canary, 0x5A);
    // The same grow followed by the store/load: the store is dropped at the backing's end and the
    // load reads zero — the canary right after the backing is untouched, no host write.
    let (r, canary) = run(0, 1, WIN, page);
    assert_eq!(
        errno_of(&r),
        0,
        "a load past the backing reads zero (the store was dropped)"
    );
    assert_eq!(canary, 0x5A, "no host write past the backing");
}

#[test]
fn protect_straddling_the_backing_end_never_reads_host_memory() {
    let page = temen_interp::host_page_size();
    // The nested_paged seed-900 shape: a protect that starts inside the window and runs past it.
    let (r, canary) = run(2, 0, WIN - 8 * page, 30 * page);
    assert_eq!(errno_of(&r), 0, "protect inside the reservation succeeds");
    assert_eq!(canary, 0x5A);
    // Followed by a store just past the backing: the page is read-only, so the interpreter faults
    // before the access — and the canary behind the buffer is untouched either way.
    let (r, canary) = run(2, 1, WIN - 8 * page, 30 * page);
    assert!(
        matches!(r, Err(Trap::MemoryFault)),
        "a store to a protected page faults, got {r:?}"
    );
    assert_eq!(canary, 0x5A);
}

#[test]
fn page_ops_inside_the_backing_still_work() {
    let page = temen_interp::host_page_size();
    // A re-commit inside the window is fine and the marker round-trips.
    let (r, canary) = run(0, 1, WIN - 4 * page, page);
    assert_eq!(
        errno_of(&r),
        424242,
        "in-backing map succeeds; marker round-trips"
    );
    assert_eq!(canary, 0x5A);
}
