//! **The reservation never exceeds a caller-provided backing** (`Mem::with_reservation_over`, found
//! by the #1151 `nested_paged` differential). Every Region-backed vCPU path (`Vcpu::new_root_with_powerbox`,
//! the browser's `Region::shared` reactor / tier-up / threads windows) asked for the 1-TiB
//! `DEFAULT_RESERVED_LOG2` reservation over a backing of the window's size. The confinement bound is
//! the reservation plus the page map, and the flat backings' word fast path is bounded by nothing
//! else — so a guest `map`/`protect` past the backing's end (admitted anywhere in `[0, reserved)`)
//! turned those pages readable/writable, and a following load/store reached **host memory** behind
//! the window. The reservation is now clamped to the backing: the page op fails with a negative errno
//! (invariant 5) and a scalar access past the backing faults `MemoryFault`. The canary byte placed
//! just past the backing pins the store side directly.

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
    assert_eq!(reserved, WIN, "the reservation is clamped to the backing");
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
fn map_past_the_backing_fails_with_errno_and_never_writes_host_memory() {
    let page = temen_interp::host_page_size();
    // A grow landing exactly past the backing: rejected as a value the guest can observe.
    let (r, canary) = run(0, 0, WIN, page);
    assert!(errno_of(&r) < 0, "map past the backing must fail probeably");
    assert_eq!(canary, 0x5A);
    // The same grow followed by the store/load: the access past the backing faults, and the canary
    // right after the backing is untouched — no host write.
    let (r, canary) = run(0, 1, WIN, page);
    assert!(
        matches!(r, Err(Trap::MemoryFault)),
        "expected MemoryFault, got {r:?}"
    );
    assert_eq!(canary, 0x5A, "no host write past the backing");
}

#[test]
fn protect_straddling_the_backing_end_fails_with_errno() {
    let page = temen_interp::host_page_size();
    // The nested_paged seed-900 shape: a protect that starts inside the window and runs past it.
    let (r, canary) = run(2, 0, WIN - 8 * page, 30 * page);
    assert!(
        errno_of(&r) < 0,
        "protect past the backing must fail probeably"
    );
    assert_eq!(canary, 0x5A);
    // Followed by a load just past the backing: faults (the pages were never admitted).
    let (r, canary) = run(2, 1, WIN - 8 * page, 30 * page);
    assert!(
        matches!(r, Err(Trap::MemoryFault)),
        "expected MemoryFault, got {r:?}"
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
