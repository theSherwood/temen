//! #1894 — a **POSIX guest freezes**. The personality is a captured capability: a freeze carries
//! its guest-visible state (the memfs, the root process's fd table and file offsets, its allocator,
//! cwd and environment) in the artifact, and a thaw's registrar rebuilds the personality from those
//! bytes ([`temen_posix::Posix::from_state`]). The guest then runs on as if it had never stopped.
//!
//! The guest `malloc`s, opens `/f` for writing, `chdir`s, then writes one byte per loop iteration;
//! the freeze lands mid-loop. After the loop it `malloc`s again, closes, re-opens `/f` and reads it
//! back. Uninterrupted and thawed alike it answers `20·1000 + 1` (twenty bytes read; the second block
//! is not the first, so the allocator's state rode), and the thawed personality's cwd is `/d`.

use std::sync::{Arc, Mutex};
use temen_durable::{
    arm_freeze_after_backedges, begin_thaw, init_durable_window, read_state,
    transform_module_assume_confined, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::Memory;

const ARENA: ShadowArena = ShadowArena::new(16448, 65536);
const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;
const HEAP: (u64, u64) = (100_000, 120_000);

/// `op(args)` on the personality: `call.cap HOST_PROC op`, every argument and result an `i64`.
const SRC: &str = r#"
memory 17
data 66000 "/f"
data 66010 "/d"
data 66020 "x"
func (i32) -> (i64) {
block 0 (vl: i32) {
  v16 = i64.const 16
  vp1 = call.cap 13 2 (i64) -> (i64) vl (v16)
  vpath = i64.const 66000
  vplen = i64.const 2
  vflags = i64.const 65
  vfd = call.cap 13 5 (i64, i64, i64) -> (i64) vl (vpath, vplen, vflags)
  vdir = i64.const 66010
  vc = call.cap 13 10 (i64, i64) -> (i64) vl (vdir, vplen)
  vz = i64.const 0
  br 1(vl, vfd, vp1, vz)
}
block 1 (vl1: i32, vfd1: i64, vp11: i64, vi: i64) {
  vx = i64.const 66020
  vone = i64.const 1
  vw = call.cap 13 0 (i64, i64, i64) -> (i64) vl1 (vfd1, vx, vone)
  vi2 = i64.add vi vone
  vlim = i64.const 20
  vmore = i64.ne vi2 vlim
  br_if vmore 1(vl1, vfd1, vp11, vi2) 2(vl1, vfd1, vp11)
}
block 2 (vl2: i32, vfd2: i64, vp12: i64) {
  v16b = i64.const 16
  vp2 = call.cap 13 2 (i64) -> (i64) vl2 (v16b)
  vcl = call.cap 13 6 (i64) -> (i64) vl2 (vfd2)
  vpath2 = i64.const 66000
  vplen2 = i64.const 2
  vro = i64.const 0
  vfd3 = call.cap 13 5 (i64, i64, i64) -> (i64) vl2 (vpath2, vplen2, vro)
  vbuf = i64.const 67000
  vcap = i64.const 100
  vn = call.cap 13 1 (i64, i64, i64) -> (i64) vl2 (vfd3, vbuf, vcap)
  vk = i64.const 1000
  vnk = i64.mul vn vk
  vgap = i64.sub vp2 vp12
  vsix = i64.const 16
  vnext = i64.eq vgap vsix
  vd64 = i64.extend_i32_u vnext
  vr = i64.add vnk vd64
  return vr
  }
}
"#;

fn instrumented() -> Arc<temen_ir::Module> {
    let mut m = temen_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(ARENA),
    });
    let inst = Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

/// A durable host granting the personality under the name `"libc"`.
fn granted() -> (Host, temen_posix::Posix, i32) {
    let mut h = Host::new();
    h.set_durable(true);
    let (lh, posix) = temen_posix::grant(&mut h, HEAP.0, HEAP.1, Vec::new());
    h.register_cap_name("libc", lh);
    (h, posix, lh)
}

fn run(inst: &temen_ir::Module, h: &mut Host, lh: i32, win: &[u8]) -> (i64, Vec<u8>) {
    let mut fuel = 10_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(inst, 0, &[Value::I32(lh)], &mut fuel, win, SIZE_LOG2, h);
    match r.expect("the run returns")[..] {
        [Value::I64(n)] => (n, snap),
        ref other => panic!("unexpected result {other:?}"),
    }
}

#[test]
fn a_posix_guest_freezes_mid_loop_and_thaws_through_the_codec() {
    let inst = instrumented();

    let (mut h, posix, lh) = granted();
    let (r, _) = run(&inst, &mut h, lh, &init_durable_window(WINDOW, ARENA));
    assert_eq!(
        r, 20_001,
        "uninterrupted: twenty bytes read, the next block"
    );
    assert_eq!(posix.read_file("/f"), Some(vec![b'x'; 20]));

    // Frozen mid-loop, with `/f` open and part-written.
    let (mut h, _posix, lh) = granted();
    let mut win = init_durable_window(WINDOW, ARENA);
    arm_freeze_after_backedges(&mut win, 16);
    let (r, snap) = run(&inst, &mut h, lh, &win);
    assert_eq!(r, 0, "the root unwinds for the freeze");
    assert_eq!(read_state(&snap), STATE_UNWINDING, "the cut was taken");
    let art = temen_snapshot::freeze(&inst, &snap, &h).expect("the personality rides");

    // Thaw into a fresh host whose registrar rebuilds the personality from the carried state.
    let thawed: Arc<Mutex<Option<temen_posix::Posix>>> = Arc::default();
    let mut th = Host::new();
    th.set_durable(true);
    let slot = Arc::clone(&thawed);
    th.set_named_cap_registrar(Box::new(move |name, state| {
        let posix = (name == "libc")
            .then(|| temen_posix::Posix::from_state(state).ok())
            .flatten()?;
        let g = temen_posix::named_grant(&posix);
        *slot.lock().unwrap() = Some(posix);
        Some(g)
    }));
    let mut twin = temen_snapshot::restore(&art, &inst, &mut th).expect("restore");
    let posix = thawed
        .lock()
        .unwrap()
        .take()
        .expect("the registrar re-granted libc");
    temen_posix::install(&mut th, &posix, lh);
    assert_eq!(posix.cwd(), "/d", "the cwd rode");
    let partial = posix.read_file("/f").expect("the file rode");
    assert!(
        !partial.is_empty() && partial.len() < 20,
        "frozen mid-loop: {} bytes written",
        partial.len()
    );

    begin_thaw(&mut twin, ARENA, 0);
    let (r, _) = run(&inst, &mut th, lh, &twin);
    assert_eq!(r, 20_001, "the thaw writes on at the carried offset");
    assert_eq!(posix.read_file("/f"), Some(vec![b'x'; 20]));

    // §12.6: the thawed personality's state re-captures to the bytes it was restored from.
    let again = temen_posix::Posix::from_state(&posix.capture_state()).expect("re-restore");
    assert_eq!(again.capture_state(), posix.capture_state());
}
