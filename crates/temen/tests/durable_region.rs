//! #2025 — **a §13 region rides a freeze with its sharing group** (R4, DURABILITY.md §4).
//!
//! A region a domain holds or maps is captured once, its bytes keyed by an artifact number; the
//! window's aliased pages name it instead of carrying bytes; and the restore rebuilds it and the run
//! re-aliases it before the guest resumes. A region that something outside the cut also holds
//! declines, as does a freeze given no page map.
//!
//! Both guests map a 64 KiB region at 128 KiB, store `1111` into it, read the clock (where the freeze
//! lands), store `2222`, and return `first + 2·second` read back through the window. The thawed run
//! reads `first` from the region's carried bytes: the restored window holds no bytes for an aliased
//! page, so it reads `0` unless the region came back aliased.

use std::sync::Arc;

use temen_durable::{
    begin_thaw, init_durable_window, read_state, transform_module_assume_confined, STATE_UNWINDING,
};
use temen_interp::{
    run_capture_reserved_with_host_prots, CapturedProt, Host, RegionNotCaptured, RegionRefusal,
    SharedBacking, Value,
};
use temen_ir::Module;
use temen_snapshot::{freeze, freeze_with_prots, restore_with_prots, FreezeError};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;
const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);
/// Where both guests map the region, and its length.
const AT: usize = 131072;
const LEN: usize = 65536;
const WANT: i64 = 1111 + 2 * 2222;

/// func 0 `(AddressSpace, Clock)`: creates its region. func 1 `(region, Clock)`: is handed one. With
/// `FLIP` they store `UNWINDING` into the state word just before the clock: the freeze lands there.
const SRC: &str = "memory 18 shadow 16448 65536
func (i32, i32) -> (i64) {
block 0 (vas: i32, vclk: i32) {
  vlen = i64.const 65536
  vrh = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vr = i32.wrap_i64 vrh
  br 1(vr, vclk)
}
block 1 (vr: i32, vc: i32) {
  voff = i64.const 131072
  vroff = i64.const 0
  vlen = i64.const 65536
  vprot = i64.const 3
  vm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vr (voff, vroff, vlen, vprot)
  va = i64.const 131080
  vone = i64.const 1111
  i64.store va vone
FLIP
  vz = i32.const 0
  vt = call.cap 2 0 (i32) -> (i64) vc (vz)
  vb = i64.const 131088
  vtwo = i64.const 2222
  i64.store vb vtwo
  vx = i64.load va
  vy = i64.load vb
  vy2 = i64.add vy vy
  vs = i64.add vx vy2
  return vs
  }
}
func (i32, i32) -> (i64) {
block 0 (vr: i32, vc: i32) {
  voff = i64.const 131072
  vroff = i64.const 0
  vlen = i64.const 65536
  vprot = i64.const 3
  vm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vr (voff, vroff, vlen, vprot)
  va = i64.const 131080
  vone = i64.const 1111
  i64.store va vone
FLIP
  vz = i32.const 0
  vt = call.cap 2 0 (i32) -> (i64) vc (vz)
  vb = i64.const 131088
  vtwo = i64.const 2222
  i64.store vb vtwo
  vx = i64.load va
  vy = i64.load vb
  vy2 = i64.add vy vy
  vs = i64.add vx vy2
  return vs
  }
}
";

/// The guest's state word (`STATE_OFF`), guest-addressable, and `UNWINDING`.
const FLIP: &str = "  vsa = i64.const 16384\n  vsu = i32.const 1\n  i32.store vsa vsu";

/// The guests, instrumented; `flip` makes them request the freeze themselves.
fn instrumented(flip: bool) -> Module {
    let m = temen_text::parse_module(&SRC.replace("FLIP", if flip { FLIP } else { "" }))
        .expect("parse");
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

fn run(
    m: &Module,
    func: u32,
    host: &mut Host,
    args: &[Value],
    win: &[u8],
    prots: Option<&[CapturedProt]>,
) -> (
    Result<Vec<Value>, temen_interp::Trap>,
    Vec<u8>,
    Vec<CapturedProt>,
) {
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host_prots(m, func, args, &mut fuel, win, prots, SIZE_LOG2, host)
}

/// The creating guest's host: a whole-window `AddressSpace` and a clock.
fn creating_host() -> (Host, [Value; 2]) {
    let mut host = Host::new();
    host.set_durable(true);
    let asp = host.grant_address_space(0, WINDOW as u64);
    let clk = host.grant_clock();
    (host, [Value::I32(asp), Value::I32(clk)])
}

/// The page map a freeze of the creating guest captured, frozen after its clock read.
fn frozen_creating() -> (Module, Host, [Value; 2], Vec<u8>, Vec<CapturedProt>) {
    let m = instrumented(true);
    let (mut host, args) = creating_host();
    let win = init_durable_window(WINDOW, TEST_ARENA);
    let (res, snap, prots) = run(&m, 0, &mut host, &args, &win, None);
    assert!(res.is_ok(), "a freeze, not a refusal: {res:?}");
    assert_eq!(read_state(&snap), STATE_UNWINDING, "frozen");
    (m, host, args, snap, prots)
}

/// **A region the guest created and mapped rides the codec.** The aliased pages are captured as
/// `Backed`, the artifact carries the region once, the restore rebuilds it, a re-freeze is
/// byte-identical (§12.6), and the thaw finishes as the uninterrupted run does — reading the value
/// stored before the cut out of the carried region.
#[test]
fn a_created_region_rides_a_freeze_and_thaws_aliased() {
    let (m, mut control, args) = {
        let m = instrumented(false);
        let (h, a) = creating_host();
        (m, h, a)
    };
    let (base, ..) = run(
        &m,
        0,
        &mut control,
        &args,
        &init_durable_window(WINDOW, TEST_ARENA),
        None,
    );
    assert_eq!(base, Ok(vec![Value::I64(WANT)]), "uninterrupted run");

    let (m, fhost, args, snap, prots) = frozen_creating();
    let aliased = prots[AT / temen_snapshot::PAGE..(AT + LEN) / temen_snapshot::PAGE]
        .iter()
        .all(|p| matches!(p, CapturedProt::Backed { writable: true, .. }));
    assert!(aliased, "the mapped pages are captured Backed");

    let artifact = freeze_with_prots(&m, &snap, &prots, SIZE_LOG2, &fhost).expect("rides");
    let mut thost = Host::new();
    thost.set_durable(true);
    let (rwin, rprots, _) = restore_with_prots(&artifact, &m, &mut thost).expect("restores");
    assert_eq!(
        &rwin[AT + 8..AT + 16],
        &[0u8; 8],
        "an aliased page carries no bytes of its own"
    );
    assert_eq!(
        freeze_with_prots(&m, &rwin, &rprots, SIZE_LOG2, &thost).expect("re-freeze"),
        artifact,
        "canonical re-freeze byte-identical"
    );

    let mut twin = rwin;
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let (thawed, ..) = run(&m, 0, &mut thost, &args, &twin, Some(&rprots));
    assert_eq!(
        thawed,
        Ok(vec![Value::I64(WANT)]),
        "the thaw reads the region's carried bytes through the re-aliased window"
    );
}

/// A plain in-memory region backing, so the test can keep a reference to it the way an embedder
/// might.
/// A heap backing; `.1` says whether something outside the VM can write it (a host file can).
struct TestBacking(std::sync::Mutex<Vec<u8>>, bool);

impl SharedBacking for TestBacking {
    fn size(&self) -> u64 {
        self.0.lock().unwrap().len() as u64
    }
    fn read_byte(&self, off: u64) -> u8 {
        self.0
            .lock()
            .unwrap()
            .get(off as usize)
            .copied()
            .unwrap_or(0)
    }
    fn write_byte(&self, off: u64, b: u8) {
        if let Some(x) = self.0.lock().unwrap().get_mut(off as usize) {
            *x = b;
        }
    }
    fn outside_writers(&self) -> bool {
        self.1
    }
}

/// Grant a region whose backing the caller keeps a reference to: a holder after the run.
fn region_kept_outside(host: &mut Host) -> (i32, Arc<dyn SharedBacking>) {
    let backing: Arc<dyn SharedBacking> =
        Arc::new(TestBacking(std::sync::Mutex::new(vec![0; LEN]), false));
    let h = host.grant_shared_region_backed(Arc::clone(&backing));
    (h, backing)
}

/// **A region held outside the cut declines** (R4: a cut that would split a sharing group), naming
/// the region, as does one something outside the VM can write; one only the domain holds rides.
#[test]
fn a_region_something_outside_the_cut_holds_declines() {
    let m = instrumented(true);
    let freeze_given = |host: &mut Host, args: [Value; 2]| {
        let win = init_durable_window(WINDOW, TEST_ARENA);
        let (res, snap, prots) = run(&m, 1, host, &args, &win, None);
        assert!(res.is_ok(), "{res:?}");
        assert_eq!(read_state(&snap), STATE_UNWINDING, "frozen");
        freeze_with_prots(&m, &snap, &prots, SIZE_LOG2, host)
    };

    let mut host = Host::new();
    host.set_durable(true);
    let (rh, kept) = region_kept_outside(&mut host);
    let clk = host.grant_clock();
    let r = freeze_given(&mut host, [Value::I32(rh), Value::I32(clk)]);
    assert!(
        matches!(
            r,
            Err(FreezeError::RegionNotCaptured(RegionNotCaptured {
                why: RegionRefusal::HolderOutsideCut,
                ..
            }))
        ),
        "the embedder still holds it: {r:?}"
    );
    drop(kept);

    let mut host = Host::new();
    host.set_durable(true);
    let rh = host.grant_shared_region_backed(Arc::new(TestBacking(
        std::sync::Mutex::new(vec![0; LEN]),
        true,
    )));
    let clk = host.grant_clock();
    let r = freeze_given(&mut host, [Value::I32(rh), Value::I32(clk)]);
    assert!(
        matches!(
            r,
            Err(FreezeError::RegionNotCaptured(RegionNotCaptured {
                why: RegionRefusal::OutsideWriters,
                ..
            }))
        ),
        "a host file's bytes are not the cut's: {r:?}"
    );

    let mut host = Host::new();
    host.set_durable(true);
    let rh = host.grant_shared_region(LEN);
    let clk = host.grant_clock();
    assert!(
        freeze_given(&mut host, [Value::I32(rh), Value::I32(clk)]).is_ok(),
        "held by the domain alone, it rides"
    );
}

/// **The flat `freeze` declines a region**: with no page map it cannot tell the region's mapping from
/// private bytes.
#[test]
fn a_freeze_without_a_page_map_declines_a_region() {
    let (m, fhost, _, snap, _) = frozen_creating();
    let r = freeze(&m, &snap, &fhost);
    assert!(
        matches!(
            r,
            Err(FreezeError::RegionNotCaptured(RegionNotCaptured {
                why: RegionRefusal::NoPageMap,
                ..
            }))
        ),
        "{r:?}"
    );
}

/// Step 2 — the parent `(Instantiator, child module, Budget, region)` spawns the child detached with
/// the region pre-mapped at the child's 128 KiB, then maps it at its own 128 KiB (a freeze requested
/// from the start lands there, the first call that can suspend), joins, and returns
/// `child + 2·(what the child stored)`, read back through its own mapping.
const SHARING_PARENT: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vmod: i32, vbud: i32, vrh: i32) {
  vrh64 = i64.extend_i32_s vrh
  vwin = i64.const 131072
  me = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  sl = i64.const 18
  ch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, me, gz, gz, gz, sl, gz, gz, gz, vrh64, vwin)
  vlen = i64.const 65536
  vprot = i64.const 3
  vm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vrh (vwin, gz, vlen, vprot)
  vr = call.cap 6 1 (i32) -> (i64) vinst (ch)
  vq = i64.const 131088
  vy = i64.load vq
  vy2 = i64.add vy vy
  vs = i64.add vr vy2
  return vs
  }
}
";

/// The child: a long polled loop the freeze cuts (the zero-length `unmap` is refused, and there only
/// to make the function may-suspend so the loop header polls), then it reads the region's `1111`
/// through its own alias, stores `3333` beside it, and returns what it read.
const SHARING_CHILD: &str = "memory 18 shadow 16448 65536
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vs0 = i32.wrap_i64 v1
  vz = i64.const 0
  vu = call.cap 5 1 (i64, i64) -> (i64) vs0 (vz, vz)
  br 1(vz)
}
block 1 (vi: i64) {
  vn = i64.const 1000000
  vc = i64.lt_s vi vn
  br_if vc 2(vi) 3(vi)
}
block 2 (vj: i64) {
  vo = i64.const 1
  vk = i64.add vj vo
  br 1(vk)
}
block 3 (vd: i64) {
  va = i64.const 131080
  vx = i64.load va
  vb = i64.const 131088
  vt = i64.const 3333
  i64.store vb vt
  return vx
  }
}
";
const SHARED_WANT: i64 = 1111 + 2 * 3333;

fn confined(src: &str) -> Module {
    let inst = transform_module_assume_confined(&temen_text::parse_module(src).expect("parse"))
        .expect("transform");
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

/// The parent's powerbox over `child`, with the authority to freeze its detached progeny, and a
/// region holding `1111` at byte 8 that only the powerbox keeps.
fn sharing_host(child: &Module) -> (Host, [Value; 4]) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator(0, WINDOW as u64);
    let modh = host.grant_durable_module(child);
    let budget = host.grant_budget(-1, 1 << 20, 4);
    let mut bytes = vec![0u8; LEN];
    bytes[8..16].copy_from_slice(&1111i64.to_le_bytes());
    let rh =
        host.grant_shared_region_backed(Arc::new(TestBacking(std::sync::Mutex::new(bytes), false)));
    host.grant_freeze_authority(temen_interp::FreezeScope::DetachedProgeny);
    (
        host,
        [
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(budget),
            Value::I32(rh),
        ],
    )
}

/// **A region a parent shares with its detached child rides the cut** (step 2). The artifact carries
/// it once — in the depth-0 artifact, named by number from the child's too — and the thaw rebuilds
/// one backing every domain re-aliases: the child reads the bytes carried from before the cut
/// through its re-aliased window (which holds none of its own), and the parent reads what the child
/// stores after it.
#[test]
fn a_region_shared_with_a_detached_child_rides_and_stays_shared() {
    let child = confined(SHARING_CHILD);
    let parent = confined(SHARING_PARENT);
    let (mut host, args) = sharing_host(&child);
    let win = init_durable_window(WINDOW, TEST_ARENA);
    let (base, ..) = run(&parent, 0, &mut host, &args, &win, None);
    assert_eq!(base, Ok(vec![Value::I64(SHARED_WANT)]), "uninterrupted run");

    let (mut fhost, args) = sharing_host(&child);
    let mut fwin = win.clone();
    temen_durable::write_state(&mut fwin, STATE_UNWINDING);
    let (res, snap, prots) = run(&parent, 0, &mut fhost, &args, &fwin, None);
    assert!(res.is_ok(), "a freeze, not a refusal: {res:?}");
    assert_eq!(fhost.captured_detached().len(), 1, "the child rides live");
    let child_aliased = fhost.captured_detached()[0].window.dense_prots()
        [AT / temen_snapshot::PAGE..(AT + LEN) / temen_snapshot::PAGE]
        .iter()
        .all(|p| matches!(p, CapturedProt::Backed { writable: true, .. }));
    assert!(
        child_aliased,
        "the child's pre-mapped pages are captured Backed"
    );

    let artifact = freeze_with_prots(&parent, &snap, &prots, SIZE_LOG2, &fhost).expect("rides");
    let restored = || {
        let mut thost = Host::new();
        thost.set_durable(true);
        thost.grant_durable_module(&child);
        let (rwin, rprots, _) =
            restore_with_prots(&artifact, &parent, &mut thost).expect("restores");
        (thost, rwin, rprots)
    };
    // A restore holds the child as thaw residue; captured again, the tree re-freezes to the same
    // bytes, its region numbered as before (§12.6).
    let (mut again, rwin, rprots) = restored();
    let recaptured = again
        .take_thawed_detached()
        .into_iter()
        .map(|t| temen_interp::CapturedDetached {
            parent_task: t.parent_task,
            slot: t.slot,
            window: t.window,
            reserved_log2: t.reserved_log2,
            host: Arc::new(std::sync::Mutex::new(t.host)),
            module: Arc::new(child.clone()),
            launch: t.launch,
        })
        .collect();
    again.set_captured_detached(recaptured);
    assert_eq!(
        freeze_with_prots(&parent, &rwin, &rprots, SIZE_LOG2, &again).expect("re-freeze"),
        artifact,
        "canonical re-freeze byte-identical"
    );
    let (mut thost, rwin, rprots) = restored();

    let mut twin = rwin;
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let (thawed, ..) = run(&parent, 0, &mut thost, &args, &twin, Some(&rprots));
    assert_eq!(
        thawed,
        Ok(vec![Value::I64(SHARED_WANT)]),
        "both domains of the thaw alias one region"
    );
}
