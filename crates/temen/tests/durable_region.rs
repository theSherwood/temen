//! #2025 — **a §13 region rides a freeze with its sharing group** (R4, DURABILITY.md §4).
//!
//! A region a domain holds or maps is captured once, its bytes keyed by an artifact number; the
//! window's aliased pages name it instead of carrying bytes; and the restore rebuilds it and the run
//! re-aliases it before the guest resumes — on the interpreter and on the native JIT, whose page map
//! names the region each aliased page maps (step 3). A region that something outside the cut also
//! holds declines, as does a freeze given no page map.
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
    run_capture_reserved_with_host_prots, CapturedProt, Host, MemLayout, RegionNotCaptured,
    RegionRefusal, SharedBacking, Value,
};
use temen_ir::Module;
use temen_jit::JitOutcome;
use temen_snapshot::{freeze, freeze_layout, freeze_with_prots, restore_with_prots, FreezeError};

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

/// [`run`] on the native JIT: `m`'s `func` over `layout`, its result (`None` for a run the freeze
/// unwound) and the window it left, page map included.
fn run_jit(
    m: &Module,
    func: u32,
    host: &mut Host,
    args: &[Value],
    layout: &MemLayout,
) -> (Option<i64>, MemLayout) {
    let args: Vec<i64> = args
        .iter()
        .map(|a| match a {
            Value::I32(x) => *x as i64,
            other => panic!("unexpected arg {other:?}"),
        })
        .collect();
    let (out, left) =
        temen_run::jit_cap_run(m, func, &args, layout, SIZE_LOG2, 0, host, None).expect("JIT run");
    let res = match out {
        JitOutcome::Returned(v) if read_state(left.bytes()) != STATE_UNWINDING => Some(v[0]),
        JitOutcome::Returned(_) => None,
        other => panic!("unexpected outcome {other:?}"),
    };
    (res, left)
}

/// A restored window as the JIT is handed it: its bytes set rewinding, under its page map.
fn thaw_layout(mut win: Vec<u8>, prots: &[CapturedProt]) -> MemLayout {
    begin_thaw(&mut win, TEST_ARENA, 0);
    MemLayout::from_dense(win, prots, WINDOW as u64)
}

/// The creating guest's host: a whole-window `AddressSpace` and a clock.
fn creating_host() -> (Host, [Value; 2]) {
    let mut host = Host::new();
    host.set_durable(true);
    // An OS-backed region, which the JIT maps for real (the interpreter reads it the same).
    host.set_region_factory(temen_run::new_shared_region);
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

/// **Step 3 — a region the guest created rides a freeze of its native JIT run.** The JIT's page map
/// names the region each aliased page maps, so the capture is the interpreter's page for page and
/// the artifact the interpreter's byte for byte; and either engine thaws it, the JIT re-aliasing the
/// rebuilt region into its fresh window before the guest resumes.
#[test]
fn a_created_region_rides_a_jit_freeze_and_thaws_on_either_engine() {
    let (m, ihost, args, isnap, iprots) = frozen_creating();
    let oracle =
        freeze_with_prots(&m, &isnap, &iprots, SIZE_LOG2, &ihost).expect("the oracle's artifact");

    let (mut host, _) = creating_host();
    let fresh = MemLayout::image(init_durable_window(WINDOW, TEST_ARENA));
    let plain = instrumented(false);
    assert_eq!(
        run_jit(&plain, 0, &mut host, &args, &fresh).0,
        Some(WANT),
        "uninterrupted run"
    );

    let (mut fhost, _) = creating_host();
    let (res, snap) = run_jit(&m, 0, &mut fhost, &args, &fresh);
    assert_eq!(res, None, "frozen");
    let prots = snap.dense_prots();
    assert_eq!(
        prots[AT / temen_snapshot::PAGE..(AT + LEN) / temen_snapshot::PAGE],
        iprots[AT / temen_snapshot::PAGE..(AT + LEN) / temen_snapshot::PAGE],
        "the JIT names the region its pages alias, as the interpreter does"
    );
    let artifact = freeze_layout(&m, &snap, SIZE_LOG2, &fhost).expect("rides");
    assert_eq!(artifact, oracle, "the JIT's artifact is the interpreter's");

    let restored = || {
        let mut thost = Host::new();
        thost.set_durable(true);
        thost.set_region_factory(temen_run::new_shared_region);
        let (rwin, rprots, _) = restore_with_prots(&artifact, &m, &mut thost).expect("restores");
        (thost, rwin, rprots)
    };
    let (mut thost, rwin, rprots) = restored();
    assert_eq!(
        run_jit(&m, 0, &mut thost, &args, &thaw_layout(rwin, &rprots)).0,
        Some(WANT),
        "the JIT thaw reads the carried bytes through its re-aliased window"
    );
    let (mut thost, mut rwin, rprots) = restored();
    begin_thaw(&mut rwin, TEST_ARENA, 0);
    assert_eq!(
        run(&m, 0, &mut thost, &args, &rwin, Some(&rprots)).0,
        Ok(vec![Value::I64(WANT)]),
        "and so does the interpreter's"
    );
}

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
    fn atomic(&self, off: u64, width: u32, f: &mut dyn FnMut(u64) -> Option<u64>) -> u64 {
        let mut buf = self.0.lock().unwrap();
        let Some(word) = buf.get_mut(off as usize..off as usize + width as usize) else {
            return 0;
        };
        let old = word
            .iter()
            .rev()
            .fold(0u64, |v, &b| (v << 8) | u64::from(b));
        if let Some(new) = f(old) {
            for (k, b) in word.iter_mut().enumerate() {
                *b = (new >> (8 * k)) as u8;
            }
        }
        old
    }
    fn outside_writers(&self) -> bool {
        self.1
    }
}

/// A `LEN`-byte heap region nothing outside the VM writes.
fn heap_region() -> Arc<dyn SharedBacking> {
    Arc::new(TestBacking(std::sync::Mutex::new(vec![0; LEN]), false))
}

/// Grant a region whose backing the caller keeps a reference to: a holder after the run.
fn region_kept_outside(host: &mut Host) -> (i32, Arc<dyn SharedBacking>) {
    let backing = heap_region();
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
/// region holding `1111` at byte 8 that only the powerbox keeps: `backing`, which the native JIT
/// needs OS-backed ([`temen_run::new_shared_region`]).
fn sharing_host(child: &Module, backing: Arc<dyn SharedBacking>) -> (Host, [Value; 4]) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator(0, WINDOW as u64);
    let modh = host.grant_durable_module(child);
    let budget = host.grant_budget(-1, 1 << 20, 4);
    for (o, b) in 1111i64.to_le_bytes().into_iter().enumerate() {
        backing.write_byte(8 + o as u64, b);
    }
    let rh = host.grant_shared_region_backed(backing);
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
    let (mut host, args) = sharing_host(&child, heap_region());
    let win = init_durable_window(WINDOW, TEST_ARENA);
    let (base, ..) = run(&parent, 0, &mut host, &args, &win, None);
    assert_eq!(base, Ok(vec![Value::I64(SHARED_WANT)]), "uninterrupted run");

    let (mut fhost, args) = sharing_host(&child, heap_region());
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

/// **Step 3 — a region a JIT parent shares with its detached child rides the cut.** The JIT freeze
/// lands at the spawn, before the parent's own `map`: the child's JIT page map names the region its
/// pre-mapped pages alias, and the parent holds the handle. The thaw rebuilds one backing; the
/// re-launched child re-aliases it into its fresh window and reads the bytes carried from before the
/// cut, and the parent maps the same backing and reads what the child stores after it — on the
/// JIT, and on the interpreter from the same artifact.
#[test]
fn a_region_shared_with_a_detached_child_rides_a_jit_freeze() {
    let child = confined(SHARING_CHILD);
    let parent = confined(SHARING_PARENT);
    let shm = || temen_run::new_shared_region(LEN);
    let fresh = init_durable_window(WINDOW, TEST_ARENA);
    let (mut host, args) = sharing_host(&child, shm());
    assert_eq!(
        run_jit(
            &parent,
            0,
            &mut host,
            &args,
            &MemLayout::image(fresh.clone())
        )
        .0,
        Some(SHARED_WANT),
        "uninterrupted run"
    );

    let mut fwin = fresh;
    temen_durable::write_state(&mut fwin, STATE_UNWINDING);
    // The JIT runs the child on its own thread, so on a loaded runner it can finish its loop before
    // the freeze reaches it; that run is not the case under test (as #1760).
    let mut attempts = 0;
    let (fhost, snap) = loop {
        let (mut fhost, _) = sharing_host(&child, shm());
        let (res, snap) = run_jit(
            &parent,
            0,
            &mut fhost,
            &args,
            &MemLayout::image(fwin.clone()),
        );
        assert_eq!(res, None, "frozen");
        if fhost.captured_detached().len() == 1 {
            break (fhost, snap);
        }
        attempts += 1;
        assert!(attempts < 50, "the JIT never froze the child live");
    };
    let child_aliased = fhost.captured_detached()[0].window.dense_prots()
        [AT / temen_snapshot::PAGE..(AT + LEN) / temen_snapshot::PAGE]
        .iter()
        .all(|p| matches!(p, CapturedProt::Backed { writable: true, .. }));
    assert!(
        child_aliased,
        "the child's pre-mapped pages are captured Backed"
    );

    let artifact = freeze_layout(&parent, &snap, SIZE_LOG2, &fhost).expect("rides");
    let restored = || {
        let mut thost = Host::new();
        thost.set_durable(true);
        thost.set_region_factory(temen_run::new_shared_region);
        thost.grant_durable_module(&child);
        let (rwin, rprots, _) =
            restore_with_prots(&artifact, &parent, &mut thost).expect("restores");
        (thost, rwin, rprots)
    };
    let (mut thost, rwin, rprots) = restored();
    assert_eq!(
        run_jit(&parent, 0, &mut thost, &args, &thaw_layout(rwin, &rprots)).0,
        Some(SHARED_WANT),
        "both domains of the JIT thaw alias one region"
    );
    let (mut thost, mut rwin, rprots) = restored();
    begin_thaw(&mut rwin, TEST_ARENA, 0);
    assert_eq!(
        run(&parent, 0, &mut thost, &args, &rwin, Some(&rprots)).0,
        Ok(vec![Value::I64(SHARED_WANT)]),
        "and of the interpreter's"
    );
}

/// #2220 — the parent `(Instantiator, child module, Budget, mailbox, region)` spawns the child
/// detached with the mailbox pre-mapped at the child's 128 KiB, then grants the running child
/// `region` (`Instantiator.grant`, op 19). `FLIP` requests the freeze just before the grant, so the
/// freeze lands at it, once the grant has taken effect. The parent then maps the mailbox at its own
/// 128 KiB, posts the child's index of the region in its first word and wakes the child, maps the
/// region at its own 192 KiB, joins, and returns `index·1_000_000 + child + 2·(what the child
/// stored)`, read back through its mapping: a thaw that granted again would answer a second index.
const GRANTING_PARENT: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vmod: i32, vbud: i32, vmb: i32, vgr: i32) {
  vmb64 = i64.extend_i32_s vmb
  vwin = i64.const 131072
  me = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  sl = i64.const 18
  ch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, me, gz, gz, gz, sl, gz, gz, gz, vmb64, vwin)
FLIP
  vgr64 = i64.extend_i32_s vgr
  vh = call.cap 6 19 (i32, i64) -> (i32) vinst (ch, vgr64)
  vlen = i64.const 65536
  vprot = i64.const 3
  vm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vmb (vwin, gz, vlen, vprot)
  vh64 = i64.extend_i32_s vh
  i64.atomic.store vwin vh64
  vone = i32.const 1
  vwoke = atomic.notify vwin vone
  vgat = i64.const 196608
  vm2 = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vgr (vgat, gz, vlen, vprot)
  vr = call.cap 6 1 (i32) -> (i64) vinst (ch)
  vq = i64.const 196624
  vy = i64.load vq
  vy2 = i64.add vy vy
  vs0 = i64.add vr vy2
  vmil = i64.const 1000000
  vhm = i64.mul vh64 vmil
  vs = i64.add vs0 vhm
  return vs
  }
}
";

/// The child: [`SHARING_CHILD`]'s polled loop, which the freeze cuts, then a wait for the index its
/// parent posts in the mailbox, parked between looks so the parent runs (`-1` if it never comes).
/// Through the index the child maps the granted region at its 192 KiB, reads the parent's `2222`,
/// stores `3333` beside it, and returns what it read. Had the region not come back in its
/// powerbox, the `map` would fail and the child would read and write its own window.
const GRANTED_CHILD: &str = "memory 18 shadow 16448 65536
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
  va = i64.const 131072
  vh = i64.atomic.load va
  vz = i64.const 0
  vw = i64.eq vh vz
  br_if vw 5(vd) 4(vh)
}
block 4 (vg: i64) {
  vr = i32.wrap_i64 vg
  vat = i64.const 196608
  vz = i64.const 0
  vlen = i64.const 65536
  vprot = i64.const 3
  vm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vr (vat, vz, vlen, vprot)
  vp = i64.const 196616
  vx = i64.load vp
  vq = i64.const 196624
  vt = i64.const 3333
  i64.store vq vt
  return vx
}
block 5 (ve: i64) {
  va = i64.const 131072
  vz = i64.const 0
  vto = i64.const 1000000
  vws = i64.atomic.wait va vz vto
  vlim = i64.const 1010000
  vmore = i64.lt_s ve vlim
  br_if vmore 2(ve) 6()
}
block 6 () {
  vgone = i64.const -1
  return vgone
  }
}
";
const GRANTED_WANT: i64 = 2222 + 2 * 3333;

fn granting_parent(flip: bool) -> Module {
    confined(&GRANTING_PARENT.replace("FLIP", if flip { FLIP } else { "" }))
}

/// [`sharing_host`] over `mailbox`, and `region`, holding `2222` at byte 8, for the parent to grant.
fn granting_host(
    child: &Module,
    mailbox: Arc<dyn SharedBacking>,
    region: Arc<dyn SharedBacking>,
) -> (Host, [Value; 5]) {
    let (mut host, [inst, modh, budget, mbox]) = sharing_host(child, mailbox);
    for (o, b) in 2222i64.to_le_bytes().into_iter().enumerate() {
        region.write_byte(8 + o as u64, b);
    }
    let rh = host.grant_shared_region_backed(region);
    (host, [inst, modh, budget, mbox, Value::I32(rh)])
}

/// The parent's answer, checked to be `index·1_000_000 + GRANTED_WANT` for some granted index.
fn granted_answer(answer: i64) -> i64 {
    assert!(
        answer / 1_000_000 > 0 && answer % 1_000_000 == GRANTED_WANT,
        "the parent answered {answer}"
    );
    answer
}

/// The regions the frozen child's powerbox holds: the one its spawn pre-mapped, and the one its
/// parent granted it once the grant has landed.
fn child_regions(host: &Host) -> usize {
    let child = host.captured_detached()[0].host.lock().unwrap();
    let handles = child.capture_durable_handles().expect("a durable powerbox");
    handles
        .iter()
        .filter(|h| matches!(h.binding, temen_interp::DurableBinding::SharedRegion { .. }))
        .count()
}

/// **#2220 — a capability granted into a running child rides a freeze like any other.** The freeze
/// lands at the grant, so the captured child's powerbox holds the granted region beside the
/// pre-mapped one; the thaw reloads the index the grant answered and restores the region under it:
/// the child maps it and reads the bytes carried from before the cut, and the parent reads what the
/// child stores.
#[test]
fn a_region_granted_into_a_running_child_rides_a_freeze() {
    let child = confined(GRANTED_CHILD);
    let win = init_durable_window(WINDOW, TEST_ARENA);
    let (mut host, args) = granting_host(&child, heap_region(), heap_region());
    let (base, ..) = run(&granting_parent(false), 0, &mut host, &args, &win, None);
    let want = match base.as_deref() {
        Ok([Value::I64(x)]) => granted_answer(*x),
        other => panic!("uninterrupted run: {other:?}"),
    };

    let parent = granting_parent(true);
    let (mut fhost, args) = granting_host(&child, heap_region(), heap_region());
    let (res, snap, prots) = run(&parent, 0, &mut fhost, &args, &win, None);
    assert!(res.is_ok(), "a freeze, not a refusal: {res:?}");
    assert_eq!(read_state(&snap), STATE_UNWINDING, "frozen");
    assert_eq!(fhost.captured_detached().len(), 1, "the child rides live");
    assert_eq!(
        child_regions(&fhost),
        2,
        "the granted region rides in its powerbox"
    );

    let artifact = freeze_with_prots(&parent, &snap, &prots, SIZE_LOG2, &fhost).expect("rides");
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.grant_durable_module(&child);
    let (mut twin, rprots, _) =
        restore_with_prots(&artifact, &parent, &mut thost).expect("restores");
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let (thawed, ..) = run(&parent, 0, &mut thost, &args, &twin, Some(&rprots));
    assert_eq!(
        thawed,
        Ok(vec![Value::I64(want)]),
        "the thawed child maps the region its parent granted it, under the same index"
    );
}

/// **#2220 — on the native JIT too.** The grant reaches the child through the powerbox the nursery
/// retains for it, a freeze captures the granted region there, and the JIT thaw and the
/// interpreter's, from the same artifact, both give the uninterrupted run's answer.
#[test]
fn a_region_granted_into_a_running_child_rides_a_jit_freeze() {
    let child = confined(GRANTED_CHILD);
    let shm = || temen_run::new_shared_region(LEN);
    let fresh = init_durable_window(WINDOW, TEST_ARENA);
    let (mut host, args) = granting_host(&child, shm(), shm());
    let image = || MemLayout::image(fresh.clone());
    let base = run_jit(&granting_parent(false), 0, &mut host, &args, &image()).0;
    let want = granted_answer(base.expect("uninterrupted run"));

    // The child cannot finish before the freeze reaches it: it waits for an index the parent posts
    // only after the cut.
    let parent = granting_parent(true);
    let (mut fhost, _) = granting_host(&child, shm(), shm());
    let (res, snap) = run_jit(&parent, 0, &mut fhost, &args, &image());
    assert_eq!(res, None, "frozen");
    assert_eq!(fhost.captured_detached().len(), 1, "the child rides live");
    assert_eq!(
        child_regions(&fhost),
        2,
        "the granted region rides in its powerbox"
    );

    let artifact = freeze_layout(&parent, &snap, SIZE_LOG2, &fhost).expect("rides");
    let restored = || {
        let mut thost = Host::new();
        thost.set_durable(true);
        thost.set_region_factory(temen_run::new_shared_region);
        thost.grant_durable_module(&child);
        let (rwin, rprots, _) =
            restore_with_prots(&artifact, &parent, &mut thost).expect("restores");
        (thost, rwin, rprots)
    };
    let (mut thost, rwin, rprots) = restored();
    assert_eq!(
        run_jit(&parent, 0, &mut thost, &args, &thaw_layout(rwin, &rprots)).0,
        Some(want),
        "the JIT thaw maps the granted region"
    );
    let (mut thost, mut rwin, rprots) = restored();
    begin_thaw(&mut rwin, TEST_ARENA, 0);
    assert_eq!(
        run(&parent, 0, &mut thost, &args, &rwin, Some(&rprots)).0,
        Ok(vec![Value::I64(want)]),
        "and so does the interpreter's"
    );
}

/// #2051 — a bytecode **reactor** (the browser save-state's engine): func 0 `(AddressSpace)` mints a
/// region, maps it at 128 KiB and again at 192 KiB, stores `1111` through the first, and returns the
/// handle; func 1 `(region)` maps a region it is handed at 128 KiB; func 2 (`tick`) stores `3333`
/// through the first mapping and returns `first + 2·second` read back through the second — `1111 +
/// 2·3333` only while both mappings alias one region holding the bytes stored before.
const REACTOR_SRC: &str = "memory 18
func (i32) -> (i32) {
block 0 (vas: i32) {
  vlen = i64.const 65536
  vrh = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vr = i32.wrap_i64 vrh
  va = i64.const 131072
  vb = i64.const 196608
  vz = i64.const 0
  vprot = i64.const 3
  vm1 = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vr (va, vz, vlen, vprot)
  vm2 = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vr (vb, vz, vlen, vprot)
  vp = i64.const 131080
  vone = i64.const 1111
  i64.store vp vone
  return vr
  }
}
func (i32) -> (i64) {
block 0 (vr: i32) {
  va = i64.const 131072
  vz = i64.const 0
  vlen = i64.const 65536
  vprot = i64.const 3
  vm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vr (va, vz, vlen, vprot)
  return vm
  }
}
func () -> (i64) {
block 0 () {
  vp = i64.const 131096
  vt = i64.const 3333
  i64.store vp vt
  vq = i64.const 196616
  vx = i64.load vq
  vw = i64.const 196632
  vy = i64.load vw
  vy2 = i64.add vy vy
  vs = i64.add vx vy2
  return vs
  }
}
";
const REACTOR_WANT: i64 = 1111 + 2 * 3333;

/// `func(args)` on a live reactor.
fn tick(
    r: &mut temen_interp::bytecode::Reactor,
    host: &mut Host,
    func: u32,
    args: &[Value],
) -> Value {
    let mut fuel = 1_000_000u64;
    let out = r.call(func, args, &mut fuel, host).expect("reactor call");
    out[0]
}

/// **#2051 — a bytecode reactor's save-state carries the region its window aliases.** The reactor's
/// live window is no holder of its own, so the region rides the freeze its frame boundary takes; the
/// artifact carries its bytes once; the thawed reactor's window re-aliases the rebuilt region at both
/// offsets, reads the bytes stored before the freeze, and shares what it stores after; a thawed
/// reactor re-freezes byte-identical. A moment, which carries no region bytes, still refuses.
#[test]
fn a_bytecode_reactor_save_state_carries_its_region() {
    use temen_interp::{
        bytecode::Reactor,
        moment::{Moment, Refusal},
    };
    let m = temen_text::parse_module(REACTOR_SRC).expect("parse");
    let mut host = Host::new();
    let asp = host.grant_address_space(0, WINDOW as u64);
    let mut r = Reactor::open(&m).expect("open");
    tick(&mut r, &mut host, 0, &[Value::I32(asp)]);
    assert_eq!(
        tick(&mut r, &mut host, 2, &[]),
        Value::I64(REACTOR_WANT),
        "live"
    );

    let layout = r.window_layout().expect("a window");
    assert!(
        layout.aliases_regions(),
        "the mapped pages are captured as the region"
    );
    assert_eq!(
        Moment::capture(layout.clone(), &host).err(),
        Some(Refusal::Region),
        "a moment cannot carry the region's bytes"
    );
    let reserved = r.window_reserved_log2().expect("a window");
    let artifact = freeze_layout(&m, &layout, reserved, &host).expect("rides");

    let thawed = || {
        let mut thost = Host::new();
        let (rlayout, _) =
            temen_snapshot::restore_layout(&artifact, &m, &mut thost).expect("restores");
        let mut r2 = Reactor::open(&m).expect("open");
        assert!(r2.restore_window(&rlayout, &thost), "re-aliases");
        (r2, thost)
    };
    let (r2, thost) = thawed();
    assert_eq!(
        freeze_layout(&m, &r2.window_layout().expect("a window"), reserved, &thost)
            .expect("re-freeze"),
        artifact,
        "canonical re-freeze byte-identical"
    );
    let (mut r2, mut thost) = thawed();
    assert_eq!(
        tick(&mut r2, &mut thost, 2, &[]),
        Value::I64(REACTOR_WANT),
        "both offsets alias the rebuilt region, which holds the bytes from before the freeze"
    );
}

/// **#2051 — a live reactor's window does not hide a holder outside the cut.** A region the embedder
/// also keeps still declines a reactor's freeze, naming the region.
#[test]
fn a_bytecode_reactor_region_the_embedder_holds_declines() {
    use temen_interp::bytecode::Reactor;
    let m = temen_text::parse_module(REACTOR_SRC).expect("parse");
    let mut host = Host::new();
    let (rh, kept) = region_kept_outside(&mut host);
    let mut r = Reactor::open(&m).expect("open");
    assert_eq!(
        tick(&mut r, &mut host, 1, &[Value::I32(rh)]),
        Value::I64(0),
        "mapped"
    );
    let layout = r.window_layout().expect("a window");
    let reserved = r.window_reserved_log2().expect("a window");
    let refused = freeze_layout(&m, &layout, reserved, &host);
    assert!(
        matches!(
            refused,
            Err(FreezeError::RegionNotCaptured(RegionNotCaptured {
                why: RegionRefusal::HolderOutsideCut,
                ..
            }))
        ),
        "the embedder still holds it: {refused:?}"
    );
    drop(kept);
}

/// **#2059 — rewinding a reactor to a moment leaves a region it has mapped since untouched.** The
/// moment was taken before the guest mapped the embedder's region, so it holds the window's own
/// bytes there; the restore writes them into the window, not through the alias into the region every
/// other holder reads.
#[test]
fn rewinding_a_reactor_past_a_region_map_leaves_the_region_alone() {
    use temen_interp::{bytecode::Reactor, moment::Moment};
    let m = temen_text::parse_module(REACTOR_SRC).expect("parse");
    let mut host = Host::new();
    let (rh, kept) = region_kept_outside(&mut host);
    for (o, b) in 0xabcdi64.to_le_bytes().into_iter().enumerate() {
        kept.write_byte(8 + o as u64, b);
    }
    let mut r = Reactor::open(&m).expect("open");
    let before =
        Moment::capture(r.window_layout().expect("a window"), &host).expect("nothing aliased yet");
    assert_eq!(
        tick(&mut r, &mut host, 1, &[Value::I32(rh)]),
        Value::I64(0),
        "mapped"
    );

    assert!(r.restore_window(&before.layout(), &host), "rewinds");
    let region: Vec<u8> = (8..16).map(|o| kept.read_byte(o)).collect();
    assert_eq!(
        region,
        0xabcdi64.to_le_bytes(),
        "the region keeps what the embedder wrote"
    );
    let window = r.window_layout().expect("a window");
    assert!(
        !window.aliases_regions(),
        "the rewound window is the moment's: private"
    );
    assert_eq!(
        &window.bytes()[AT + 8..AT + 16],
        &[0u8; 8],
        "and reads the moment's bytes"
    );
}
