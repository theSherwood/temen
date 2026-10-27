//! **JS capabilities** (`browser/src/jspb.rs`): the page stages capabilities by name for an
//! on-ramp open, the module's import manifest binds them by name with the open's other grants, and
//! every call is serviced through one seam.
//!
//! In the browser that seam is the `temen_host.js_cap_call` wasm import; natively it is the test
//! hook these tests install — so everything *between* the two (staging, the manifest binding, slot
//! dispatch, the bounds-checked window accessors, and the open refused for an import nothing
//! grants) is proven here, off-browser, on every CI run.

use std::sync::Mutex;

use temen_browser::jspb::{set_test_dispatch, temen_jspb_read, temen_jspb_write, JsGuestMem};
use temen_browser::{
    last_notes, onramp_exec, temen_coop_close, temen_coop_open, temen_coop_run,
    temen_onramp_stage_budget, temen_onramp_stage_clear, temen_onramp_stage_js_cap,
    temen_onramp_stage_module, temen_run_value, temen_status, temen_trap_len, temen_trap_ptr,
    PbOutcome, COOP_LEAF_ROOT, COOP_NO_REGIONS, COOP_RUN_DONE, COOP_RUN_TRAP, STATUS_OK,
    STATUS_UNSUPPORTED,
};

#[path = "../../crates/temen-interp/tests/support/rec.rs"]
mod rec;

/// What is staged for an open is per thread, and the test hook process-global, so the tests
/// serialize — the same convention as `coop_tierup_driver.rs`.
static FFI_LOCK: Mutex<()> = Mutex::new(());

/// What the stand-in "JS" servicer logged, in call order (`slot`-tagged).
static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Run `src` as `definePowerbox` does: `names` staged as its JS capabilities, slot `i` the `i`-th,
/// then an ordinary on-ramp run.
fn run(names: &[&str], src: &str) -> PbOutcome {
    temen_onramp_stage_clear();
    for (i, name) in names.iter().enumerate() {
        assert_eq!(
            stage_js_cap(name),
            i as i32,
            "staging {name} takes the next slot"
        );
    }
    let m = temen_text::parse_module(src).expect("guest parses");
    onramp_exec(&m, b"")
}

/// Read `len` bytes of the calling guest's window through the JS-facing accessor.
fn read(mem: *mut JsGuestMem, ptr: i64, len: i64) -> Option<Vec<u8>> {
    // SAFETY: `mem` is the handle of the in-flight capability call.
    let p = unsafe { temen_jspb_read(mem, ptr as u64, len as usize) };
    if p.is_null() {
        return None;
    }
    Some(unsafe { core::slice::from_raw_parts(p, len as usize) }.to_vec())
}

// ---- a guest that writes a greeting through a JS capability and adds through another ------------

/// `js.log(ptr, len)` (reads the window) then `js.add(n, n)`. The imports are declared in *this*
/// order but bound in the opposite one below — the result proves binding is by **name**, not order.
const TWO_CAPS: &str = r#"
memory 16
data 16384 "hello, js!"
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = i32.const 0
  v1 = i64.const 16384
  v2 = i64.const 10
  v3 = call.sym "js.log" (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i32.const 0
  v5 = call.sym "js.add" (i64, i64) -> (i64) v4 (v3, v3)
  return v5
  }
}
"#;

/// Slot 0 = `js.add` (returns the sum), slot 1 = `js.log` (reads the window, logs, returns `len`).
fn two_caps_servicer(slot: u32, _op: u32, args: &[i64], mem: *mut JsGuestMem) -> i64 {
    match slot {
        0 => args[0] + args[1],
        1 => {
            let bytes = read(mem, args[0], args[1]).expect("in-window read");
            LOG.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&bytes).into_owned());
            args[1]
        }
        _ => -38,
    }
}

#[test]
fn named_js_capabilities_bind_by_name_and_run() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    LOG.lock().unwrap().clear();
    temen_onramp_stage_clear();
    assert_eq!(stage_js_cap("js.add"), 0, "first stage takes slot 0");
    assert_eq!(
        stage_js_cap("js.add"),
        -1,
        "a name is one capability — staging it again is refused"
    );
    set_test_dispatch(Some(two_caps_servicer));
    // Staged in the opposite order to the guest's import declarations.
    let out = run(&["js.add", "js.log"], TWO_CAPS);
    set_test_dispatch(None);

    assert_eq!(out.status, STATUS_OK, "run status ({:?})", last_notes());
    assert_eq!(out.value, 20, "js.log returned 10, js.add doubled it");
    assert_eq!(
        LOG.lock().unwrap().as_slice(),
        ["hello, js!".to_string()],
        "the JS side read the greeting out of the guest window"
    );
}

// ---- a JS capability that writes back into the guest window -------------------------------------

/// `js.upper(ptr, len)` uppercases the bytes **in the window**; the guest then loads the first byte
/// back and returns it, so the assertion can only pass if the host's write landed in guest memory.
const WRITE_BACK: &str = r#"
memory 16
data 16384 "abc"
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = i32.const 0
  v1 = i64.const 16384
  v2 = i64.const 3
  v3 = call.sym "js.upper" (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i64.load8_u v1
  return v4
  }
}
"#;

fn upper_servicer(_slot: u32, _op: u32, args: &[i64], mem: *mut JsGuestMem) -> i64 {
    let Some(bytes) = read(mem, args[0], args[1]) else {
        return -14; // -EFAULT
    };
    let up = bytes.to_ascii_uppercase();
    // SAFETY: `mem` is the handle of the in-flight capability call.
    unsafe { temen_jspb_write(mem, args[0] as u64, up.as_ptr(), up.len()) as i64 }
}

#[test]
fn a_js_capability_writes_back_into_the_guest_window() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_test_dispatch(Some(upper_servicer));
    let out = run(&["js.upper"], WRITE_BACK);
    set_test_dispatch(None);
    assert_eq!(out.status, STATUS_OK, "run status ({:?})", last_notes());
    assert_eq!(
        out.value, b'A' as i64,
        "the guest read back the host's uppercase"
    );
}

/// An out-of-window read/write fails closed rather than over-reading the host's memory.
#[test]
fn window_accessors_fail_closed_outside_the_window() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_test_dispatch(Some(|_slot, _op, args: &[i64], mem: *mut JsGuestMem| {
        // The window is 2^16 bytes; both of these are wholly outside it.
        assert!(
            read(mem, 1 << 20, 8).is_none(),
            "out-of-window read refused"
        );
        let byte = [0u8; 1];
        // SAFETY: `mem` is the in-flight call's handle.
        assert_eq!(
            unsafe { temen_jspb_write(mem, 1 << 20, byte.as_ptr(), 1) },
            -1,
            "out-of-window write refused"
        );
        args[0]
    }));
    let out = run(&["js.upper"], WRITE_BACK);
    set_test_dispatch(None);
    assert_eq!(out.status, STATUS_OK, "run status ({:?})", last_notes());
    assert_eq!(
        out.value, b'a' as i64,
        "the window is untouched — still lowercase"
    );
}

// ---- fail-closed: an import the page never defined ----------------------------------------------

#[test]
fn an_unbound_import_refuses_the_run_and_names_itself() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    LOG.lock().unwrap().clear();
    set_test_dispatch(Some(two_caps_servicer));
    // Only `js.log`, but the guest also imports `js.add`.
    let out = run(&["js.log"], TWO_CAPS);
    set_test_dispatch(None);

    assert_eq!(out.value, 0, "nothing ran");
    assert_eq!(
        out.status, STATUS_UNSUPPORTED,
        "refused before the first op"
    );
    assert_eq!(
        last_notes(),
        vec!["refused the run: no grant satisfies its import `js.add`".to_string()],
        "the run's notes name the missing capability"
    );
    assert!(
        LOG.lock().unwrap().is_empty(),
        "the guest never executed, so the bound capability was never called either"
    );
}

/// With no servicer at all, a capability call is `-ENOSYS` — the guest sees a failed call, not a
/// host crash or an unchecked read.
#[test]
fn a_capability_with_no_servicer_returns_enosys() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_test_dispatch(None);
    let out = run(&["js.add", "js.log"], TWO_CAPS);
    assert_eq!(out.status, STATUS_OK, "run status ({:?})", last_notes());
    assert_eq!(
        out.value, -38,
        "every unserviced call is -ENOSYS, and the guest returns the last one"
    );
}

// ---- the playground card's own guest ------------------------------------------------------------

/// The guest the **"JS powerbox" playground card** ships (`browser/web/play.js`), kept in step with
/// it: three capabilities — one that reads the window, one that writes it, one pure scalar — so the
/// card's shape is gated by ordinary CI, not only by the real-browser job.
const CARD_GUEST: &str = r#"
; Every capability this guest calls is implemented in JavaScript, bound by name at instantiation.
memory 16
data 16384 "hello from the guest\n"
data 16448 "the host is javascript\n"
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  ; js.log(ptr, len) -> bytes written
  v0 = i32.const 0
  v1 = i64.const 16384
  v2 = i64.const 21
  v3 = call.sym "js.log" (i64, i64) -> (i64) v0 (v1, v2)
  ; js.upper(ptr, len) uppercases those bytes inside the guest's OWN window ...
  v4 = i32.const 0
  v5 = i64.const 16448
  v6 = i64.const 23
  v7 = call.sym "js.upper" (i64, i64) -> (i64) v4 (v5, v6)
  ; ... and the guest logs them back, so the host's write is visible from inside
  v8 = i32.const 0
  v9 = call.sym "js.log" (i64, i64) -> (i64) v8 (v5, v6)
  ; js.now() -> the host clock, returned as the guest's result
  v10 = i32.const 0
  v11 = call.sym "js.now" () -> (i64) v10 ()
  return v11
  }
}
"#;

/// Slot 0 `js.log`, slot 1 `js.upper`, slot 2 `js.now` — the card's three handlers, in Rust.
fn card_servicer(slot: u32, _op: u32, args: &[i64], mem: *mut JsGuestMem) -> i64 {
    match slot {
        0 => {
            let bytes = read(mem, args[0], args[1]).expect("in-window read");
            LOG.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&bytes).into_owned());
            args[1]
        }
        1 => {
            let up = read(mem, args[0], args[1])
                .expect("in-window read")
                .to_ascii_uppercase();
            // SAFETY: `mem` is the in-flight call's handle.
            unsafe { temen_jspb_write(mem, args[0] as u64, up.as_ptr(), up.len()) as i64 }
        }
        2 => 1_700_000_000_000,
        _ => -38,
    }
}

#[test]
fn the_playground_cards_guest_runs_under_its_three_capabilities() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    LOG.lock().unwrap().clear();
    set_test_dispatch(Some(card_servicer));
    let out = run(&["js.log", "js.upper", "js.now"], CARD_GUEST);
    set_test_dispatch(None);

    assert_eq!(out.status, STATUS_OK, "run status ({:?})", last_notes());
    assert_eq!(
        out.value, 1_700_000_000_000,
        "the guest returns the host clock"
    );
    assert_eq!(
        LOG.lock().unwrap().as_slice(),
        [
            "hello from the guest\n".to_string(),
            "THE HOST IS JAVASCRIPT\n".to_string(),
        ],
        "the second line came back uppercased — the host wrote into the guest's window"
    );
}

// ---- #2579: JS capabilities staged for an on-ramp open, on the cooperative session --------------

/// Open `src` as a cooperative session in `mode` (no declared caps, no regions to emit) and pump it
/// to its end: the run's value, or the trap's name. The modes these tests use run interpreted, so
/// any other event fails the test.
fn coop_run(src: &str, mode: i32) -> Result<i64, String> {
    let m = temen_text::parse_module(src).expect("guest parses");
    let bytes = temen_encode::encode_module(&m);
    let null = core::ptr::null();
    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), null, 0, 0, null, 0, mode);
    assert_eq!(opened, 0, "open (status {})", temen_status());
    let end = loop {
        match temen_coop_run() {
            COOP_RUN_DONE => break Ok(temen_run_value()),
            COOP_RUN_TRAP => {
                let (p, n) = (temen_trap_ptr(), temen_trap_len());
                // SAFETY: the cdylib-managed trap name is live until the next run.
                let name = unsafe { core::slice::from_raw_parts(p, n) };
                break Err(String::from_utf8_lossy(name).into_owned());
            }
            ev => panic!("an interpreted run surfaced event {ev}"),
        }
    };
    temen_coop_close();
    end
}

fn stage_js_cap(name: &str) -> i32 {
    temen_onramp_stage_js_cap(name.as_ptr(), name.len())
}

fn stage_module(name: &str, src: &str) -> i32 {
    let m = temen_text::parse_module(src).expect("module parses");
    let bytes = temen_encode::encode_module(&m);
    temen_onramp_stage_module(name.as_ptr(), name.len(), bytes.as_ptr(), bytes.len())
}

/// The one-shot guest above, on the cooperative session: its imports bind to the staged JS
/// capabilities by name, and each call is answered inside it, the window read through the seam.
#[test]
fn staged_js_capabilities_answer_a_coop_run_inside_the_call() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    LOG.lock().unwrap().clear();
    set_test_dispatch(Some(two_caps_servicer));
    assert_eq!(stage_js_cap("js.add"), 0, "the first staged cap is slot 0");
    assert_eq!(stage_js_cap("js.log"), 1, "the second is slot 1");
    assert_eq!(stage_js_cap("js.add"), -1, "a name is staged once");
    assert_eq!(stage_js_cap(""), -1, "an empty name is refused");
    let got = coop_run(TWO_CAPS, COOP_NO_REGIONS);
    set_test_dispatch(None);
    assert_eq!(got, Ok(20), "js.log returned 10, and js.add doubled it");
    assert_eq!(
        LOG.lock().unwrap().as_slice(),
        ["hello, js!".to_string()],
        "the JS side read the greeting out of the guest's window"
    );
}

/// Where the spawn record and its one grant record sit in the root's window.
const REC: u64 = 17408;
const GRANTS: u64 = 17536;

/// A root that tags its own bytes through `js.tag` (op 7), spawns the staged module `kid` with
/// `js.tag` re-granted by name, paid from `budget`, and joins it. It answers
/// `join status * 1_000_000 + its own tag * 1000 + the child's`, or `7777` if the spawn is refused.
fn tagging_root() -> String {
    let rec = temen_ir::SpawnRec {
        size_log2: 16,
        pager: u32::MAX,
        modh: 0,
        budget: 0,
        grants_ptr: GRANTS,
        grants_n: 1,
        args: (0, 0),
        region: -1,
        child_off: 0,
    };
    format!(
        r#"memory 16
data 16384 "instantiator"
data 16400 "js.tag"
data 16416 "budget"
data 16432 "kid"
data 16448 "root!"
{rec}export 0 func "_start" 0
func () -> (i64) {{
block 0 () {{
  vnp = i64.const 16384
  vnl = i64.const 12
  vinst = self.resolve vnp vnl
  vcp = i64.const 16400
  vcl = i64.const 6
  vcap = self.resolve vcp vcl
  vbp = i64.const 16416
  vbl = i64.const 6
  vbud = self.resolve vbp vbl
  vkp = i64.const 16432
  vkl = i64.const 3
  vkid = self.resolve vkp vkl
  vtp = i64.const 16448
  vtl = i64.const 5
  vroot = call.cap 13 7 (i64, i64) -> (i64) vcap (vtp, vtl)
{grant}  vr0 = i64.const {REC}
  i32.store vr0 vkid offset=8
  i32.store vr0 vbud offset=12
  vh = call.cap 6 17 (i64) -> (i32) vinst (vr0)
  vz = i32.const 0
  vbad = i32.lt_s vh vz
  br_if vbad 1() 2(vinst, vh, vroot)
}}
block 1 () {{
  vno = i64.const 7777
  return vno
}}
block 2 (vi: i32, vh2: i32, vroot2: i64) {{
  vst, vgot = call.cap 6 1 (i32) -> (i64, i64) vi (vh2)
  vm = i64.const 1000000
  vsm = i64.mul vst vm
  vk = i64.const 1000
  vrk = i64.mul vroot2 vk
  vs1 = i64.add vsm vrk
  vsum = i64.add vs1 vgot
  return vsum
  }}
}}
"#,
        rec = rec::segment(REC, &rec),
        grant = rec::grant("g", GRANTS, 16400, 6, "vcap"),
    )
}

/// The child: its window holds its own module's data, so `js.tag`'s name is at 16400 there too, and
/// other bytes than the root's at 16448. It answers what `js.tag` made of them.
const KID: &str = r#"memory 16
data 16400 "js.tag"
data 16448 "kid!!"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vcp = i64.const 16400
  vcl = i64.const 6
  vcap = self.resolve vcp vcl
  vp = i64.const 16448
  vl = i64.const 5
  vr = call.cap 13 7 (i64, i64) -> (i64) vcap (vp, vl)
  return vr
  }
}
"#;

/// `js.tag` (op 7): `1` for the root's bytes, `2` for the child's, read through the calling guest's
/// window.
fn tag_servicer(_slot: u32, op: u32, args: &[i64], mem: *mut JsGuestMem) -> i64 {
    match (op, read(mem, args[0], args[1]).as_deref()) {
        (7, Some(b"root!")) => 1,
        (7, Some(b"kid!!")) => 2,
        _ => 99,
    }
}

/// The staged module and JS capability serve the root and its child: the child resolves the module
/// by name, holds the capability by re-grant, and its call reads the child's own window. A leaf mode
/// whose leaf the engine declines (a spawning root cannot run whole) grants the run after it the
/// same staged grants.
#[test]
fn a_child_calls_a_regranted_js_capability_in_its_own_window() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_test_dispatch(Some(tag_servicer));
    let root = tagging_root();
    for mode in [COOP_NO_REGIONS, COOP_LEAF_ROOT | COOP_NO_REGIONS] {
        assert_eq!(stage_js_cap("js.tag"), 0);
        assert_eq!(stage_module("kid", KID), 0);
        assert_eq!(
            coop_run(&root, mode),
            Ok(1002),
            "mode {mode}: the root's tag is 1, the child's 2, and the join succeeds"
        );
    }
    set_test_dispatch(None);
}

/// A staged budget is the one a spawn pays from: one whose `mem` cannot hold the child's window
/// refuses the spawn that the default budget (one window) admits, and an unbounded one admits it.
#[test]
fn a_staged_budget_is_what_a_spawn_pays_from() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_test_dispatch(Some(tag_servicer));
    let root = tagging_root();
    let run = |budget: Option<[i64; 3]>| {
        stage_js_cap("js.tag");
        stage_module("kid", KID);
        if let Some([fuel, mem, spawn]) = budget {
            temen_onramp_stage_budget(fuel, mem, spawn);
        }
        coop_run(&root, COOP_NO_REGIONS)
    };
    assert_eq!(run(None), Ok(1002), "the default budget holds one window");
    assert_eq!(
        run(Some([-1, 1000, -1])),
        Ok(7777),
        "1000 bytes hold no window"
    );
    assert_eq!(
        run(Some([-1, -1, -1])),
        Ok(1002),
        "an unbounded budget holds it"
    );
    set_test_dispatch(None);
}

/// What is staged serves the next open alone, and a clear drops it: the root then resolves no
/// `js.tag`, and its call on that handle traps.
#[test]
fn staged_grants_serve_one_open_and_a_clear_drops_them() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_test_dispatch(Some(tag_servicer));
    let root = tagging_root();
    let garbage = [0u8; 4];
    let name = "kid";
    assert!(
        temen_onramp_stage_module(name.as_ptr(), name.len(), garbage.as_ptr(), garbage.len()) < 0,
        "bytes that do not decode are refused"
    );
    stage_js_cap("js.tag");
    stage_module("kid", KID);
    temen_onramp_stage_clear();
    let cleared = coop_run(&root, COOP_NO_REGIONS);
    assert!(
        cleared.is_err(),
        "nothing staged, so no js.tag to call: {cleared:?}"
    );
    stage_js_cap("js.tag");
    stage_module("kid", KID);
    assert_eq!(coop_run(&root, COOP_NO_REGIONS), Ok(1002));
    let again = coop_run(&root, COOP_NO_REGIONS);
    assert!(
        again.is_err(),
        "the open before took what was staged: {again:?}"
    );
    set_test_dispatch(None);
}
