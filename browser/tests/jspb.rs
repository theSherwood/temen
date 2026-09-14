//! **A powerbox defined in JS** (`browser/src/jspb.rs`): the page names the capabilities, the
//! module's import manifest binds them by name, and every call is serviced through one seam.
//!
//! In the browser that seam is the `temen_host.js_cap_call` wasm import; natively it is the test
//! hook these tests install — so everything *between* the two (the name registry, the manifest
//! binding, slot dispatch, the bounds-checked window accessors, and the fail-closed refusal of an
//! unbound import) is proven here, off-browser, on every CI run.

use std::sync::Mutex;

use temen_browser::jspb::{
    set_test_dispatch, temen_jspb_bind, temen_jspb_error_len, temen_jspb_error_ptr,
    temen_jspb_read, temen_jspb_reset, temen_jspb_run, temen_jspb_write, JsGuestMem,
};
use temen_browser::{temen_status, STATUS_OK, STATUS_UNSUPPORTED};

/// The `temen_jspb_*` registry is process-global (single-threaded wasm by design), so the tests
/// serialize on it — the same convention as `coop_tierup_driver.rs`.
static FFI_LOCK: Mutex<()> = Mutex::new(());

/// What the stand-in "JS" servicer logged, in call order (`slot`-tagged).
static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn bind(name: &str) -> i32 {
    temen_jspb_bind(name.as_ptr(), name.len())
}

fn run(src: &str) -> i64 {
    let m = temen_text::parse_module(src).expect("guest parses");
    let bytes = temen_encode::encode_module(&m);
    temen_jspb_run(bytes.as_ptr(), bytes.len())
}

fn last_error() -> String {
    let (p, n) = (temen_jspb_error_ptr(), temen_jspb_error_len());
    if p.is_null() || n == 0 {
        return String::new();
    }
    // SAFETY: the cdylib-managed stash is live until the next run.
    String::from_utf8_lossy(unsafe { core::slice::from_raw_parts(p, n) }).into_owned()
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
    temen_jspb_reset();
    set_test_dispatch(Some(two_caps_servicer));
    // Bound in the opposite order to the guest's import declarations.
    assert_eq!(bind("js.add"), 0, "first bind takes slot 0");
    assert_eq!(bind("js.log"), 1, "second bind takes slot 1");
    assert_eq!(
        bind("js.add"),
        -1,
        "a name is one capability — rebinding is refused"
    );

    let value = run(TWO_CAPS);
    set_test_dispatch(None);

    assert_eq!(temen_status(), STATUS_OK, "run status ({})", last_error());
    assert_eq!(value, 20, "js.log returned 10, js.add doubled it");
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
    temen_jspb_reset();
    set_test_dispatch(Some(upper_servicer));
    bind("js.upper");
    let value = run(WRITE_BACK);
    set_test_dispatch(None);
    assert_eq!(temen_status(), STATUS_OK, "run status ({})", last_error());
    assert_eq!(
        value, b'A' as i64,
        "the guest read back the host's uppercase"
    );
}

/// An out-of-window read/write fails closed rather than over-reading the host's memory.
#[test]
fn window_accessors_fail_closed_outside_the_window() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    temen_jspb_reset();
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
    bind("js.upper");
    let value = run(WRITE_BACK);
    set_test_dispatch(None);
    assert_eq!(temen_status(), STATUS_OK, "run status ({})", last_error());
    assert_eq!(
        value, b'a' as i64,
        "the window is untouched — still lowercase"
    );
}

// ---- fail-closed: an import the page never defined ----------------------------------------------

#[test]
fn an_unbound_import_refuses_the_run_and_names_itself() {
    let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    LOG.lock().unwrap().clear();
    temen_jspb_reset();
    set_test_dispatch(Some(two_caps_servicer));
    bind("js.log"); // ... but the guest also imports `js.add`
    let value = run(TWO_CAPS);
    set_test_dispatch(None);

    assert_eq!(value, 0, "nothing ran");
    assert_eq!(
        temen_status(),
        STATUS_UNSUPPORTED,
        "refused before the first op"
    );
    let err = last_error();
    assert!(
        err.contains("unbound import") && err.contains("js.add"),
        "the error names the missing capability, got {err:?}"
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
    temen_jspb_reset();
    set_test_dispatch(None);
    bind("js.add");
    bind("js.log");
    let value = run(TWO_CAPS);
    assert_eq!(temen_status(), STATUS_OK, "run status ({})", last_error());
    assert_eq!(
        value, -38,
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
    temen_jspb_reset();
    set_test_dispatch(Some(card_servicer));
    for name in ["js.log", "js.upper", "js.now"] {
        assert!(bind(name) >= 0, "bind {name}");
    }
    let value = run(CARD_GUEST);
    set_test_dispatch(None);

    assert_eq!(temen_status(), STATUS_OK, "run status ({})", last_error());
    assert_eq!(value, 1_700_000_000_000, "the guest returns the host clock");
    assert_eq!(
        LOG.lock().unwrap().as_slice(),
        [
            "hello from the guest\n".to_string(),
            "THE HOST IS JAVASCRIPT\n".to_string(),
        ],
        "the second line came back uppercased — the host wrote into the guest's window"
    );
}
