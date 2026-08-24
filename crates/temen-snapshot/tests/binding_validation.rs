//! Validation of `AddressSpace`/`Instantiator` handle bindings (the §14 sub-window base/size) at
//! both codec boundaries. A frozen artifact is untrusted, persisted input and its
//! (non-cryptographic) module digest does not cover the handle bindings — so the
//! window-containment invariant the grant path guarantees (and the §14 JIT instantiator's
//! `unsafe` window copy *assumes*) must be re-checked on restore: a forged `base` smuggled into a
//! binding drives an out-of-window host pointer on the next `instantiate` — a host-memory escape.
//! Since the CONSOLIDATION §4 fold, **freeze applies the same gate** (`FreezeError::
//! BindingOutOfWindow`) so a live host holding an out-of-window range fails closed at freeze
//! rather than emitting an artifact restore would reject — which is why the reject pins below
//! forge their artifact the way an adversary would: by tampering the encoded bytes of a valid one.
//!
//! (base+size u64 overflow is pinned at the predicate's own unit test in `temen-snapshot`'s
//! `binding_window_tests` — an overflowing base has a longer ULEB spelling than any legitimate
//! one, so it can't be built by same-length tampering.)

use temen_interp::Host;
use temen_ir::Module;
use temen_snapshot::{freeze, restore, FreezeError, RestoreError};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

fn module() -> Module {
    temen_text::parse_module(&format!(
        "memory {SIZE_LOG2}\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
         \x20 v0 = i64.const 0\n\
         \x20 return v0\n\
           }}\n\
         }}\n"
    ))
    .expect("parse")
}

/// Replace the unique occurrence of `find` in `art` with the same-length `replace` — the
/// adversarial edit: the handle bindings are outside every digest, so a byte-level forgery is
/// exactly what restore must survive. Same-length keeps the section framing (ULEB lengths) intact.
fn tamper(art: &mut [u8], find: &[u8], replace: &[u8]) {
    assert_eq!(find.len(), replace.len(), "framing-preserving edit only");
    let hits: Vec<usize> = art
        .windows(find.len())
        .enumerate()
        .filter_map(|(i, w)| (w == find).then_some(i))
        .collect();
    assert_eq!(hits.len(), 1, "the binding's encoding must be unique");
    art[hits[0]..hits[0] + find.len()].copy_from_slice(replace);
}

// The encoded forms this test forges around (see `write_binding`): tag byte, ULEB base, ULEB size.
// 0x10000 and 0x40000 both encode as 3-byte ULEBs, so the swap is framing-preserving.
const B_ADDRESS_SPACE: u8 = 5;
const B_INSTANTIATOR: u8 = 6;
const ULEB_0X10000: [u8; 3] = [0x80, 0x80, 0x04];
const ULEB_0X40000: [u8; 3] = [0x80, 0x80, 0x10];

fn binding_bytes(tag: u8, base: [u8; 3], size: [u8; 3]) -> Vec<u8> {
    let mut v = vec![tag];
    v.extend_from_slice(&base);
    v.extend_from_slice(&size);
    v
}

#[test]
fn restore_accepts_in_window_address_space_binding() {
    let m = module();
    let window = vec![0u8; WINDOW];
    let mut host = Host::new();
    // The legitimate sized whole-window root grant (`base = 0`, `size = window`).
    host.grant_address_space(0, WINDOW as u64);
    let art = freeze(&m, &window, &host).expect("freeze");
    let mut rhost = Host::new();
    restore(&art, &m, &mut rhost).expect("an in-window binding must restore");
}

#[test]
fn whole_window_form_round_trips() {
    // §4: `grant_memory` is now the whole-window `AddressSpace { 0, u64::MAX }` — payload-free
    // ("the window, whatever its size"), so it must freeze and restore like the retired `Memory`
    // kind did.
    let m = module();
    let window = vec![0u8; WINDOW];
    let mut host = Host::new();
    host.grant_memory();
    let art = freeze(&m, &window, &host).expect("the whole-window form must freeze");
    let mut rhost = Host::new();
    restore(&art, &m, &mut rhost).expect("the whole-window form must restore");
}

#[test]
fn freeze_rejects_out_of_window_binding() {
    // §4: the freeze boundary now applies the restore gate too, so a live host holding an
    // out-of-window range (the grant path's caller contract violated, or a beyond-window `sub`
    // child minted off the whole-window form) fails closed here instead of emitting an
    // artifact restore would reject.
    let m = module();
    let window = vec![0u8; WINDOW];
    let mut host = Host::new();
    host.grant_instantiator(WINDOW as u64, 4096);
    assert!(
        matches!(
            freeze(&m, &window, &host),
            Err(FreezeError::BindingOutOfWindow { slot: 0 })
        ),
        "an out-of-window binding must be rejected at freeze"
    );
}

#[test]
fn restore_rejects_out_of_window_instantiator_binding() {
    let m = module();
    let window = vec![0u8; WINDOW];
    let mut host = Host::new();
    // A legitimate interior carve — then forge its encoded base out the top of the window
    // (0x10000 → 0x40000 = the window size; base+size lands past `mapped`). Restore must reject:
    // otherwise the §14 JIT instantiator would later compute `parent_mem_base + base + off`
    // outside the window and `from_raw_parts`/`copy_*` through it.
    host.grant_instantiator(0x10000, 0x10000);
    let mut art = freeze(&m, &window, &host).expect("freeze");
    tamper(
        &mut art,
        &binding_bytes(B_INSTANTIATOR, ULEB_0X10000, ULEB_0X10000),
        &binding_bytes(B_INSTANTIATOR, ULEB_0X40000, ULEB_0X10000),
    );
    let mut rhost = Host::new();
    assert_eq!(
        restore(&art, &m, &mut rhost),
        Err(RestoreError::BindingOutOfWindow),
        "an out-of-window Instantiator binding must be rejected on restore"
    );
}

#[test]
fn restore_rejects_out_of_window_address_space_binding() {
    // The AddressSpace twin of the Instantiator pin above — same forged edit, other kind. (Note
    // the whole-window form `{0, u64::MAX}` is *not* reachable by this tamper: it pattern-matches
    // exactly, and any other u64::MAX-sized spelling still fails the power-of-two checks.)
    let m = module();
    let window = vec![0u8; WINDOW];
    let mut host = Host::new();
    host.grant_address_space(0x10000, 0x10000);
    let mut art = freeze(&m, &window, &host).expect("freeze");
    tamper(
        &mut art,
        &binding_bytes(B_ADDRESS_SPACE, ULEB_0X10000, ULEB_0X10000),
        &binding_bytes(B_ADDRESS_SPACE, ULEB_0X40000, ULEB_0X10000),
    );
    let mut rhost = Host::new();
    assert_eq!(
        restore(&art, &m, &mut rhost),
        Err(RestoreError::BindingOutOfWindow),
        "an out-of-window AddressSpace binding must be rejected on restore"
    );
}
