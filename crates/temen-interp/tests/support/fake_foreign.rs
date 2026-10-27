//! A detached child's window in a memory of the test's own, as a browser's is a `WebAssembly.Memory`
//! of the child's own (#1414 B6-3c, 5d): byte vectors the engine reaches only through [`FAKE_OPS`],
//! handed to a run by [`temen_interp::Host::set_child_window`]. Include with
//! `#[path = "support/fake_foreign.rs"] mod fake_foreign;`.
#![allow(dead_code)] // each test binary uses a different subset

use std::sync::{Arc, Mutex};

use temen_interp::{ForeignOps, Region};

/// The test's own foreign memories, as a browser's are a child's `WebAssembly.Memory`: byte
/// vectors the engine reaches only through [`FAKE_OPS`].
static FAKE_MEMS: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

fn fake_read_span(id: u32, off: u64, out: &mut [u8]) {
    let o = off as usize;
    out.copy_from_slice(&FAKE_MEMS.lock().unwrap()[id as usize][o..o + out.len()]);
}

fn fake_write_span(id: u32, off: u64, data: &[u8]) {
    let o = off as usize;
    FAKE_MEMS.lock().unwrap()[id as usize][o..o + data.len()].copy_from_slice(data);
}

unsafe extern "C" fn fake_read(id: u32, off: u64, dst: *mut u8, len: usize) {
    // SAFETY: the caller's contract (`ForeignOps::read`).
    fake_read_span(id, off, unsafe { std::slice::from_raw_parts_mut(dst, len) })
}

unsafe extern "C" fn fake_write(id: u32, off: u64, src: *const u8, len: usize) {
    // SAFETY: the caller's contract (`ForeignOps::write`).
    fake_write_span(id, off, unsafe { std::slice::from_raw_parts(src, len) })
}

extern "C" fn fake_load(id: u32, off: u64, width: u32) -> u64 {
    let mut b = [0u8; 8];
    fake_read_span(id, off, &mut b[..width as usize]);
    u64::from_le_bytes(b)
}

extern "C" fn fake_store(id: u32, off: u64, width: u32, val: u64) {
    fake_write_span(id, off, &val.to_le_bytes()[..width as usize]);
}

extern "C" fn fake_fill(id: u32, off: u64, len: u64, b: u8) {
    FAKE_MEMS.lock().unwrap()[id as usize][off as usize..(off + len) as usize].fill(b);
}

extern "C" fn fake_copy(id: u32, dst: u64, src: u64, len: u64) {
    let range = src as usize..(src + len) as usize;
    FAKE_MEMS.lock().unwrap()[id as usize].copy_within(range, dst as usize);
}

/// A load, a store or a compare-exchange: what these guests' runs perform.
extern "C" fn fake_atomic(id: u32, kind: u32, off: u64, width: u32, a: u64, b: u64) -> u64 {
    let mut mems = FAKE_MEMS.lock().unwrap();
    let cell = &mut mems[id as usize][off as usize..(off + width as u64) as usize];
    let mut raw = [0u8; 8];
    raw[..cell.len()].copy_from_slice(cell);
    let old = u64::from_le_bytes(raw);
    let new = match kind {
        0 => return old,
        1 => a,
        8 if old == a => b,
        8 => return old,
        k => unreachable!("atomic kind {k}: these runs load, store and compare-exchange only"),
    };
    let len = cell.len();
    cell.copy_from_slice(&new.to_le_bytes()[..len]);
    old
}

extern "C" fn fake_grow(id: u32, len: u64) -> u32 {
    let mut mems = FAKE_MEMS.lock().unwrap();
    let m = &mut mems[id as usize];
    if m.len() < len as usize {
        m.resize(len as usize, 0);
    }
    1
}

/// The fake memories the engine let go of ([`ForeignOps::release`]), in order.
static RELEASED: Mutex<Vec<u32>> = Mutex::new(Vec::new());

extern "C" fn fake_release(id: u32) {
    RELEASED.lock().unwrap().push(id);
}

pub static FAKE_OPS: ForeignOps = ForeignOps {
    read: fake_read,
    write: fake_write,
    load: fake_load,
    store: fake_store,
    fill: fake_fill,
    copy_within: fake_copy,
    atomic: fake_atomic,
    grow: fake_grow,
    release: fake_release,
};

/// A child window in a fresh foreign memory of the test's own
/// ([`temen_interp::Host::set_child_window`]).
pub fn fake_child_window(size_log2: u8) -> Option<Arc<Region>> {
    let len = 1u64 << size_log2;
    let mut mems = FAKE_MEMS.lock().unwrap();
    mems.push(vec![0; len as usize]);
    let id = mems.len() as u32 - 1;
    Some(Arc::new(Region::foreign(id, len, FAKE_OPS)))
}

/// The word at `at` of the test's foreign memory `id`.
pub fn fake_word(id: u32, at: usize) -> i64 {
    let mems = FAKE_MEMS.lock().unwrap();
    i64::from_le_bytes(mems[id as usize][at..at + 8].try_into().unwrap())
}

/// Whether the engine let go of fake memory `id` ([`ForeignOps::release`]).
pub fn released(id: u32) -> bool {
    RELEASED.lock().unwrap().contains(&id)
}
