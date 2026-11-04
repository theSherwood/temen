//! The opcode pin the spec suites share (SPEC.md suite 3, the SIMD rows, slice 7). A module and a
//! baseline that differ only in one op of function 0 encode the same bytes up to that function's
//! length, which differs when the two ops differ in size (wire v17, #2596). Past it, the first byte
//! at which the two bodies differ is the op's opcode.

use temen_ir::Module;

/// The offset in `bytes`, `m` encoded by `enc`, of the first byte of function 0's body that differs
/// from the body in `base_bytes`, a module that differs from `m` only in function 0. `None` if the
/// bodies agree as far as the shorter goes.
pub fn first_divergence(
    m: &Module,
    bytes: &[u8],
    base_bytes: &[u8],
    enc: impl Fn(&Module) -> Vec<u8>,
) -> Option<usize> {
    // The same module with no functions ends with what comes before the bodies, then the count:
    // one byte, as it is for any count under 128. Function 0's length follows the count.
    assert!(m.funcs.len() < 128 && m.debug_info.is_none());
    let none = Module {
        funcs: Vec::new(),
        ..m.clone()
    };
    let len_at = enc(&none).len();
    let body = |b: &[u8]| len_at + b[len_at..].iter().position(|&x| x & 0x80 == 0).unwrap() + 1;
    let (at, base_at) = (body(bytes), body(base_bytes));
    let i = bytes[at..]
        .iter()
        .zip(&base_bytes[base_at..])
        .position(|(a, b)| a != b)?;
    Some(at + i)
}
