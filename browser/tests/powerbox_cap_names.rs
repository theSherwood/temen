//! THREADS/BROWSER parity (PR #118, F7/F9): the wasm **powerbox** must register its granted
//! capabilities under their canonical names in the `cap_names` directory, so a guest can resolve them
//! at runtime with `self.resolve` — exactly as `temen-run`'s powerbox does. Without the registration,
//! `self.resolve("stdout")` would `-EINVAL` and the guest couldn't re-find its own handles, a
//! silent divergence from the native powerbox ground truth `powerbox_exec` is meant to mirror verbatim.

use temen_browser::{powerbox_exec, STATUS_OK};
use temen_text::parse_module;

// Resolve "stdout" → its handle at runtime (never reading the param slot), then write through the
// resolved handle. Works only if the powerbox registered the canonical name.
const RESOLVE_STDOUT: &str = r#"
memory 16
data 17008 "hi from resolve\n"
data 17000 "stdout"
func (i32, i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32, v2: i32) {
  v3 = i64.const 17000
  v4 = i64.const 6
  v5 = self.resolve v3 v4
  v6 = i64.const 17008
  v7 = i64.const 16
  v8 = call.cap 0 1 (i64, i64) -> (i64) v5(v6, v7)
  v9 = i32.const 0
  return v9
  }
}
"#;

#[test]
fn powerbox_registers_canonical_cap_names_for_resolve() {
    let m = parse_module(RESOLVE_STDOUT).expect("parse");
    let out = powerbox_exec(&m, &[]);
    assert_eq!(
        out.status, STATUS_OK,
        "guest should run cleanly (resolve found the stdout handle), got status {}",
        out.status
    );
    assert_eq!(
        out.stdout, b"hi from resolve\n",
        "the name-resolved stdout handle must be the real, working capability"
    );
}
