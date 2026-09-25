//! #1817 — a capability park is keyed by **which domain's** handle it is parked through, not by the
//! bare handle number. Handle numbers are domain-local: a §14 child's table (or a fork twin's copy)
//! numbers its own capabilities, so the same number names different connections in different
//! domains. `cap_waiters` was keyed by the number alone, so a `Stream.close` in one domain completed
//! another domain's parked read with `CAP_REVOKED` although that domain's handle was still open.

use temen_interp::{run_with_host, Host, StreamRole, Value};

/// The §3.6 revocation completion status: `-EBADF`.
const CAP_REVOKED: i64 = -9;

/// Root `(ih, hin, hout)`: spawn thread 2, which sleeps 300 ms, raises a marker, and closes the
/// root's stdin; spawn a §14 child (func 1) granted the root's stdout as `"o"`; then block reading
/// stdin. The child resolves `"o"` — landing, in its own table, on the number the root's stdin has
/// — sleeps 100 ms (so the root is parked), closes it, and returns the number. Only the root's own
/// close may end the read, so the marker is up when it returns. Result:
/// `marker * 1_000_000 + child_number * 1_000 - read_status`.
const SRC: &str = r#"
memory 18
data 16684 "o"
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, vin: i32, vout: i32) {
  vin64 = i64.extend_i32_u vin
  vt = thread.spawn 2 vin64 vin64
  va0 = i64.const 16640
  vnp = i32.const 16684
  i32.store va0 vnp
  va1 = i64.const 16644
  vnl = i32.const 1
  i32.store va1 vnl
  va2 = i64.const 16648
  i32.store va2 vout
  vr0 = i64.const 4294967296
  vr8 = i64.const 131072
  vr16 = i64.const -4294967284
  vr24 = i64.const 4294967295
  vr32 = i64.const 0
  vr40 = i64.const 16640
  vr48 = i64.const 1
  vp0 = i64.const 17856
  i64.store vp0 vr0
  vp8 = i64.const 17864
  i64.store vp8 vr8
  vp16 = i64.const 17872
  i64.store vp16 vr16
  vp24 = i64.const 17880
  i64.store vp24 vr24
  vp32 = i64.const 17888
  i64.store vp32 vr32
  vp40 = i64.const 17896
  i64.store vp40 vr40
  vp48 = i64.const 17904
  i64.store vp48 vr48
  vc = call.cap 6 17 (i64) -> (i32) v0 (vp0)
  vbuf = i64.const 16392
  vcap = i64.const 4
  vr = call.cap 0 0 (i64, i64) -> (i64) vin (vbuf, vcap)
  vma = i64.const 16400
  vm = i64.load vma
  vtj = thread.join vt
  vk = call.cap 6 1 (i32) -> (i64) v0 (vc)
  vmil = i64.const 1000000
  vmm = i64.mul vm vmil
  vthou = i64.const 1000
  vkk = i64.mul vk vthou
  vs = i64.add vmm vkk
  vres = i64.sub vs vr
  return vres
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vo = i64.const 111
  vz = i64.const 0
  i64.store vz vo
  vl = i64.const 1
  vh = self.resolve vz vl
  vwa = i64.const 64
  vwe = i32.const 0
  vwt = i64.const 100000000
  vw = i32.atomic.wait vwa vwe vwt
  vcl = call.cap 0 2 () -> (i64) vh ()
  vh64 = i64.extend_i32_s vh
  return vh64
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vharg: i64) {
  vwa = i64.const 16384
  vwe = i32.const 0
  vwt = i64.const 300000000
  vw = i32.atomic.wait vwa vwe vwt
  vma = i64.const 16400
  vone = i64.const 1
  i64.store vma vone
  vh = i32.wrap_i64 vharg
  vc = call.cap 0 2 () -> (i64) vh ()
  return vc
  }
}
"#;

#[test]
fn a_close_in_one_domain_does_not_revoke_another_domains_parked_read() {
    let m = std::sync::Arc::new(temen_text::parse_module(SRC).expect("parse"));
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    host.set_self_module(&m);
    let ih = host.grant_instantiator(0, 1u64 << 18);
    let hout = host.grant_stream(StreamRole::Out);
    let hin = host.grant_stream(StreamRole::In);
    host.set_stdin_blocking(true);
    let mut fuel = u64::MAX;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(hin), Value::I32(hout)],
        &mut fuel,
        &mut host,
    )
    .expect("no trap");
    let [Value::I64(r)] = r[..] else {
        panic!("one i64: {r:?}")
    };
    let (marker, child_h, status) = (r / 1_000_000, r / 1_000 % 1_000, -(r % 1_000));
    assert_eq!(
        child_h, hin as i64,
        "fixture: the child's \"o\" must share the root stdin's number (root in={hin} out={hout}, r={r})"
    );
    assert_eq!(status, CAP_REVOKED, "the root's read ends revoked (r={r})");
    assert_eq!(
        marker, 1,
        "only the root's own close may end its read — the child's close of its own handle {child_h} reached it"
    );
}
