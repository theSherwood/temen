//! #1297 — **fork carries live §22 guest-JIT state.** A domain that has compiled *and installed* a
//! unit forks (`clone_caller`, the FORK.md §8 twin); both copies `call.dyn` the slot installed before
//! the fork (the twin's dispatch table is a snapshot of the original's, its units `Arc`-shared), and
//! each then `install`s again into its **own** table — landing at the same next slot on both sides,
//! which is only possible if the tables are separate (a shared table would hand out consecutive
//! slots). Previously `Host::fork_powerbox` failed closed on any live JIT state ("per-image artifacts
//! the core cannot duplicate") — a classification snapshot had already outgrown (DURABILITY.md
//! §12.5 carries exactly this state). Tree-walker ≡ bytecode.
//!
//! Topology = `clone_caller.rs`'s `SRC_TWIN` (root spawns a serving domain S whose `fork` handler
//! `clone_caller(100, 200)`s and whose `wait` handler reaps; then the caller C with `"svc"`, `"o"` and
//! — new — `"jit"` re-granted by name). C: resolve the three, stage the unit, compile + install
//! (slot 5: C has five functions and a 16-slot table from the root's `Jit` reservation), fork; the
//! original (reply 100) reaps the twin (reply 200); both write `call.dyn(slot 5) * 1000 + install2`
//! (= 42 · 1000 + 6) to the shared stdout and return their reply (the run's value is the original's).

use temen_encode::encode_module;
use temen_interp::{bytecode, run_with_host, Host, StreamRole, Value};
use temen_run::grant_jit;
use temen_text::parse_module;
use temen_verify::verify_module;

/// Where C stages the unit blob in its own 4-KiB window.
const BLOB_OFF: i64 = 1024;

/// The unit: `() -> 42`, declaring the program's memory (`memory 18` — C is a same-module child, so
/// its re-granted table's precondition is the root's).
fn blob() -> Vec<u8> {
    let m = parse_module(
        "memory 18\nfunc () -> (i32) {\nblock 0 () {\n  v0 = i32.const 42\n  return v0\n  }\n}\n",
    )
    .expect("parse unit");
    verify_module(&m).expect("verify unit");
    encode_module(&m)
}

fn src() -> String {
    let b = blob();
    let mut stores = String::new();
    for (i, chunk) in b.chunks(8).enumerate() {
        let mut w = [0u8; 8];
        w[..chunk.len()].copy_from_slice(chunk);
        stores.push_str(&format!(
            "  bp{i} = i64.const {}\n  bw{i} = i64.const {}\n  i64.store bp{i} bw{i}\n",
            BLOB_OFF + (i as i64) * 8,
            i64::from_le_bytes(w),
        ));
    }
    format!(
        r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface {{ fork: 0, wait: 0 }}
export 0 interface "svc" 1 {{ fork: 2, wait: 3 }}
data 16684 "svc"
data 16694 "o"
data 16704 "jit"
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, vout: i32, vjit: i32) {{
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 131072
  q1v2 = i64.const -4294967284
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1a0 = i64.const 17600
  i64.store q1a0 q1v0
  q1a1 = i64.const 17608
  i64.store q1a1 q1v1
  q1a2 = i64.const 17616
  i64.store q1a2 q1v2
  q1a3 = i64.const 17624
  i64.store q1a3 q1v3
  q1a4 = i64.const 17632
  i64.store q1a4 q1v4
  q1a5 = i64.const 17640
  i64.store q1a5 q1v4
  q1a6 = i64.const 17648
  i64.store q1a6 q1v4
  vs = call.cap 6 17 (i64) -> (i32) v0 (q1a0)
  vz0 = i64.const 0
  vcap = call.cap 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 16640
  vnp = i32.const 16684
  i32.store va0 vnp
  va1 = i64.const 16644
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 16648
  i32.store va2 vcap
  va3 = i64.const 16656
  vnp2 = i32.const 16694
  i32.store va3 vnp2
  va4 = i64.const 16660
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 16664
  i32.store va5 vout
  va6 = i64.const 16672
  vnp3 = i32.const 16704
  i32.store va6 vnp3
  va7 = i64.const 16676
  i32.store va7 vnl
  va8 = i64.const 16680
  i32.store va8 vjit
  q2v0 = i64.const 17179869184
  q2v1 = i64.const 135168
  q2v2 = i64.const -4294967284
  q2v3 = i64.const 4294967295
  q2v4 = i64.const 0
  q2v5 = i64.const 16640
  q2v6 = i64.const 3
  q2a0 = i64.const 17664
  i64.store q2a0 q2v0
  q2a1 = i64.const 17672
  i64.store q2a1 q2v1
  q2a2 = i64.const 17680
  i64.store q2a2 q2v2
  q2a3 = i64.const 17688
  i64.store q2a3 q2v3
  q2a4 = i64.const 17696
  i64.store q2a4 q2v4
  q2a5 = i64.const 17704
  i64.store q2a5 q2v5
  q2a6 = i64.const 17712
  i64.store q2a6 q2v6
  vc = call.cap 6 17 (i64) -> (i32) v0 (q2a0)
  vjc = call.cap 6 1 (i32) -> (i64) v0 (vc)
  return vjc
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  br 1()
  }}
block 1 () {{
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  br 1()
  }}
}}
func (i64) -> (i64) {{
block 0 (vx: i64) {{
  vz = i32.const 0
  vro = i64.const 100
  vrt = i64.const 200
  vt = call.cap 4294967295 11 (i64, i64) -> (i64) vz (vro, vrt)
  return vt
  }}
}}
func (i64) -> (i64) {{
block 0 (vpid: i64) {{
  vz = i32.const 0
  vt = call.cap 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vsvc = i64.const 6518387
  vzero = i64.const 0
  i64.store vzero vsvc
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  vjname = i64.const 7629162
  va24 = i64.const 24
  i64.store va24 vjname
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = self.resolve vp0 vl3
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = self.resolve vp8 vl1
  vhj = self.resolve va24 vl3
{stores}  vbp = i64.const {BLOB_OFF}
  vbl = i64.const {}
  vc = call.cap 11 0 (i64, i64) -> (i64) vhj (vbp, vbl)
  vs0 = call.cap 11 3 (i64) -> (i64) vhj (vc)
  br 1(vhsvc, vho, vhj, vc, vs0)
  }}
block 1 (vhsvc: i32, vho: i32, vhj: i32, vc: i64, vs0: i64) {{
  varg = i64.const 7
  vr = call.cap 268435456 0 (i64) -> (i64) vhsvc (varg)
  v200 = i64.const 200
  vistwin = i64.eq vr v200
  br_if vistwin 4(vr, vho, vhj, vc, vs0) 2(vr, vhsvc, vho, vhj, vc, vs0)
  }}
block 2 (vr: i64, vhsvc: i32, vho: i32, vhj: i32, vc: i64, vs0: i64) {{
  vpid3 = i64.const 3
  vstatus = call.cap 268435456 1 (i64) -> (i64) vhsvc (vpid3)
  veagain = i64.const -11
  viseagain = i64.eq vstatus veagain
  br_if viseagain 2(vr, vhsvc, vho, vhj, vc, vs0) 3(vr, vstatus, vhsvc, vho, vhj, vc, vs0)
  }}
block 3 (vr: i64, vstatus: i64, vhsvc: i32, vho: i32, vhj: i32, vc: i64, vs0: i64) {{
  vechild = i64.const -10
  visechild = i64.eq vstatus vechild
  br_if visechild 1(vhsvc, vho, vhj, vc, vs0) 4(vr, vho, vhj, vc, vs0)
  }}
block 4 (vr: i64, vho: i32, vhj: i32, vc: i64, vs0: i64) {{
  vs0i = i32.wrap_i64 vs0
  vd = call.dyn () -> (i32) vs0i ()
  vs1 = call.cap 11 3 (i64) -> (i64) vhj (vc)
  vd64 = i64.extend_i32_s vd
  vk = i64.const 1000
  vm = i64.mul vd64 vk
  vval = i64.add vm vs1
  vp16 = i64.const 16
  i64.store vp16 vval
  vlen = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  return vr
  }}
}}
"#,
        b.len()
    )
}

/// Run on the tree-walker (`bytecode = false`) or the bytecode engine; returns the run's value and
/// the shared stdout sink's i64 words, sorted (twin/original write order is scheduler-dependent).
fn run(bytecode: bool) -> (Vec<Value>, Vec<i64>) {
    let m = parse_module(&src()).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    host.set_self_module(&std::sync::Arc::new(m.clone()));
    let ih = host.grant_instantiator(0, 1u64 << 18);
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let jh = grant_jit(&mut host, &m, 4); // 16 install slots, carried into C by the grant
    let args = [Value::I32(ih), Value::I32(out_h), Value::I32(jh)];
    let mut fuel = 40_000_000u64;
    let r = if bytecode {
        bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
            .expect("the fork module runs natively on the bytecode engine")
            .expect("bytecode run")
    } else {
        run_with_host(&m, 0, &args, &mut fuel, &mut host).expect("oracle run")
    };
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mut words: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    words.sort();
    (r, words)
}

#[test]
fn a_domain_with_live_jit_state_forks_and_both_copies_keep_their_own_table() {
    let (tw, tw_words) = run(false);
    assert_eq!(
        tw,
        vec![Value::I64(100)],
        "tree-walker: the original resumes with reply_orig — the fork was admitted"
    );
    // Both copies: the pre-fork install (slot 5) dispatches to the unit (42), and the post-fork
    // install lands at slot 6 in EACH copy's own table.
    assert_eq!(
        tw_words,
        vec![42_006, 42_006],
        "tree-walker: original + twin both call.dyn the inherited slot and install into their own table"
    );
    let (bc, bc_words) = run(true);
    assert_eq!(bc, tw, "bytecode engine agrees on the run value");
    assert_eq!(
        bc_words, tw_words,
        "bytecode engine agrees on both copies' writes"
    );
}
