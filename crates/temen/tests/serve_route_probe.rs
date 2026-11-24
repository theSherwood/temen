//! #2804 PROBE ONLY (not for merge): a serve handler's per-request cost on the JIT, run inline on the
//! serve op's stack (`invoke_handler`, what a durable or pool-less domain does today) against started
//! on a handler fiber (the one path). Run each route in its own process, interleaved:
//!
//!   cargo test --release -p temen --test serve_route_probe -- --ignored --nocapture
//!   TEMEN_PROBE_SERVE_INLINE=1 cargo test --release -p temen --test serve_route_probe -- --ignored --nocapture

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{Host, MemLayout};
use temen_jit::JitOutcome;

const N: i64 = 20_000;

fn caller() -> String {
    format!(
        r#"memory 17

func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {{
  vrm = i64.const 17416
  i32.store vrm v1
  vrb = i64.const 17420
  i32.store vrb v2
  vrp = i64.const 17408
  v5 = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  v6 = i64.const 0
  v7 = call.cap 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  vi0 = i64.const 0
  vacc0 = i64.const 0
  br 1(v0, v5, v7, vi0, vacc0)
}}
block 1 (w0: i32, w5: i32, w7: i32, wi: i64, wacc: i64) {{
  vone = i64.const 1
  vr = call.cap 268435456 0 (i64, i64) -> (i64) w7 (wi, vone)
  wacc1 = i64.add wacc vr
  wi1 = i64.add wi vone
  vn = i64.const {N}
  vmore = i64.lt_s wi1 vn
  br_if vmore 1(w0, w5, w7, wi1, wacc1) 2(w0, w5, wacc1)
}}
block 2 (x0: i32, x5: i32, xacc: i64) {{
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) x0 (x5)
  return xacc
  }}
}}
{rec}"#,
        rec = rec::segment(17408, &temen_ir::SpawnRec::default())
    )
}

fn server() -> String {
    format!(
        r#"memory 12
type 0 func (i64, i64) -> (i64)
type 1 interface {{ add: 0 }}
export 0 interface "adder" 1 {{ add: 1 }}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vt0 = i64.const 0
  br 1(vt0)
}}
block 1 (wt: i64) {{
  vz = i32.const 0
  vn = svc.wait vz
  wt1 = i64.add wt vn
  vk = i64.const {N}
  vless = i64.lt_s wt1 vk
  br_if vless 1(wt1) 2(wt1)
}}
block 2 (xt: i64) {{
  return xt
  }}
}}

func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  vs = i64.add va vb
  return vs
  }}
}}"#
    )
}

fn module(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

fn once() -> (i64, std::time::Duration) {
    let (a, b) = (module(&caller()), module(&server()));
    let mut host = Host::new();
    host.set_self_module(&std::sync::Arc::new(a.clone()));
    host.set_handoff(false);
    let args = [
        host.grant_instantiator(),
        host.grant_module(&b),
        host.grant_budget(-1, 1 << 20, -1),
        host.grant_clock(),
    ];
    let image = MemLayout::image(vec![0u8; 128 << 10]);
    let t = std::time::Instant::now();
    let (out, _) =
        temen_run::jit_cap_run(&a, 0, &args.map(i64::from), &image, 0, 0, &mut host, None)
            .expect("the JIT compiles the caller");
    let dt = t.elapsed();
    match out {
        JitOutcome::Returned(v) => (v[0], dt),
        other => panic!("{other:?}"),
    }
}

#[test]
#[ignore]
fn serve_route_per_request() {
    let route = if std::env::var_os("TEMEN_PROBE_SERVE_INLINE").is_some() {
        "inline"
    } else {
        "fiber"
    };
    let want = (0..N).map(|i| i + 1).sum::<i64>();
    let mut ns = Vec::new();
    for _ in 0..7 {
        let (sum, dt) = once();
        assert_eq!(sum, want);
        ns.push(dt.as_nanos() as f64 / N as f64);
    }
    ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "PROBE route={route} ns/req median {:.0} [{:.0}..{:.0}]",
        ns[3], ns[0], ns[6]
    );
}
