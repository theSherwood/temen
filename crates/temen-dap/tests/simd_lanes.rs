//! `__m128` lane inspection over the DAP. chibicc emits `__m128`'s debug type as a 4-element
//! `float` array (its f32x4 layout), so the existing aggregate-expansion path renders a
//! `__m128` variable as four per-lane `float` children — no v128-specific consumer code. This
//! compiles a real `#include <xmmintrin.h>` program with `-g`, stops with the vector live, and
//! asserts the four lanes read back in order.
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain, like the frontend suites).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use temen_dap::{DapServer, Json};

mod support;
use support::{req, response};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn chibicc() -> &'static Path {
    static CC: OnceLock<PathBuf> = OnceLock::new();
    CC.get_or_init(|| {
        let dir = repo_root().join("frontend/chibicc");
        let status = Command::new("make")
            .arg("-s")
            .current_dir(&dir)
            .status()
            .expect("run `make` to build the chibicc fork");
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}

/// Compile C to `-g` text IR (attached `-I` — cc1 only accepts `-Ipath`).
fn c_to_ir(src: &str) -> String {
    let base = std::env::temp_dir().join(format!("temen_simdlane_{}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("temen");
    std::fs::write(&cfile, src).unwrap();
    let inc = repo_root().join("browser/playground-include");
    let ok = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
            "-g",
            &format!("-I{}", inc.to_str().unwrap()),
            "-cc1-input",
            cfile.to_str().unwrap(),
            "-cc1-output",
            irfile.to_str().unwrap(),
            cfile.to_str().unwrap(),
        ])
        .status()
        .expect("run chibicc")
        .success();
    assert!(ok, "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

fn body_array<'a>(resp: &'a Json, key: &str) -> &'a [Json] {
    resp.get("body")
        .and_then(|b| b.get(key))
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("body.{key} array"))
}

/// The lanes read back as `float`s in lane order (`_mm_set_ps(e3,e2,e1,e0)` puts e0 in lane 0),
/// each as its own expandable-array child of the `__m128` variable.
#[test]
fn m128_variable_expands_to_four_float_lanes() {
    let ir = c_to_ir(
        r#"#include <xmmintrin.h>
int main() {
  __m128 v = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
  float out[4];
  _mm_storeu_ps(out, v);
  return (int)(out[0] + out[1] + out[2] + out[3]);
}"#,
    );

    // chibicc records the source path in `debug.file 0 "<path>"`; break against that exact string.
    let src_path = ir
        .lines()
        .find_map(|l| l.strip_prefix("debug.file 0 \""))
        .and_then(|r| r.strip_suffix('"'))
        .expect("debug.file present")
        .to_string();

    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(&ir)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    // Break on the `_mm_storeu_ps` line (source line 5), where `v` is fully constructed and live.
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s(&src_path))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(5))])]),
            ),
        ]),
    ));
    let cont = s.handle(&req(4, "continue", Json::obj(vec![])));
    assert!(
        cont.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("stopped")),
        "stopped at the breakpoint"
    );

    // stackTrace → scopes → the Locals variablesReference.
    let st = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frame_id = body_array(response(&st), "stackFrames")[0]
        .get("id")
        .unwrap()
        .as_i64()
        .unwrap();
    let sc = s.handle(&req(
        6,
        "scopes",
        Json::obj(vec![("frameId", Json::i(frame_id))]),
    ));
    let scope_ref = body_array(response(&sc), "scopes")[0]
        .get("variablesReference")
        .unwrap()
        .as_i64()
        .unwrap();

    // The `v` local is an expandable `__m128` (a 4-float array), so it carries a nonzero reference.
    let vars = s.handle(&req(
        7,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(scope_ref))]),
    ));
    let v = body_array(response(&vars), "variables")
        .iter()
        .find(|x| x.get("name").and_then(|n| n.as_str()) == Some("v"))
        .expect("local `v` present");
    assert_eq!(
        v.get("type").and_then(|t| t.as_str()),
        Some("__m128"),
        "render name stays __m128"
    );
    let v_ref = v.get("variablesReference").unwrap().as_i64().unwrap();
    assert!(v_ref != 0, "__m128 is expandable (has lane children)");

    // Expand → four lanes `[0]..[3]` reading 1.0, 2.0, 3.0, 4.0.
    let lanes_resp = s.handle(&req(
        8,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(v_ref))]),
    ));
    let lanes = body_array(response(&lanes_resp), "variables");
    assert_eq!(lanes.len(), 4, "four lanes");
    let read = |i: usize| -> f64 {
        lanes[i]
            .get("value")
            .and_then(|v| v.as_str())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or_else(|| panic!("lane {i} value is a float, got {:?}", lanes[i].get("value")))
    };
    for (i, want) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
        assert_eq!(
            lanes[i].get("name").and_then(|n| n.as_str()),
            Some(&*format!("[{i}]"))
        );
        assert!(
            (read(i) - want).abs() < 1e-6,
            "lane {i} = {} (want {want})",
            read(i)
        );
    }
}
