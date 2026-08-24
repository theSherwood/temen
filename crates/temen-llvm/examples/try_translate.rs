//! Scratch probe: translate a `.bc`/`.ll`, resolve capability imports, verify, run under the
//! powerbox, and (optionally) diff stdout against a native oracle binary. Used while closing
//! whole-program gaps (SQLite Phase A); not a test.
fn main() {
    let p = std::env::args()
        .nth(1)
        .expect("usage: try_translate <bc> [native-exe]");
    let native = std::env::args().nth(2);
    let t0 = std::time::Instant::now();
    // `.ll` → in-house textual reader; anything else → bitcode via `llvm-dis`.
    let is_ll = std::path::Path::new(&p)
        .extension()
        .is_some_and(|e| e == "ll");
    // `TEMEN_STUB_EXTERNS=1` mints trap stubs for genuinely-undefined externals instead of failing
    // closed at translate — the large-program bring-up mode (unreached OS calls compiled into a big
    // guest, e.g. Tcl's file/socket surface in the minimal no-Tcl_Init REPL).
    let stub = std::env::var("TEMEN_STUB_EXTERNS").is_ok_and(|v| v != "0" && !v.is_empty());
    let opts = temen_llvm::TranslateOptions {
        stub_unresolved_externs: stub,
        ..Default::default()
    };
    let translated = if is_ll {
        temen_llvm::translate_ll_path_with_options(&p, opts)
    } else {
        temen_llvm::translate_bc_path_with_options(&p, opts)
    };
    let t = match translated {
        Ok(t) => t,
        Err(e) => {
            println!("TRANSLATE ERR: {e:?}");
            std::process::exit(1);
        }
    };
    println!(
        "TRANSLATED in {:?}: {} funcs",
        t0.elapsed(),
        t.module.funcs.len()
    );
    // Phase 3: the manifest binds at instantiation — no rewrite.
    let exports = t.exports.clone();
    let module = t.module;
    if let Err(e) = temen_verify::verify_module(&module) {
        println!("VERIFY ERR: {e:?}");
        // Localize a `TypeMismatch { func: N, .. }` to the guest function name (via the export map),
        // for gap-walking a big program.
        let msg = format!("{e:?}");
        if let Some(rest) = msg.split("func:").nth(1) {
            if let Ok(idx) = rest
                .trim()
                .split([',', ' '])
                .next()
                .unwrap_or("")
                .parse::<u32>()
            {
                if let Some((name, _)) = exports.iter().find(|(_, i)| *i == idx) {
                    println!("  ↳ func {idx} = `{name}`");
                } else {
                    println!("  ↳ func {idx} = <synthesized/unexported>");
                }
            }
        }
        std::process::exit(1);
    }
    println!("VERIFIED");
    let t1 = std::time::Instant::now();
    // `TEMEN_STDIN=1` pipes the process's stdin to the guest (so a REPL guest can be driven).
    let stdin_bytes: Vec<u8> = if std::env::var_os("TEMEN_STDIN").is_some() {
        use std::io::Read;
        let mut b = Vec::new();
        std::io::stdin().read_to_end(&mut b).ok();
        b
    } else {
        Vec::new()
    };
    let run = match temen_run::run_powerbox(&module, &stdin_bytes) {
        Ok(r) => r,
        Err(e) => {
            println!("RUN ERR: {e}");
            std::process::exit(1);
        }
    };
    println!("RAN in {:?}: outcome {:?}", t1.elapsed(), run.outcome);
    if let Some(exe) = native {
        let out = std::process::Command::new(&exe)
            .output()
            .expect("run native");
        if run.stdout == out.stdout {
            println!("STDOUT MATCHES NATIVE ({} bytes)", out.stdout.len());
        } else {
            println!(
                "STDOUT MISMATCH: temen {} bytes vs native {} bytes",
                run.stdout.len(),
                out.stdout.len()
            );
            let sv = String::from_utf8_lossy(&run.stdout);
            let nv = String::from_utf8_lossy(&out.stdout);
            for (i, (a, b)) in sv.lines().zip(nv.lines()).enumerate() {
                if a != b {
                    println!("line {}: temen    {a:?}", i + 1);
                    println!("line {}: native {b:?}", i + 1);
                    break;
                }
            }
            std::process::exit(2);
        }
    } else {
        println!("--- stdout ---\n{}", String::from_utf8_lossy(&run.stdout));
    }
}
