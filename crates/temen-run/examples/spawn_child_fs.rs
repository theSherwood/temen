//! **Drive a §14 child-entry phase (e.g. `nifler`/`hexer` `--child-entry`) via op-13 over a shared
//! memfs.** The guest half of the "nimony in the browser" driver: op-13-spawn a child-entry `.temen`
//! with argv seeded into its carve, a memfs seeded from a fixture dir and re-granted as `"fs"`, plus a
//! `stdout` Stream and an `exit` cap for its `write`/`read`/`exit` imports, then dump every file the
//! phase *wrote* to an output dir. The mechanism proven in `child_entry_argv_fs` / `rust_driver_nifler`
//! (temen-llvm tests), on a real compiled phase.
//!
//! ```text
//! cargo run -q --release -p temen-run --example spawn_child_fs -- \
//!     <child.temen> <fixture-dir> <out-dir> -- <argv0> <argv1> ...
//! ```
//!
//! Like `nimphase_run`, but the phase runs as a **confined op-13 child** (verify_module + spawn) rather
//! than a top-level powerbox program — so a Rust-on-Temen driver guest can fan phases out the same way.
//! The child's imports `exit`/`read`/`write`/`vm_map` bind by the reference policy to the re-granted
//! Exit/Stream and the auto-granted AddressSpace; `fs` resolves by name from the grant list. #964/#1094:
//! argv seeds at `carve + module_args_base(child)` — one guard up for a `__null_guard`-marked child (the
//! guarded nifler_ce), the legacy 128 otherwise; the grant records/cap-names stay in the parent window
//! (the op-13 handler reads them in the parent's context, so the child's guard never touches them).

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value};

/// Walk `dir` into `(relative-key, bytes)` memfs seed entries.
fn seed_dir(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out = vec![];
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read_dir {d:?}: {e}")) {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                stack.push(path);
            } else {
                let key = path
                    .strip_prefix(dir)
                    .expect("strip fixture prefix")
                    .to_string_lossy()
                    .into_owned();
                out.push((key, std::fs::read(&path).expect("read fixture file")));
            }
        }
    }
    out
}

/// Build the text-IR op-13 parent that spawns `child` (window `child_sl`, carve at `carve_off`) with the
/// three-entry grant list `{fs, stdout, exit}` and `argv` seeded at `carve + args_base`. #964/#1094:
/// `args_base` is the child's [`temen_ir::module_args_base`] — one guard up for a `__null_guard`-marked
/// child, the legacy 128 otherwise — since the child's `_start` reads argv there. The grant records and
/// cap-names stay at their parent-window offsets: the op-13 handler reads those in the *parent's*
/// context (`m.read_window`), so the child's guard never touches them.
fn parent_src(child_sl: u32, carve_off: u64, args_base: u64, argv: &[String]) -> String {
    let parent_sl = child_sl + 1;
    let argv_off = carve_off + args_base;
    let mut blob = Vec::new();
    blob.extend_from_slice(&(argv.len() as u32).to_le_bytes()); // argc
    blob.extend_from_slice(&0u32.to_le_bytes()); // envc
    for s in argv {
        blob.extend_from_slice(s.as_bytes());
        blob.push(0);
    }
    let argv_esc: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    let rec = |off: u64, name_off: u64, name_len: u64| -> String {
        let w0 = name_off | (name_len << 32);
        format!(
            "  x{off} = i64.const {w0}\n  o{off} = i64.const {off}\n  i64.store o{off} x{off}\n"
        )
    };
    format!(
        r#"memory {parent_sl}
data 2048 "fs"
data 2064 "stdout"
data 2080 "exit"
data {argv_off} "{argv_esc}"
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32) {{
{r0}  hf = i64.extend_i32_u v2
  ohf = i64.const 1032
  i64.store ohf hf
{r1}  hs = i64.extend_i32_u v3
  ohs = i64.const 1048
  i64.store ohs hs
{r2}  he = i64.extend_i32_u v4
  ohe = i64.const 1064
  i64.store ohe he
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 1024
  vgn = i64.const 3
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {child_sl}
  vq = i64.const 0
  vh = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = cap.call 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#,
        r0 = rec(1024, 2048, 2),
        r1 = rec(1040, 2064, 6),
        r2 = rec(1056, 2080, 4),
    )
}

fn main() {
    let mut a = std::env::args().skip(1);
    let temen = a
        .next()
        .expect("usage: spawn_child_fs <child.temen> <fixture-dir> <out-dir> -- <argv...>");
    let fixture = a.next().expect("missing <fixture-dir>");
    let out_dir = a.next().expect("missing <out-dir>");
    let argv: Vec<String> = a.skip_while(|s| s == "--").collect();
    assert!(!argv.is_empty(), "missing argv after --");

    let seed = seed_dir(Path::new(&fixture));
    let seed_keys: BTreeSet<String> = seed.iter().map(|(k, _)| k.clone()).collect();

    let bytes = std::fs::read(&temen).unwrap_or_else(|e| panic!("read {temen}: {e}"));
    let child = temen_encode::decode_module(&bytes).expect("decode child .temen");
    temen_verify::verify_module(&child).expect("child verifies");

    // Carve at least the declared window, generously larger for `malloc` heap room; the carve in the
    // parent window's upper half.
    let decl = child.memory.as_ref().expect("child window").size_log2 as u32;
    let child_sl = (decl + 3).max(24);
    let carve_off = 1u64 << child_sl;
    let parent =
        temen_text::parse_module(&parent_src(
            child_sl,
            carve_off,
            temen_ir::module_args_base(&child),
            &argv,
        ))
        .expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(seed, vec![]);
    let factory = Arc::new(factory);

    let mut host = Host::new();
    let fs_init: HostProc = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| ForkedProc::shared((*f)()))
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork);
    let sink = host.shared_stdout();
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    let inst = host.grant_instantiator(0, 1u64 << (child_sl + 1));
    let modh = host.grant_module(&child);

    let mut fuel = 400_000_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(fs_h),
            Value::I32(stdout_h),
            Value::I32(exit_h),
        ],
        &mut fuel,
        &mut host,
    );
    let stream = sink.lock().unwrap().clone();
    if !stream.is_empty() {
        eprintln!("--- child stdout/stderr ---");
        std::io::stderr().write_all(&stream).unwrap();
        eprintln!("\n--- end ---");
    }
    match &r {
        Ok(v) => eprintln!("child joined: {v:?}"),
        Err(t) => {
            eprintln!("child trapped: {t:?}");
            exit(1);
        }
    }

    // Every store key not in the seed is a phase output — dump it under out-dir.
    let (files, _dirs) = handle.seed();
    let mut wrote = 0usize;
    for (key, bytes) in files {
        if seed_keys.contains(&key) {
            continue;
        }
        let dest = PathBuf::from(&out_dir).join(&key);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).expect("mkdir out subdir");
        }
        std::fs::write(&dest, &bytes).expect("write phase output");
        wrote += 1;
    }
    eprintln!("phase produced {wrote} file(s) → {out_dir}");
}
