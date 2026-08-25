//! **Drive a §14 child-entry phase (e.g. `nifler --child-entry`) via op-13 over a shared memfs.** The
//! guest half of the "nimony in the browser" driver: op-13-spawn a child-entry `.temen` with argv seeded
//! into its carve (`nifler p /in.nim /out.nif`), a shared memfs re-granted as `"fs"`, plus a `stdout`
//! Stream and an `exit` cap for its `write`/`read`/`exit` imports, then read the emitted `/out.nif` back
//! out of the shared store. The mechanism proven in `child_entry_argv_fs` (temen-llvm tests), now on a
//! real compiled phase.
//!
//! ```text
//! cargo run -q --release -p temen-run --example spawn_child_fs -- <child.temen> <input.nim> <out.nif>
//! ```
//!
//! The child is decoded and `verify_module`'d (NOT `instantiate`d — that gate wants a top-level paramless
//! `_start`; a child entry takes a starter and returns an i64 status, bound at op-13 spawn). Its imports
//! `exit`/`read`/`write`/`vm_map` bind by the reference policy to the re-granted Exit/Stream and the
//! auto-granted AddressSpace; `fs` resolves by name from the grant list. `scratch = 0` (the asset is
//! built without `--null-guard`), so argv seeds at `carve + POWERBOX_ARGS_BASE`.

use std::io::Write;
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value};

fn main() {
    let mut a = std::env::args().skip(1);
    let temen = a
        .next()
        .expect("usage: spawn_child_fs <child.temen> <input.nim> <out.nif>");
    let input = a.next().expect("missing <input.nim>");
    let out_name = a.next().expect("missing <out.nif>");
    let out_key = out_name.trim_start_matches('/').to_string();

    let src = std::fs::read(&input).unwrap_or_else(|e| panic!("read {input}: {e}"));
    let bytes = std::fs::read(&temen).unwrap_or_else(|e| panic!("read {temen}: {e}"));
    let child = temen_encode::decode_module(&bytes).expect("decode child .temen");
    temen_verify::verify_module(&child).expect("child verifies");

    // The child entry is the exported `_start` (func 0): a starter cap in, an i64 status out.
    let entry = 0u32;
    let decl = child.memory.as_ref().expect("child window").size_log2;
    // Carve the child a window at least its declared size, generously larger for heap room (its `malloc`
    // grows into `[heap_base, carve)` through the `vm_map` import). Parent window = twice the carve, the
    // carve in its upper half.
    let child_sl: u32 = (decl as u32 + 3).max(24); // >= declared, >= 16 MiB
    let carve_off: u64 = 1u64 << child_sl;
    let parent_sl: u32 = child_sl + 1;
    let argv_off = carve_off + temen_ir::POWERBOX_ARGS_BASE;

    // Seed argv `nifler p /in.nim /out.nif` (argc=4, envc=0) at the carve's POWERBOX_ARGS_BASE as a
    // NUL-separated blob, then three 16-byte grant records {name_off|name_len<<32, handle} for the caps
    // the child binds: `fs` (by name), `stdout` (Stream → write/read), `exit` (Exit). Names sit at 2048+.
    let in_arg = "/in.nim";
    let out_arg = format!("/{out_key}");
    let mut argv_blob = Vec::new();
    argv_blob.extend_from_slice(&4u32.to_le_bytes()); // argc
    argv_blob.extend_from_slice(&0u32.to_le_bytes()); // envc
    for s in ["nifler", "p", in_arg, &out_arg] {
        argv_blob.extend_from_slice(s.as_bytes());
        argv_blob.push(0);
    }
    let argv_esc: String = argv_blob.iter().map(|b| format!("\\x{b:02x}")).collect();

    let rec = |off: u64, name_off: u64, name_len: u64| -> String {
        let w0 = name_off | (name_len << 32);
        format!(
            "  x{off} = i64.const {w0}\n  o{off} = i64.const {off}\n  i64.store o{off} x{off}\n",
        )
    };
    // Records at 1024/1040/1056; their handle words at +8 come from params v2/v3/v4.
    let parent_src = format!(
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
  ventry = i64.const {entry}
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
    );
    let parent = temen_text::parse_module(&parent_src).expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");

    // A cross-domain shared memfs seeded with the source as `in.nim` (the guest's os_shim strips the
    // leading `/` of `/in.nim`). The handle observes the same store, so we read `out.nif` back after.
    let (factory, handle) =
        temen_run::fs::mem_fs_shared_factory(vec![("in.nim".to_string(), src)], vec![]);
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
    let inst = host.grant_instantiator(0, 1u64 << parent_sl);
    let modh = host.grant_module(&child);

    let mut fuel = 200_000_000_000u64;
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
    let stderr_bytes = sink.lock().unwrap().clone();
    if !stderr_bytes.is_empty() {
        eprintln!("--- child stdout/stderr ---");
        std::io::stderr().write_all(&stderr_bytes).unwrap();
        eprintln!("\n--- end ---");
    }
    match &r {
        Ok(v) => eprintln!("child joined: {v:?}"),
        Err(t) => {
            eprintln!("child trapped: {t:?}");
            std::process::exit(1);
        }
    }

    // Read the emitted `.nif` back out of the shared store and forward it to stdout (the caller diffs).
    let (files, _dirs) = handle.seed();
    match files.into_iter().find(|(name, _)| name == &out_key) {
        Some((_, emitted)) => {
            std::io::stdout().write_all(&emitted).unwrap();
            eprintln!("wrote {} bytes to {out_key}", emitted.len());
        }
        None => {
            eprintln!("child wrote no `{out_key}` into the shared memfs");
            std::process::exit(2);
        }
    }
}
