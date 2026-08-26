//! **Run the real `nimsem` (sema) phase as a confined §14 op-13 child** — the front-end driver's
//! nested-spawn shape on Temen (NIM.md §3c, W5). `nim_frontend_driver` runs nimsem *top-level*; this
//! runs it as an op-13 child, the way a Rust-on-Temen driver guest fans phases out. The wrinkle vs
//! nifler/hexer: nimsem is itself a driver — it `system("nifler … parse <src> <out.p.nif>")`s to parse
//! stdlib modules on demand, routed by the shim to an **`exec`** cap. So the op-13 grant list carries
//! **four** caps — `{fs, stdout, exit, exec}` — and the re-granted `exec` (a `domain_exec_with_fs` over
//! the *same* shared memfs) lets nimsem-the-child spawn `nifler` grandchildren that write into the store
//! nimsem reads. The emitted `.s.nif` is compared (path-normalized) to native nimsem by the caller.
//!
//! ```text
//! cargo run -q --release -p temen-run --example nimsem_child_driver -- \
//!     <nimsem_ce.temen> <nifler.temen> <libdir> <sys.p.nif> <sys-stem> <out-dir>
//! ```

use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, StreamRole, Value};
use temen_run::exec::{domain_exec_with_fs, DomainProgram};
use temen_run::{instantiate, HostCap, Limits};

fn collect(dir: &Path, prefix: &str, out: &mut Vec<(String, Vec<u8>)>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read_dir {d:?}: {e}")) {
            let e = e.expect("entry");
            let p = e.path();
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                stack.push(p);
            } else {
                let rel = p
                    .strip_prefix(dir)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((format!("{prefix}{rel}"), std::fs::read(&p).expect("read")));
            }
        }
    }
}

/// The op-13 parent: four grant records `{fs, stdout, exit, exec}` at 1024.., their names at 2048.., and
/// `argv` at `carve + POWERBOX_ARGS_BASE`. Spawns the child (window `child_sl`, carve at `carve_off`).
fn parent_src(child_sl: u32, carve_off: u64, argv: &[String]) -> String {
    let parent_sl = child_sl + 1;
    let argv_off = carve_off + temen_ir::POWERBOX_ARGS_BASE;
    let mut blob = Vec::new();
    blob.extend_from_slice(&(argv.len() as u32).to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    for s in argv {
        blob.extend_from_slice(s.as_bytes());
        blob.push(0);
    }
    let argv_esc: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    // record `i` at 1024+i*16: word0 = name_off | (name_len<<32) at that offset, handle (param v(2+i)) at +8.
    let names = [
        ("fs", 2u64, 2048u64),
        ("stdout", 6, 2064),
        ("exit", 4, 2080),
        ("exec", 4, 2096),
    ];
    let mut records = String::new();
    for (i, (_n, len, noff)) in names.iter().enumerate() {
        let off = 1024 + i as u64 * 16;
        let w0 = noff | (len << 32);
        records.push_str(&format!(
            "  x{off} = i64.const {w0}\n  o{off} = i64.const {off}\n  i64.store o{off} x{off}\n  h{off} = i64.extend_i32_u v{vi}\n  oh{off} = i64.const {hoff}\n  i64.store oh{off} h{off}\n",
            vi = 2 + i,
            hoff = off + 8,
        ));
    }
    format!(
        r#"memory {parent_sl}
data 2048 "fs"
data 2064 "stdout"
data 2080 "exit"
data 2096 "exec"
data {argv_off} "{argv_esc}"
func (i32, i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i32) {{
{records}  vmh = i64.extend_i32_u v1
  vgptr = i64.const 1024
  vgn = i64.const 4
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {child_sl}
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#,
    )
}

fn main() {
    let mut a = std::env::args().skip(1);
    let nimsem_p = a.next().expect(
        "usage: nimsem_child_driver <nimsem_ce.temen> <nifler.temen> <libdir> <sys.p.nif> <sys-stem> <out-dir>",
    );
    let nifler_p = a.next().expect("missing <nifler.temen>");
    let libdir = a.next().expect("missing <libdir>");
    let sys_pnif = a.next().expect("missing <sys.p.nif>");
    let sys_stem = a.next().expect("missing <sys-stem>");
    let out_dir = a.next().expect("missing <out-dir>");

    // Seed the shared memfs: stdlib under `lib/` (preserving `std/` and flattened), plus the parsed
    // system nif at `nimcache/<stem>.p.nif`.
    let mut files = vec![];
    collect(Path::new(&libdir), "lib/", &mut files);
    let flat: Vec<(String, Vec<u8>)> = files
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("lib/std/")
                .map(|r| (format!("lib/{r}"), v.clone()))
        })
        .collect();
    files.extend(flat);
    files.push((
        format!("nimcache/{sys_stem}.p.nif"),
        std::fs::read(&sys_pnif).unwrap_or_else(|e| panic!("read {sys_pnif}: {e}")),
    ));
    let seed_keys: std::collections::BTreeSet<String> =
        files.iter().map(|(k, _)| k.clone()).collect();
    eprintln!("seeded {} files into the shared memfs", files.len());

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(files, vec!["nimcache".into()]);
    let factory = Arc::new(factory);

    // The exec cap: nifler (top-level) spawnable by nimsem, over the SAME shared store.
    let nifler = std::fs::read(&nifler_p).unwrap_or_else(|e| panic!("read {nifler_p}: {e}"));
    let nifler_inst = Arc::new(
        instantiate(temen_encode::decode_module(&nifler).expect("decode nifler.temen"))
            .expect("inst nifler"),
    );
    let programs: Vec<DomainProgram> = ["nifler", "/bin/nifler"]
        .iter()
        .map(|n| DomainProgram {
            name: (*n).into(),
            instance: nifler_inst.clone(),
            limits: Limits::default(),
        })
        .collect();
    let child_fs = {
        let f = factory.clone();
        HostCap::host_proc(0, move || (f)())
    };
    let exec_cap = domain_exec_with_fs(programs, child_fs);

    // The child-entry nimsem module + carve geometry (generous heap — nimsem's system semcheck is large).
    let nimsem = temen_encode::decode_module(
        &std::fs::read(&nimsem_p).unwrap_or_else(|e| panic!("read {nimsem_p}: {e}")),
    )
    .expect("decode nimsem_ce.temen");
    temen_verify::verify_module(&nimsem).expect("nimsem verifies");
    let decl = nimsem.memory.as_ref().expect("nimsem window").size_log2 as u32;
    let child_sl = (decl + 3).max(28); // >= declared; nimsem's no-GC system semcheck peaks in (128, 256] MiB
    let carve_off = 1u64 << child_sl;
    let parent_win = 1u64 << (child_sl + 1);

    let argv: Vec<String> = [
        "nimsem",
        "--define:nimNativeAlloc",
        "--define:nimNativeIo",
        "m",
        "--isSystem",
        &format!("nimcache/{sys_stem}.p.nif"),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let parent =
        temen_text::parse_module(&parent_src(child_sl, carve_off, &argv)).expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");

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
    let exec_h = exec_cap.install(&mut host, parent_win);
    let inst = host.grant_instantiator(0, parent_win);
    let modh = host.grant_module(&nimsem);

    let mut fuel = 2_000_000_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(fs_h),
            Value::I32(stdout_h),
            Value::I32(exit_h),
            Value::I32(exec_h),
        ],
        &mut fuel,
        &mut host,
    );
    let stream = sink.lock().unwrap().clone();
    if !stream.is_empty() {
        eprintln!(
            "--- child stream ---\n{}\n---",
            String::from_utf8_lossy(&stream)
        );
    }
    let (produced, _) = handle.seed();
    let pnif = produced
        .iter()
        .filter(|(k, _)| !seed_keys.contains(k) && k.ends_with(".p.nif"))
        .count();
    // The `.p.nif` count is the nifler grandchildren the re-granted `exec` spawned (one per stdlib import).
    eprintln!("nifler grandchildren parsed {pnif} stdlib module(s) via the re-granted exec cap");
    match &r {
        Ok(v) => eprintln!("nimsem child joined: {v:?}"),
        Err(t) => {
            eprintln!("nimsem child trapped: {t:?}");
            exit(1);
        }
    }

    let mut wrote = 0usize;
    for (key, bytes) in produced {
        if seed_keys.contains(&key) {
            continue;
        }
        let dest = PathBuf::from(&out_dir).join(&key);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&dest, &bytes).expect("write output");
        wrote += 1;
    }
    eprintln!("nimsem child produced {wrote} file(s) → {out_dir}");
}
