//! **Run the real `nimsem` (sema) phase as a confined §14 child** — the front-end driver's
//! nested-spawn shape on Temen (NIM.md §3c, W5). `nim_frontend_driver` runs nimsem *top-level*; this
//! runs it as a child (through the [`temen_run::conductor`]), the way a Rust-on-Temen driver guest fans
//! phases out. The wrinkle vs nifler/hexer: nimsem is itself a driver — it `system("nifler … parse
//! <src> <out.p.nif>")`s to parse stdlib modules on demand, routed by the shim to an **`exec`** cap. So
//! the grant list carries **four** caps — `{fs, stdout, exit, exec}` — and the re-granted `exec` (a
//! `domain_exec_with_fs` over the *same* shared memfs) lets nimsem-the-child spawn `nifler`
//! grandchildren that write into the store nimsem reads. The emitted `.s.nif` is compared
//! (path-normalized) to native nimsem by the caller.
//!
//! ```text
//! cargo run -q --release -p temen-run --example nimsem_child_driver -- \
//!     <nimsem_ce.temen> <nifler.temen> <libdir> <sys.p.nif> <sys-stem> <out-dir>
//! ```

use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;

use temen_interp::{run_with_host_traced, ForkedProc, Host, HostProcFork, StreamRole, Value};
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

    // The child-entry nimsem module.
    let nimsem = temen_encode::decode_module(
        &std::fs::read(&nimsem_p).unwrap_or_else(|e| panic!("read {nimsem_p}: {e}")),
    )
    .expect("decode nimsem_ce.temen");
    temen_verify::verify_module(&nimsem).expect("nimsem verifies");
    let log2 = nimsem.memory.as_ref().expect("nimsem window").size_log2;
    let sys_pnif_key = format!("nimcache/{sys_stem}.p.nif");
    let parent = temen_run::conductor(
        log2,
        &["fs", "stdout", "exit", "exec"],
        &[
            "nimsem",
            "--define:nimNativeAlloc",
            "--define:nimNativeIo",
            "m",
            "--isSystem",
            &sys_pnif_key,
        ],
    );

    let mut host = Host::new();
    let (fs_init, fs_init_state) = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| {
            let (h, s) = (*f)();
            ForkedProc::shared(h, s)
        })
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork, fs_init_state);
    let sink = host.shared_stdout();
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    let exec_h = exec_cap.install(&mut host, 1u64 << temen_run::CONDUCTOR_LOG2);
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &nimsem);

    let mut fuel = 2_000_000_000_000u64;
    // `_traced`, not the plain `run_with_host`: the trap this driver exists to report belongs to the
    // **child**, whose window dies with its outcome, so `Err(t)` alone is a bare `MemoryFault`
    // with nothing left to ask (#1591). The backtrace it returns is the first-wins trap-origin
    // capture — the child's frames, not the parent's join site.
    let (r, trap_bt, _fiber) = run_with_host_traced(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(budget),
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
            // The child's module names the frames: they are its funcs, not the parent driver's. The
            // faulting address is worth as much as the trace — a small one says the pointer was
            // never initialized, a wild one says the arithmetic that produced it was wrong, and the
            // backtrace alone does not separate them.
            eprintln!(
                "{}",
                temen_run::with_backtrace(
                    format!("nimsem child trapped: {t:?}"),
                    &trap_bt,
                    temen_interp::last_capture_fault_addr(),
                    &nimsem,
                )
            );
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
