//! **The nimony front-end chain, every phase a confined §14 child, over ONE shared memfs** (NIM.md
//! §3c, W5 — the compiler driver on Temen). A native driver (the small trusted core) spawns `nimsem`
//! then `hexer` through the [`temen_run::conductor`] (each in a detached window of its own), handing the
//! semchecked `.s.nif` from one to the next through a single shared `mem_fs` — the file-based nifmake
//! hand-off, but with the phases *confined* rather than run top-level:
//!
//!   system.p.nif ─nimsem(child)─▶ .s.nif ─hexer(child)─▶ .x.nif (Leng)
//!                        │
//!                        └─ nifler grandchildren (via the re-granted `exec` cap) parse stdlib on demand
//!
//! nimsem gets a four-cap grant list `{fs, stdout, exit, exec}` (the re-granted `exec` lets it spawn its
//! nifler grandchildren over the same store); hexer gets `{fs, stdout, exit}` and reads the `.s.nif`
//! nimsem left in the store. Both `.x.nif` and `.dce.nif` are diffed (path-normalized) against native
//! hexer by the caller. The driver stays host-side Rust — the phases are the sandboxed guests. (The
//! file keeps its `op13` name because `build_frontend.sh` invokes it by that name.)
//!
//! ```text
//! cargo run -q --release -p temen-run --example nim_chain_op13 -- \
//!     <nimsem_ce.temen> <hexer_ce.temen> <nifler.temen> <libdir> <sys.p.nif> <sys-stem> <out-dir>
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

/// Spawn one phase `module` through the [`temen_run::conductor`] with `argv` and the named `caps`
/// (each already granted in `host`, handles in `cap_handles`). Returns the joined status.
fn spawn_phase(
    host: &mut Host,
    module: &temen_ir::Module,
    argv: &[&str],
    caps: &[&str],
    cap_handles: &[i32],
) -> i64 {
    let log2 = module.memory.as_ref().expect("phase window").size_log2;
    let parent = temen_run::conductor(log2, caps, argv);
    let (inst, modh, budget) = temen_run::grant_conductor(host, module);
    let mut args = vec![Value::I32(inst), Value::I32(modh), Value::I32(budget)];
    args.extend(cap_handles.iter().map(|h| Value::I32(*h)));
    let mut fuel = 4_000_000_000_000u64;
    match run_with_host(&parent, 0, &args, &mut fuel, host) {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => *x,
            [Value::I32(x)] => *x as i64,
            _ => panic!("phase result: {v:?}"),
        },
        Err(t) => {
            eprintln!("phase trapped: {t:?}");
            exit(1);
        }
    }
}

fn grant_fs(
    host: &mut Host,
    factory: &Arc<impl Fn() -> (HostProc, temen_interp::CapState) + Send + Sync + 'static>,
) -> i32 {
    let (init, state) = (*factory)();
    let f = Arc::clone(factory);
    let fork: HostProcFork = Arc::new(move |_pid| {
        let (h, s) = (*f)();
        ForkedProc::shared(h, s)
    });
    host.grant_host_proc_forkable(init, fork, state)
}

fn main() {
    let mut a = std::env::args().skip(1);
    let nimsem_p = a.next().expect(
        "usage: nim_chain_op13 <nimsem_ce.temen> <hexer_ce.temen> <nifler.temen> <libdir> <sys.p.nif> <sys-stem> <out-dir>",
    );
    let hexer_p = a.next().expect("missing <hexer_ce.temen>");
    let nifler_p = a.next().expect("missing <nifler.temen>");
    let libdir = a.next().expect("missing <libdir>");
    let sys_pnif = a.next().expect("missing <sys.p.nif>");
    let sys = a.next().expect("missing <sys-stem>");
    let out_dir = a.next().expect("missing <out-dir>");

    // One shared memfs for the whole chain: stdlib + the parsed system nif.
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
        format!("nimcache/{sys}.p.nif"),
        std::fs::read(&sys_pnif).unwrap_or_else(|e| panic!("read {sys_pnif}: {e}")),
    ));
    let seed_keys: std::collections::BTreeSet<String> =
        files.iter().map(|(k, _)| k.clone()).collect();
    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(files, vec!["nimcache".into()]);
    let factory = Arc::new(factory);

    let load = |p: &str| {
        temen_encode::decode_module(&std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}")))
            .expect("decode")
    };
    let nimsem = load(&nimsem_p);
    let hexer = load(&hexer_p);
    temen_verify::verify_module(&nimsem).expect("nimsem verifies");
    temen_verify::verify_module(&hexer).expect("hexer verifies");

    // The exec cap for nimsem: nifler (top-level) over the SAME shared store.
    let nifler_inst = Arc::new(instantiate(load(&nifler_p)).expect("inst nifler"));
    let programs: Vec<DomainProgram> = ["nifler", "/bin/nifler"]
        .iter()
        .map(|n| DomainProgram {
            name: (*n).into(),
            instance: nifler_inst.clone(),
            limits: Limits::default(),
        })
        .collect();

    // ---- Phase 1: nimsem (a child, exec re-granted) — semcheck the system module. ---------------------
    let mut h1 = Host::new();
    let fs1 = grant_fs(&mut h1, &factory);
    let out1 = h1.grant_stream(StreamRole::Out);
    let ex1 = h1.grant_exit();
    let child_fs = {
        let f = factory.clone();
        HostCap::host_proc(0, move || (f)())
    };
    let exec1 =
        domain_exec_with_fs(programs, child_fs).install(&mut h1, 1u64 << temen_run::CONDUCTOR_LOG2);
    let sys_pnif_key = format!("nimcache/{sys}.p.nif");
    let s1 = spawn_phase(
        &mut h1,
        &nimsem,
        &[
            "nimsem",
            "--define:nimNativeAlloc",
            "--define:nimNativeIo",
            "m",
            "--isSystem",
            &sys_pnif_key,
        ],
        &["fs", "stdout", "exit", "exec"],
        &[fs1, out1, ex1, exec1],
    );
    eprintln!("nimsem (child) joined: {s1}");
    assert!(
        handle
            .seed()
            .0
            .iter()
            .any(|(k, _)| k == &format!("nimcache/{sys}.s.nif")),
        "nimsem produced no .s.nif"
    );

    // ---- Phase 2: hexer (a child) — lower the .s.nif nimsem just wrote into the shared store. ----------
    let mut h2 = Host::new();
    let fs2 = grant_fs(&mut h2, &factory);
    let out2 = h2.grant_stream(StreamRole::Out);
    let ex2 = h2.grant_exit();
    let sys_snif_key = format!("nimcache/{sys}.s.nif");
    let s2 = spawn_phase(
        &mut h2,
        &hexer,
        &["hexer", "c", &sys_snif_key],
        &["fs", "stdout", "exit"],
        &[fs2, out2, ex2],
    );
    eprintln!("hexer (child) joined: {s2}");

    // Dump the phase outputs (everything not seeded).
    let (produced, _) = handle.seed();
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
    eprintln!("chain produced {wrote} file(s) → {out_dir} (incl {sys}.s.nif from nimsem, {sys}.x.nif from hexer)");
}
