//! **The nimony front-end chain on the Cranelift JIT — every phase a confined §14 child** (NIM.md §3c,
//! W5). The exact twin of `nim_chain_op13`, but the native driver runs the [`temen_run::conductor`]
//! for each phase on **emitted code** (`compile_and_run_capture_reserved_with_host_ex` + the
//! granted-spawn hooks) rather than the tree-walker — the tier-up-capable engine a browser wasm-JIT
//! compile card also uses.
//!
//!   system.p.nif ─nimsem(JIT child)─▶ .s.nif ─hexer(JIT child)─▶ .x.nif (Leng)
//!                        │
//!                        └─ nifler grandchildren (via the re-granted `exec` cap) parse stdlib on demand
//!
//! nimsem gets a four-cap grant list `{fs, stdout, exit, exec}` (the re-granted `exec` lets it spawn its
//! nifler grandchildren over the same store); hexer gets `{fs, stdout, exit}` and reads the `.s.nif`
//! nimsem left in the store. The `.x.nif` is diffed (path-normalized) against native hexer by the caller.
//! Each phase runs in a detached window of its own, its malloc heap growing into the window's reserved
//! tail — no carve to size, so no carve to outgrow the JIT's window cap (#1591).
//!
//! ```text
//! cargo run -q --release -p temen-run --example nim_chain_op13_jit -- \
//!     <nimsem_ce.temen> <hexer_ce.temen> <nifler.temen> <libdir> <sys.p.nif> <sys-stem> <out-dir>
//! ```

use core::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;

use temen_interp::{ForkedProc, Host, HostProc, HostProcFork, StreamRole};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
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

/// The production granted-spawn hook table (temen-run's child build/bind/release/mint/thunk/serve) —
/// the same table `nifler_child_jit` / `rust_guest_op13` install to run a child on emitted code.
/// #1234 — the production table, derived from one [`temen_run::CapCtx`] so the hook family and
/// the parent pointer it decodes are chosen together (this used to hand-roll both, and nothing
/// checked that the pointer matched the ctx the run baked).
fn grant_hooks(host: *mut temen_interp::Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

/// Spawn one phase `module` through the [`temen_run::conductor`] **on the JIT**, with `argv` and the
/// named `caps` (each already granted in `host`, handles in `cap_handles`). Returns the joined status.
/// The conductor is the interp twin's — only the engine differs.
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
    let mut args = vec![inst as i64, modh as i64, budget as i64];
    args.extend(cap_handles.iter().map(|h| *h as i64));
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(host as *mut Host)),
    )
    .expect("jit run");
    match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        JitOutcome::Exited(c) => c as i64,
        ref o => {
            eprintln!("phase ended abnormally on the JIT: {o:?}");
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
        "usage: nim_chain_op13_jit <nimsem_ce.temen> <hexer_ce.temen> <nifler.temen> <libdir> <sys.p.nif> <sys-stem> <out-dir>",
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

    // ---- Phase 1: nimsem (a JIT child, exec re-granted) — semcheck the system module. ----------------
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
    eprintln!("nimsem (JIT child) joined: {s1}");
    assert!(
        handle
            .seed()
            .0
            .iter()
            .any(|(k, _)| k == &format!("nimcache/{sys}.s.nif")),
        "nimsem produced no .s.nif on the JIT"
    );

    // ---- Phase 2: hexer (a JIT child) — lower the .s.nif nimsem just wrote into the shared store. -----
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
    eprintln!("hexer (JIT child) joined: {s2}");

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
    eprintln!("JIT chain produced {wrote} file(s) → {out_dir} (incl {sys}.s.nif from nimsem, {sys}.x.nif from hexer)");
}
