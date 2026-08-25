//! **The nimony front-end chain on the Cranelift JIT — every phase a confined §14 op-13 child**
//! (NIM.md §3c, W5). The exact twin of `nim_chain_op13`, but the native conductor spawns each phase
//! on **emitted code** (`compile_and_run_capture_reserved_with_host_ex` + the granted-spawn hooks)
//! rather than the tree-walker — the tier-up-capable engine a browser wasm-JIT compile card also uses.
//!
//!   system.p.nif ─nimsem(op-13 JIT child)─▶ .s.nif ─hexer(op-13 JIT child)─▶ .x.nif (Leng)
//!                        │
//!                        └─ nifler grandchildren (via the re-granted `exec` cap) parse stdlib on demand
//!
//! nimsem gets a four-cap grant list `{fs, stdout, exit, exec}` (the re-granted `exec` lets it spawn its
//! nifler grandchildren over the same store); hexer gets `{fs, stdout, exit}` and reads the `.s.nif`
//! nimsem left in the store. The `.x.nif` is diffed (path-normalized) against native hexer by the caller.
//! The op-13 `mod_ok` relaxation (`declared <= carve`, matching the interpreter — FORK.md §8.6 / #773)
//! is what lets these malloc-heavy phases carve the heap room they need on the JIT.
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

/// The op-13 parent for a phase spawn: `caps` grant records at 1024.. (names at 2048..), `argv` at
/// `carve + POWERBOX_ARGS_BASE`. Params: `(inst, module, cap0, cap1, …)`. Identical to the interp twin.
fn parent_src(child_sl: u32, carve_off: u64, argv: &[String], caps: &[&str]) -> String {
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
    let n = caps.len() + 2;
    let sig: String = vec!["i32"; n].join(", ");
    let bparams: String = (0..n)
        .map(|i| format!("v{i}: i32"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut data = String::new();
    let mut records = String::new();
    for (i, name) in caps.iter().enumerate() {
        let noff = 2048 + i as u64 * 16;
        data.push_str(&format!("data {noff} \"{name}\"\n"));
        let off = 1024 + i as u64 * 16;
        let w0 = noff | ((name.len() as u64) << 32);
        records.push_str(&format!(
            "  x{off} = i64.const {w0}\n  o{off} = i64.const {off}\n  i64.store o{off} x{off}\n  h{off} = i64.extend_i32_u v{vi}\n  oh{off} = i64.const {hoff}\n  i64.store oh{off} h{off}\n",
            vi = 2 + i,
            hoff = off + 8,
        ));
    }
    let gn = caps.len();
    format!(
        r#"memory {parent_sl}
{data}data {argv_off} "{argv_esc}"
func ({sig}) -> (i64) {{
block 0 ({bparams}) {{
{records}  vmh = i64.extend_i32_u v1
  vgptr = i64.const 1024
  vgn = i64.const {gn}
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
    )
}

/// The production granted-spawn hook table (temen-run's child build/bind/release/mint/thunk/serve) —
/// the same table `nifler_child_jit` / `rust_guest_op13` install to run an op-13 child on emitted code.
fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: temen_run::grant_child_build,
        build_named: temen_run::grant_named_child_build,
        bind_imports: temen_run::child_bind_imports,
        release: temen_run::grant_child_release,
        mint: temen_run::child_offer_mint,
        thunk: temen_run::cap_thunk_locked,
        register_serve: temen_run::child_register_serve,
    }
}

/// op-13-spawn one phase `module` with `argv` and the named `caps` **on the JIT** (each cap already
/// granted in `host`, handles in `cap_handles`), into a `carve_sl`-sized carve. Returns the joined
/// status. The parent text-IR is identical to the interp twin — only the engine differs.
#[allow(clippy::too_many_arguments)]
fn spawn_phase(
    host: &mut Host,
    module: &temen_ir::Module,
    argv: &[String],
    caps: &[&str],
    cap_handles: &[i32],
    carve_sl: u32,
) -> i64 {
    let carve_off = 1u64 << carve_sl;
    let parent = temen_text::parse_module(&parent_src(carve_sl, carve_off, argv, caps))
        .expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");
    let inst = host.grant_instantiator(0, 1u64 << (carve_sl + 1));
    let modh = host.grant_module(module);
    let mut args = vec![inst as i64, modh as i64];
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
        Some(grant_hooks()),
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

fn grant_fs(host: &mut Host, factory: &Arc<impl Fn() -> HostProc + Send + Sync + 'static>) -> i32 {
    let init: HostProc = (*factory)();
    let f = Arc::clone(factory);
    let fork: HostProcFork = Arc::new(move |_pid| ForkedProc::shared((*f)()));
    host.grant_host_proc_forkable(init, fork)
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

    // ---- Phase 1: nimsem (op-13 JIT child, exec re-granted) — semcheck the system module. -------------
    let nimsem_carve = (nimsem.memory.unwrap().size_log2 as u32 + 3).max(28); // 256 MiB (no-GC peak)
    let win1 = 1u64 << (nimsem_carve + 1);
    let mut h1 = Host::new();
    let fs1 = grant_fs(&mut h1, &factory);
    let out1 = h1.grant_stream(StreamRole::Out);
    let ex1 = h1.grant_exit();
    let child_fs = {
        let f = factory.clone();
        HostCap::host_proc(0, move || (f)())
    };
    let exec1 = domain_exec_with_fs(programs, child_fs).install(&mut h1, win1);
    let argv1: Vec<String> = [
        "nimsem",
        "--define:nimNativeAlloc",
        "--define:nimNativeIo",
        "m",
        "--isSystem",
        &format!("nimcache/{sys}.p.nif"),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let s1 = spawn_phase(
        &mut h1,
        &nimsem,
        &argv1,
        &["fs", "stdout", "exit", "exec"],
        &[fs1, out1, ex1, exec1],
        nimsem_carve,
    );
    eprintln!("nimsem (op-13 JIT child) joined: {s1}");
    assert!(
        handle
            .seed()
            .0
            .iter()
            .any(|(k, _)| k == &format!("nimcache/{sys}.s.nif")),
        "nimsem produced no .s.nif on the JIT"
    );

    // ---- Phase 2: hexer (op-13 JIT child) — lower the .s.nif nimsem just wrote into the shared store. --
    let hexer_carve = (hexer.memory.unwrap().size_log2 as u32 + 3).max(28); // system module lowering peaks high (no GC)
    let mut h2 = Host::new();
    let fs2 = grant_fs(&mut h2, &factory);
    let out2 = h2.grant_stream(StreamRole::Out);
    let ex2 = h2.grant_exit();
    let argv2: Vec<String> = ["hexer", "c", &format!("nimcache/{sys}.s.nif")]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let s2 = spawn_phase(
        &mut h2,
        &hexer,
        &argv2,
        &["fs", "stdout", "exit"],
        &[fs2, out2, ex2],
        hexer_carve,
    );
    eprintln!("hexer (op-13 JIT child) joined: {s2}");

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
