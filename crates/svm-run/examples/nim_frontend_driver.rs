//! The nimony FRONT-END driver on the SVM (NIM.md §3c/§3e, "compile Nim in the browser"): run the
//! real `nimsem` (sema) phase as an SVM guest that, being itself a driver, spawns the real `nifler`
//! (parse) phase as an exec child to parse stdlib modules on demand. This is the W4 multi-binary
//! driver applied to the actual front-end — nimsem's `system("nifler … parse <src> <out.p.nif>")`
//! (deps.nim) is routed by the shim to the `exec` capability, which runs `nifler.svmb` as an isolated
//! child domain sharing nimsem's memfs, so the `.p.nif` nifler writes is the one nimsem reads back.
//!
//! ```text
//! cargo run -q --release -p svm-run --example nim_frontend_driver -- \
//!     <nimsem.svmb> <nifler.svmb> <libdir> <sys.p.nif> <sys-stem>
//! ```
//!
//! Seeds a shared memfs with the nimony stdlib sources (under `lib/`) + the system module's parsed
//! `.p.nif`, grants nimsem `fs` + `exec`, runs the system semcheck, and dumps what it produced.

use std::path::Path;
use std::sync::Arc;

use svm_run::exec::{domain_exec_with_fs, DomainProgram};
use svm_run::{instantiate, Backend, HostCap, Limits, Outcome, RunConfig};

/// Recursively collect `(relative-key, bytes)` under `dir`, prefixing each key with `prefix`.
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
        "usage: nim_frontend_driver <nimsem.svmb> <nifler.svmb> <libdir> <sys.p.nif> <sys-stem>",
    );
    let nifler_p = a.next().expect("missing <nifler.svmb>");
    let libdir = a.next().expect("missing <libdir>");
    let sys_pnif = a.next().expect("missing <sys.p.nif>");
    let sys_stem = a.next().expect("missing <sys-stem>");

    // Seed: the stdlib sources under `lib/` (both preserving `std/` and flattened, so either module-
    // path layout resolves), plus the system module's parsed nif at `nimcache/<stem>.p.nif`.
    let mut files = vec![];
    collect(Path::new(&libdir), "lib/", &mut files);
    // flattened copy: lib/std/system/x.nim also reachable as lib/system/x.nim
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
    eprintln!("seeded {} files into the shared memfs", files.len());

    let (factory, handle) = svm_run::fs::mem_fs_shared_factory(files, vec!["nimcache".into()]);
    let factory = Arc::new(factory);
    // nimsem's own fs grant, and the shared fs the exec children inherit — same store.
    let nimsem_fs = {
        let f = factory.clone();
        HostCap::host_proc(0, move || (f)())
    };
    let child_fs = {
        let f = factory.clone();
        HostCap::host_proc(0, move || (f)())
    };

    let nifler = std::fs::read(&nifler_p).unwrap_or_else(|e| panic!("read {nifler_p}: {e}"));
    let nifler_inst = Arc::new(
        instantiate(svm_encode::decode_module(&nifler).expect("decode nifler.svmb"))
            .expect("instantiate nifler"),
    );
    // Register under the names nimsem may use for argv[0].
    let programs: Vec<DomainProgram> = ["nifler", "/bin/nifler"]
        .iter()
        .map(|n| DomainProgram {
            name: (*n).into(),
            instance: nifler_inst.clone(),
            limits: Limits::default(),
        })
        .collect();

    let nimsem = std::fs::read(&nimsem_p).unwrap_or_else(|e| panic!("read {nimsem_p}: {e}"));
    let inst = instantiate(svm_encode::decode_module(&nimsem).expect("decode nimsem.svmb"))
        .expect("instantiate nimsem");

    let cfg = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: [
            "nimsem",
            "--define:nimNativeAlloc",
            "--define:nimNativeIo",
            "m",
            "--isSystem",
            &format!("nimcache/{sys_stem}.p.nif"),
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect(),
        env: vec![],
        ..RunConfig::default()
    };

    let run = inst
        .run_with_caps(
            Backend::TreeWalk,
            &cfg,
            &[
                ("fs", nimsem_fs),
                ("exec", domain_exec_with_fs(programs, child_fs)),
            ],
        )
        .expect("run nimsem driver");

    eprintln!("outcome: {:?}", run.outcome);
    if !run.stdout.is_empty() {
        eprintln!(
            "--- guest stream ---\n{}\n---",
            String::from_utf8_lossy(&run.stdout)
        );
    }
    let (produced, _) = handle.seed();
    let out_key = format!("nimcache/{sys_stem}.s.nif");
    match produced.iter().find(|(k, _)| k == &out_key) {
        Some((_, bytes)) => {
            std::fs::write("/tmp/svm_sys.s.nif", bytes).ok();
            println!(
                "✅ nimsem-on-SVM produced {out_key} ({} bytes) → /tmp/svm_sys.s.nif",
                bytes.len()
            );
        }
        None => {
            let n = produced
                .iter()
                .filter(|(k, _)| k.ends_with(".p.nif"))
                .count();
            println!(
                "no {out_key} yet; store has {} files ({} .p.nif produced by nifler children)",
                produced.len(),
                n
            );
        }
    }
    std::process::exit(match run.outcome {
        Outcome::Exited(0) | Outcome::Returned(_) => 0,
        _ => 1,
    });
}
