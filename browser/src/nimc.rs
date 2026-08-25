//! **The nimony compiler, in the browser** (NIM.md §3c/§3e; #958). The wasm port of the headless
//! `nim_selfdrive` driver (`crates/temen-run/examples/nim_selfdrive.rs`): given only Nim source + the
//! stdlib — **no native nimony** — it plays nifmake itself (computes each module's cache stem exactly
//! as nimony does, crawls the `import` graph with `nifler`), runs `nimsem` + `hexer` over the closure
//! as sandboxed Temen guests (nimsem spawns nifler over the shared memfs via a wasm-native `exec` cap),
//! and links + runs the result with `temen-leng`. Every phase runs on the browser's bytecode engine;
//! the whole thing is client-side.
//!
//! Guests here run through [`bytecode::compile_and_run_capture_reserved_with_host`] — the same engine
//! `temen_run_nifler_fs` (the single-phase card) already runs `nifler` on. The `exec` cap's child run is
//! a *nested* guest run inside the parent's cap dispatch (as `temen-run`'s `domain_exec` does natively).

use std::collections::BTreeMap;
use std::sync::Arc;

use temen_interp::{bytecode, Host, HostProc, StreamRole, Trap, Value};
use temen_ir::Module;

use crate::{onramp_cap_resolver, onramp_check, pg_args_blob};

// ---- nimony's module-stem hash (gear2/modnames.nim + lib/tinyhashes.nim), reproduced exactly -------

fn uhash(s: &str) -> u32 {
    let mut h: u32 = 0;
    for c in s.bytes() {
        h = h.wrapping_add(c as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h
}

fn base36(mut id: u32) -> String {
    const B36: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut r = String::new();
    while id > 0 {
        r.push(B36[(id % 36) as usize] as char);
        id /= 36;
    }
    r
}

fn relative_path(path: &str, base: &str) -> String {
    let p: Vec<&str> = path
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let b: Vec<&str> = base
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let mut i = 0;
    while i < p.len() && i < b.len() && p[i] == b[i] {
        i += 1;
    }
    let mut out: Vec<&str> = vec![".."; b.len() - i];
    out.extend_from_slice(&p[i..]);
    out.join("/")
}

/// `gear2/modnames.moduleSuffix` — `name[0..3]` + base36(`uhash`) of the shortest of the file's path
/// relative to the cwd (`/`) and to each search path (`/lib`).
fn module_suffix(file: &str) -> String {
    let mut rel = relative_path(file, "/");
    let c = relative_path(file, "/lib");
    if c.len() < rel.len() {
        rel = c;
    }
    let name = rel.rsplit('/').next().unwrap_or(&rel);
    let name = name.strip_suffix(".nim").unwrap_or(name);
    let mut stem: String = name.chars().take(3).collect();
    stem.push_str(&base36(uhash(&rel)));
    stem
}

// ---- .p.deps.nif import crawl (mirrors nim_selfdrive) ----------------------------------------------

fn balanced(s: &str, head: &str) -> Option<String> {
    let start = s.find(&format!("({head}"))?;
    let b = s.as_bytes();
    let (mut j, mut depth) = (start, 0i32);
    while j < b.len() {
        match b[j] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..=j].to_string());
                }
            }
            _ => {}
        }
        j += 1;
    }
    None
}

/// Active `import` targets as absolute memfs file paths; platform-`when`-guarded imports are skipped.
fn parse_imports(deps_nif: &str, importer_dir: &str) -> Vec<String> {
    let mut out = vec![];
    for kw in ["import", "fromimport"] {
        let mut rest = deps_nif;
        while let Some(block) = balanced(rest, kw) {
            let adv = rest.find(&block).unwrap() + block.len();
            rest = &rest[adv..];
            if block.contains("(when") {
                continue;
            }
            if let Some(inf) = balanced(&block, "infix") {
                let segs: Vec<&str> = inf
                    .trim_start_matches("(infix")
                    .trim_end_matches(')')
                    .split_whitespace()
                    .filter(|t| *t != "/" && !t.starts_with('('))
                    .collect();
                if !segs.is_empty() {
                    out.push(format!("/lib/{}.nim", segs.join("/")));
                }
            } else if let Some(pre) = balanced(&block, "prefix") {
                let segs: Vec<&str> = pre
                    .trim_start_matches("(prefix")
                    .trim_end_matches(')')
                    .split_whitespace()
                    .filter(|t| !t.contains("2E") && *t != "/" && !t.starts_with('\\'))
                    .collect();
                if !segs.is_empty() {
                    out.push(format!("{importer_dir}/{}.nim", segs.join("/")));
                }
            }
        }
    }
    out
}

// ---- run a phase guest (nifler/nimsem/hexer) on the bytecode engine, granting the powerbox + caps --

/// A fresh `fs` `HostProc` over the shared memfs store (a new grant per phase run / per exec spawn).
type FsFactory = Arc<dyn Fn() -> HostProc + Send + Sync>;

/// Run phase module `m` with `argv` on the bytecode engine, granting the on-ramp powerbox
/// (stdout/stdin/exit/memory), a fresh `fs` grant, and — for `nimsem` — an `exec` cap. Returns the
/// guest's captured stdout and its exit/return code. Mirrors `temen-run`'s `run_with_caps` (and the
/// browser's `pg_setup`) but with the shared-factory `fs` + optional `exec` the multibinary driver needs.
fn run_phase(m: &Module, argv: &[&str], fs: HostProc, exec: Option<HostProc>) -> (Vec<u8>, i64) {
    if onramp_check(m).is_err() {
        return (b"phase module is not a manifest module".to_vec(), -1);
    }
    let mut host = Host::new();
    let out = host.grant_stream(StreamRole::Out);
    host.register_cap_name("stdout", out);
    let inp = host.grant_stream(StreamRole::In);
    host.register_cap_name("stdin", inp);
    let exit = host.grant_exit();
    host.register_cap_name("exit", exit);
    let memory = host.grant_memory();
    host.register_cap_name("memory", memory);
    let fsh = host.grant_host_proc(fs);
    host.register_cap_name("fs", fsh);
    if let Some(e) = exec {
        let eh = host.grant_host_proc(e);
        host.register_cap_name("exec", eh);
    }
    // Manifest slot bindings for the on-ramp powerbox imports (stdout/stdin/exit/memory) — fs/exec are
    // reached by name (`cap.self.resolve`) instead, so they're not bound here.
    if !m.imports.is_empty() {
        use temen_interp::cap_id;
        let bindings = m
            .imports
            .iter()
            .map(|im| {
                let Some(cap) = onramp_cap_resolver(&im.name) else {
                    return temen_interp::BoundImport::rebindable(0, 0, None);
                };
                let handle = match (cap.type_id, cap.op) {
                    (cap_id::STREAM, 1) => out,
                    (cap_id::STREAM, _) => inp,
                    (cap_id::EXIT, _) => exit,
                    (cap_id::ADDRESS_SPACE, _) => memory,
                    _ => return temen_interp::BoundImport::rebindable(0, 0, None),
                };
                temen_interp::BoundImport::required(cap.type_id, cap.op, handle)
            })
            .collect();
        host.set_import_bindings(bindings);
    }
    // Seed argv at the module's args base (the on-ramp `_start` parses argc/argv from it). #964/#1094:
    // a phase guest reads its args one guard up, at `module_args_base` (guard + POWERBOX_ARGS_BASE) —
    // the unconditional guarded layout; key off `module_args_base`, not a bare constant, so
    // nifler/nimsem/hexer find argv where their `_start` looks.
    let blob = pg_args_blob(&argv.iter().map(|s| s.as_bytes()).collect::<Vec<_>>());
    let base = temen_ir::module_args_base(m) as usize;
    let mut init_mem = vec![0u8; base + blob.len()];
    init_mem[base..].copy_from_slice(&blob);

    let mut fuel = u64::MAX;
    let outcome = bytecode::compile_and_run_capture_reserved_with_host(
        m,
        0,
        &[],
        &mut fuel,
        &init_mem,
        temen_ir::DEFAULT_RESERVED_LOG2,
        &mut host,
    );
    let code = match outcome {
        Some((Ok(vals), _)) => vals.first().map_or(0, |v| match v {
            Value::I64(x) => *x,
            Value::I32(x) => *x as i64,
            _ => 0,
        }),
        Some((Err(Trap::Exit(c)), _)) => c as i64,
        Some((Err(_), _)) => -1,
        None => -2, // bytecode engine declined (should not happen for these on-ramp guests)
    };
    (host.stdout, code)
}

/// The wasm-native `exec` cap: `nimsem`'s `system("nifler … parse …")` (routed by the shim to the
/// `exec` capability) runs `nifler` as a nested guest over the **same** memfs. Only `nifler` is in the
/// registry (argv[0] `nifler` or `/bin/nifler`); anything else is refused. Non-`run` ops
/// (status/read/close) go through `temen_exec::JobTable`, exactly like `temen-run`'s `domain_exec`.
fn make_exec(nifler: Arc<Module>, fs_factory: FsFactory) -> HostProc {
    let mut jobs = temen_exec::JobTable::default();
    Box::new(move |op, args, mem, _minter| {
        if op != temen_exec::EXEC_RUN {
            return Ok(vec![jobs.handle(op, args, mem)]);
        }
        let (argv, _stdin) = match temen_exec::run_args(args, mem.as_deref()) {
            Ok(x) => x,
            Err(e) => return Ok(vec![e]),
        };
        if !matches!(
            argv.first().map(String::as_str),
            Some("nifler") | Some("/bin/nifler")
        ) {
            return Ok(vec![temen_ir::errno::EPERM]);
        }
        let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();
        let (stdout, exit) = run_phase(&nifler, &argv_refs, (fs_factory)(), None);
        Ok(vec![jobs.push(temen_exec::Job {
            stdout,
            stderr: vec![],
            out_pos: 0,
            err_pos: 0,
            exit,
        })])
    })
}

// ---- the driver: crawl → nimsem → hexer → link → run ----------------------------------------------

#[derive(PartialEq, Clone, Copy)]
enum Role {
    System,
    Main,
    Import,
}

struct Mod {
    stem: String,
    role: Role,
    deps: Vec<String>,
}

/// Compile `main_nim` (present in `files` at its bare name, alongside any imported siblings) plus the
/// stdlib (also in `files`, under `lib/`) entirely on the Temen through the real nimony phases, link it
/// through the nim→powerbox bridge, **run `_start` under the powerbox**, and return its captured
/// **stdout** (empty for a program that writes nothing). `nifler`/`nimsem`/`hexer` are the decoded
/// phase modules. `Ok(stdout)` on success, `Err(diagnostic)` on any phase/link/run failure.
pub fn compile_nim(
    nifler: &[u8],
    nimsem: &[u8],
    hexer: &[u8],
    files: Vec<(String, Vec<u8>)>,
    main_nim: &str,
) -> Result<String, String> {
    let nifler_m = Arc::new(temen_encode::decode_module(nifler).map_err(|_| "decode nifler")?);
    let nimsem_m = temen_encode::decode_module(nimsem).map_err(|_| "decode nimsem")?;
    let hexer_m = temen_encode::decode_module(hexer).map_err(|_| "decode hexer")?;

    let (factory, handle) = temen_fs::mem_fs_shared_factory(files, vec!["nimcache".into()]);
    let factory: FsFactory = Arc::new(factory);

    // ---- phase 1: parse + crawl the import closure with nifler ------------------------------------
    let mut mods: BTreeMap<String, Mod> = BTreeMap::new();
    let mut work = vec![
        ("/lib/std/system.nim".to_string(), Role::System),
        (format!("/{main_nim}"), Role::Main),
    ];
    while let Some((file, role)) = work.pop() {
        let stem = module_suffix(&file);
        if mods.contains_key(&stem) {
            continue;
        }
        let out = format!("/nimcache/{stem}.p.nif");
        let (_o, code) = run_phase(
            &nifler_m,
            &["nifler", "--portablePaths", "--deps", "parse", &file, &out],
            (factory)(),
            None,
        );
        if code != 0 && code != 5 {
            return Err(format!("nifler failed on {file} (code {code})"));
        }
        let deps_nif = read(&handle, &format!("nimcache/{stem}.p.deps.nif"))
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default();
        let dir = file
            .rsplit_once('/')
            .map(|(d, _)| d)
            .unwrap_or("")
            .to_string();
        let mut deps = vec![];
        for imp in parse_imports(&deps_nif, &dir) {
            deps.push(module_suffix(&imp));
            work.push((imp, Role::Import));
        }
        mods.insert(stem.clone(), Mod { stem, role, deps });
    }

    // ---- dependency order (DFS postorder), system first ------------------------------------------
    let order = toposort(&mods);
    let main_stem = mods
        .values()
        .find(|m| m.role == Role::Main)
        .map(|m| m.stem.clone())
        .ok_or("no main module")?;

    // ---- phase 2: nimsem (dependency-ordered), driving nifler via exec ----------------------------
    for stem in &order {
        let m = &mods[stem];
        let pnif = format!("nimcache/{stem}.p.nif");
        let mut argv = vec![
            "nimsem",
            "--define:nimNativeAlloc",
            "--define:nimNativeIo",
            "m",
        ];
        match m.role {
            Role::System => argv.push("--isSystem"),
            Role::Main => argv.push("--isMain"),
            Role::Import => {}
        }
        argv.push(&pnif);
        let exec = make_exec(nifler_m.clone(), factory.clone());
        let (_o, code) = run_phase(&nimsem_m, &argv, (factory)(), Some(exec));
        if code != 0 && code != 5 {
            return Err(format!("nimsem failed on {stem} (code {code})"));
        }
        if read(&handle, &format!("nimcache/{stem}.s.nif")).is_none() {
            return Err(format!("nimsem produced no {stem}.s.nif"));
        }
    }

    // ---- phase 3: hexer (main gets the app-entry glue) -------------------------------------------
    let outdir = format!("nimcache/{main_stem}");
    let mut leng: Vec<(String, String)> = Vec::new();
    for stem in &order {
        let is_main = stem == &main_stem;
        let s_nif = format!("nimcache/{stem}.s.nif");
        let outdir_arg = format!("--outdir:{outdir}");
        let argv: Vec<&str> = if is_main {
            vec![
                "hexer",
                "c",
                "--bits:64",
                "--cpu:le",
                "--flags:br",
                "--isMain",
                "--app:console",
                &outdir_arg,
                &s_nif,
            ]
        } else {
            vec!["hexer", "c", &s_nif]
        };
        let (_o, code) = run_phase(&hexer_m, &argv, (factory)(), None);
        if code != 0 && code != 5 {
            return Err(format!("hexer failed on {stem} (code {code})"));
        }
        let key = if is_main {
            format!("{outdir}/{stem}.x.nif")
        } else {
            format!("nimcache/{stem}.x.nif")
        };
        let x = read(&handle, &key).ok_or(format!("hexer produced no {key}"))?;
        leng.push((stem.clone(), String::from_utf8_lossy(&x).into_owned()));
    }

    // ---- phase 4: link + run (main first, system last) -------------------------------------------
    let mut ordered: Vec<&(String, String)> = leng.iter().collect();
    ordered.sort_by_key(|(stem, _)| (*stem != main_stem, stem.starts_with("sysv")));
    let units: Vec<temen_leng::WholeModule> = ordered
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();

    let m = temen_leng::link_nim_powerbox(&units).map_err(|e| format!("nim→powerbox link: {e}"))?;
    temen_verify::verify_module(&m).map_err(|e| format!("verify: {e:?}"))?;
    let out = crate::onramp_exec(&m, &[]);
    if out.status != crate::STATUS_OK && out.status != crate::STATUS_EXIT {
        return Err(format!("run failed (status {})", out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn read(handle: &temen_fs::MemFsHandle, key: &str) -> Option<Vec<u8>> {
    let (files, _) = handle.seed();
    files.into_iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

#[cfg(test)]
mod tests {
    //! Native validation of the in-browser compiler over the bytecode engine (the same engine the
    //! wasm cdylib uses). Skips unless the phase `.temen` are staged at `/tmp/e2e_temen` and the stdlib
    //! at `.nimtool/nimony/lib` — build them with `demos/nim_e2e_chain/build_e2e_chain.sh`.
    use super::*;

    fn seed() -> Option<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<(String, Vec<u8>)>)> {
        let dir = std::path::Path::new("/tmp/e2e_temen");
        let lib = std::path::Path::new("../.nimtool/nimony/lib");
        if !dir.join("nifler.temen").exists() || !lib.exists() {
            eprintln!("SKIP: phase .temen or stdlib absent");
            return None;
        }
        let mut files = vec![];
        let mut stack = vec![lib.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    let rel = p
                        .strip_prefix(lib)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    let bytes = std::fs::read(&p).unwrap();
                    files.push((format!("lib/{rel}"), bytes.clone()));
                    if let Some(r) = rel.strip_prefix("std/") {
                        files.push((format!("lib/{r}"), bytes));
                    }
                }
            }
        }
        Some((
            std::fs::read(dir.join("nifler.temen")).unwrap(),
            std::fs::read(dir.join("nimsem.temen")).unwrap(),
            std::fs::read(dir.join("hexer.temen")).unwrap(),
            files,
        ))
    }

    #[test]
    fn io_hello() {
        let Some((nifler, nimsem, hexer, mut files)) = seed() else {
            return;
        };
        // Mirrors the playground `nimc` card's default source (browser/web/play.js): a `proc` with a
        // `string` parameter, string concatenation (`&`), and two `write`s. The native oracle for the
        // in-browser compile — same bytecode engine the wasm cdylib runs — so card breadth can't drift
        // past what compiles here.
        files.push((
            "prog.nim".into(),
            b"import std/syncio\n\nproc greet(name: string): string =\n  \"hello, \" & name & \"\\n\"\n\nwrite(stdout, greet(\"Nim\"))\nwrite(stdout, greet(\"the Temen\"))\n".to_vec(),
        ));
        let r = compile_nim(&nifler, &nimsem, &hexer, files, "prog.nim");
        assert_eq!(
            r.as_deref(),
            Ok("hello, Nim\nhello, the Temen\n"),
            "in-browser compile+run of a proc + string-concat program"
        );
    }
}

fn toposort(mods: &BTreeMap<String, Mod>) -> Vec<String> {
    let mut seen: BTreeMap<String, u8> = BTreeMap::new();
    let mut order = vec![];
    fn visit(
        s: &str,
        mods: &BTreeMap<String, Mod>,
        seen: &mut BTreeMap<String, u8>,
        order: &mut Vec<String>,
    ) {
        if seen.get(s).copied().unwrap_or(0) != 0 {
            return;
        }
        seen.insert(s.to_string(), 1);
        if let Some(m) = mods.get(s) {
            for d in &m.deps {
                visit(d, mods, seen, order);
            }
        }
        seen.insert(s.to_string(), 2);
        order.push(s.to_string());
    }
    for s in mods.keys() {
        visit(s, mods, &mut seen, &mut order);
    }
    order.sort_by_key(|s| mods.get(s).map(|m| m.role != Role::System).unwrap_or(true));
    order
}
