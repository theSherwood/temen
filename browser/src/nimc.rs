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

use temen_interp::{
    bytecode, ForkedProc, Host, HostProc, HostProcFork, Region, StreamRole, Trap, Value,
};
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
pub(crate) fn module_suffix(file: &str) -> String {
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
pub(crate) fn parse_imports(deps_nif: &str, importer_dir: &str) -> Vec<String> {
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
    // reached by name (`self.resolve`) instead, so they're not bound here.
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
    let base = temen_ir::module_args_base() as usize;
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

// ---- run a phase as a confined §14 op-13 child on the resumable (tier-up-capable) engine (#1025) ---
// A phase run this way executes as a **separate-module confined child** over a sub-window carve instead
// of inline in the driver's own powerbox: the same resumable bytecode engine `run_phase` uses, but on
// the tier-up-capable path (`new_confined_child_over_host`) a JIT'd phase rides — matching the native
// op-13 conductor (`temen-run/examples/nim_chain_op13.rs`). `child` is a **child-entry** phase module
// (func 0 = `[I64]->[I64]`, built `--child-entry`); `{fs, stdout, exit}` are re-granted into it (`vm_map`
// auto-binds to the child's AddressSpace), argv is seeded into its carve, and its joined status returns.

/// The text-IR op-13 parent that spawns `child` (window `child_sl`, carve at `carve_off`) with the grant
/// list `{fs, stdout, exit}` and `argv` seeded at `carve + args_base`. Mirrors `nifler_child_asset.rs` /
/// `spawn_child_fs.rs`; the guarded child's `module_args_base` places records at 17408.., names at
/// 18432.. (above the #1094 NULL guard the parent itself carries), argv at `carve + args_base`.
pub(crate) fn op13_parent_src(child_sl: u32, carve_off: u64, args_base: u64, argv: &[&str]) -> String {
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
data 18432 "fs"
data 18448 "stdout"
data 18464 "exit"
data {argv_off} "{argv_esc}"
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32) {{
{r0}  hf = i64.extend_i32_u v2
  ohf = i64.const 17416
  i64.store ohf hf
{r1}  hs = i64.extend_i32_u v3
  ohs = i64.const 17432
  i64.store ohs hs
{r2}  he = i64.extend_i32_u v4
  ohe = i64.const 17448
  i64.store ohe he
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 17408
  vgn = i64.const 3
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
        r0 = rec(17408, 18432, 2),
        r1 = rec(17424, 18448, 6),
        r2 = rec(17440, 18464, 4),
    )
}

/// The resumable-engine drive loop (mirrors `temen-run/tests/child_entry_fs.rs`): on `Instantiate`, take
/// the op-13 re-granted powerbox (`take_granted_host`) and run the child over it
/// (`new_confined_child_over_host`, which binds the child manifest against that powerbox); `Join` delivers
/// the child's result. Single-threaded here (the driver worker), so the window base travels as a raw ptr.
fn drive_op13<'p>(
    prog: &'p bytecode::VcpuProgram,
    base: *mut u8,
    mut vcpu: bytecode::Vcpu<'p>,
) -> Result<Vec<Value>, Trap> {
    let mut children: Vec<Result<Vec<Value>, Trap>> = Vec::new();
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => return Ok(v),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::Instantiate {
                module,
                entry,
                carve,
                size_log2,
                fuel,
            } => {
                let granted = vcpu.take_granted_host();
                // SAFETY: the engine validated the carve within this vCPU's window (which outlives the
                // child); the child region aliases that sub-window — the §14 shared data plane.
                let child_base = unsafe { base.add(carve as usize) };
                let back =
                    std::sync::Arc::new(unsafe { Region::shared(child_base, 1u64 << size_log2) });
                let child = match granted {
                    Some(host) => bytecode::Vcpu::new_confined_child_over_host(
                        prog, module, entry, back, size_log2, fuel, host,
                    ),
                    None => bytecode::Vcpu::new_confined_child(
                        prog, module, entry, back, size_log2, fuel,
                    ),
                };
                let r = match child {
                    Ok(c) => drive_op13(prog, child_base, c),
                    Err(t) => Err(t),
                };
                let handle = children.len() as i32;
                children.push(r);
                vcpu.deliver_handle(handle);
            }
            bytecode::VcpuEvent::Join { handle } => {
                vcpu.deliver_join(children[handle as usize].clone());
            }
            _ => return Err(Trap::Malformed),
        }
    }
}

/// Run one child-entry phase `child` with `argv` as a confined §14 op-13 child over the shared memfs
/// `factory`, on the resumable (tier-up-capable) engine. Returns the joined exit/return status. The carve
/// is `(declared + 3).max(24)` (≥16 MiB — heap room above `1<<declared` for the phase's malloc).
fn run_phase_op13(child: &Module, argv: &[&str], factory: &FsFactory) -> i64 {
    let decl = child.memory.as_ref().map_or(24, |m| u32::from(m.size_log2));
    let child_sl = (decl + 3).max(24);
    let carve_off = 1u64 << child_sl;
    let src = op13_parent_src(child_sl, carve_off, temen_ir::module_args_base(), argv);
    let Ok(parent) = temen_text::parse_module(&src) else {
        return -1;
    };
    let Some(prog) = bytecode::VcpuProgram::compile(&parent) else {
        return -1;
    };
    let mut host = Host::new();
    let fs_init: HostProc = (*factory)();
    let fs_fork: HostProcFork = {
        let f = std::sync::Arc::clone(factory);
        std::sync::Arc::new(move |_pid| ForkedProc::shared((*f)()))
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork);
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    let win = 1u64 << (child_sl + 1);
    let inst = host.grant_instantiator(0, win);
    let modh = host.grant_module(child);

    let size = win as usize;
    let Ok(layout) = std::alloc::Layout::from_size_align(size, 8) else {
        return -1;
    };
    // SAFETY: non-zero 8-aligned layout; `size` valid bytes owned here until the dealloc below, after
    // every vCPU and region view is dropped.
    let mem_base = unsafe { std::alloc::alloc_zeroed(layout) };
    if mem_base.is_null() {
        return -1;
    }
    let back = std::sync::Arc::new(unsafe { Region::shared(mem_base, win) });
    let status = match bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(fs_h),
            Value::I32(stdout_h),
            Value::I32(exit_h),
        ],
        std::sync::Arc::clone(&back),
        &[],
        host,
    ) {
        Ok(root) => drive_op13(&prog, mem_base, root),
        Err(t) => Err(t),
    };
    drop(back);
    // SAFETY: same layout; the root vCPU and its region views are dropped above.
    unsafe { std::alloc::dealloc(mem_base, layout) };
    match status {
        Ok(v) => v.first().map_or(0, |x| match x {
            Value::I64(n) => *n,
            Value::I32(n) => *n as i64,
            _ => 0,
        }),
        Err(Trap::Exit(c)) => c as i64,
        Err(_) => -1,
    }
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
    compile_nim_ce(nifler, None, nimsem, hexer, files, main_nim)
}

/// [`compile_nim`] with an optional **child-entry** nifler (`nifler_ce`, built `--child-entry`). When
/// present, the phase-1 import crawl runs nifler as a confined §14 op-13 child on the tier-up-capable
/// engine ([`run_phase_op13`], #1025) instead of inline in the driver's powerbox; otherwise it stays on
/// the inline [`run_phase`] path (byte-identical output either way). nimsem/hexer stay inline for now
/// (their 256 MiB carves are the #816 half of the browser story).
pub fn compile_nim_ce(
    nifler: &[u8],
    nifler_ce: Option<&[u8]>,
    nimsem: &[u8],
    hexer: &[u8],
    files: Vec<(String, Vec<u8>)>,
    main_nim: &str,
) -> Result<String, String> {
    let nifler_m = Arc::new(temen_encode::decode_module(nifler).map_err(|_| "decode nifler")?);
    let nifler_ce_m = match nifler_ce {
        Some(bytes) => {
            let m = temen_encode::decode_module(bytes).map_err(|_| "decode nifler_ce")?;
            temen_verify::verify_module(&m).map_err(|_| "verify nifler_ce")?;
            Some(m)
        }
        None => None,
    };
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
        // #1025 route A: if the JS crawl already ran nifler on this module (on the wasm-JIT tier) and
        // seeded its `.p.nif` into the memfs, skip the (interpreter) nifler run here — the crawl and this
        // phase produce byte-identical NIF (proven by `op13_nifler_crawl_matches_inline`). Best-effort:
        // a module the JS crawl missed has no `.p.nif` present and runs nifler inline as before.
        if read(&handle, &format!("nimcache/{stem}.p.nif")).is_none() {
            let argv = ["nifler", "--portablePaths", "--deps", "parse", &file, &out];
            let code = match &nifler_ce_m {
                // #1025: the crawl phase runs nifler as a confined op-13 child on the tier-up engine.
                Some(ce) => run_phase_op13(ce, &argv, &factory),
                None => run_phase(&nifler_m, &argv, (factory)(), None).1,
            };
            if code != 0 && code != 5 {
                return Err(format!("nifler failed on {file} (code {code})"));
            }
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
    // Stream the compiled program's stdout live (#1143): the tee fires the `stdout_chunk` host import,
    // relayed to the page only while a streaming Run is active (a no-op otherwise).
    let out = crate::onramp_exec_with_tee(&m, &[], crate::stream_tee());
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

    /// Inflate a committed `.gz` asset with the system `gzip` (matching `nifler_child_asset.rs`).
    fn inflate(path: &str) -> Option<Vec<u8>> {
        use std::io::Write;
        let bytes = std::fs::read(path).ok()?;
        let mut c = std::process::Command::new("gzip")
            .args(["-dc"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .ok()?;
        let mut stdin = c.stdin.take()?;
        std::thread::spawn(move || {
            let _ = stdin.write_all(&bytes);
        });
        let out = c.wait_with_output().ok()?;
        out.status.success().then_some(out.stdout)
    }

    /// The committed **child-entry** nifler (`nifler_ce.temen.gz`) — the op-13 phase asset.
    fn inflate_nifler_ce() -> Option<Vec<u8>> {
        inflate("../crates/temen-run/demos/nifler_temen/nifler_ce.temen.gz")
    }

    /// Fast differential (no full compile): the phase-1 nifler crawl over one file must produce
    /// **byte-identical** `.p.nif` + `.p.deps.nif` whether run inline ([`run_phase`], top-level nifler)
    /// or as a confined op-13 child ([`run_phase_op13`], child-entry nifler_ce). Proves the #1025 wiring
    /// is output-preserving without paying the ~5 min end-to-end compile.
    #[test]
    fn op13_nifler_crawl_matches_inline() {
        // Both nifler builds are committed browser/demo assets — no nim toolchain needed, so this gates
        // in per-PR CI (unlike the `seed()`-gated end-to-end tests): the top-level `_start` nifler the
        // card ships and the child-entry `nifler_ce`, which share the parser, must emit identical NIF.
        let Some(ce) = inflate_nifler_ce() else {
            eprintln!("SKIP: nifler_ce.temen.gz unavailable / gzip missing");
            return;
        };
        let Some(top) = inflate("web/assets/nifler.temen.gz") else {
            eprintln!("SKIP: web/assets/nifler.temen.gz unavailable");
            return;
        };
        let ce_m = temen_encode::decode_module(&ce).expect("decode nifler_ce");
        temen_verify::verify_module(&ce_m).expect("verify nifler_ce");
        let top_m = Arc::new(temen_encode::decode_module(&top).expect("decode nifler"));

        let src = b"let x = 1\n".to_vec();
        let argv = [
            "nifler",
            "--portablePaths",
            "--deps",
            "parse",
            "/in.nim",
            "/nimcache/in.p.nif",
        ];
        let run = |m_inline: Option<&Arc<Module>>, ce: Option<&Module>| {
            let (factory, handle) = temen_fs::mem_fs_shared_factory(
                vec![("in.nim".into(), src.clone())],
                vec!["nimcache".into()],
            );
            let factory: FsFactory = Arc::new(factory);
            let code = match (m_inline, ce) {
                (Some(m), _) => run_phase(m, &argv, (factory)(), None).1,
                (None, Some(c)) => run_phase_op13(c, &argv, &factory),
                _ => unreachable!(),
            };
            (
                code,
                read(&handle, "nimcache/in.p.nif"),
                read(&handle, "nimcache/in.p.deps.nif"),
            )
        };
        let (c_inline, nif_inline, deps_inline) = run(Some(&top_m), None);
        let (c_op13, nif_op13, deps_op13) = run(None, Some(&ce_m));

        assert!(
            nif_op13.is_some(),
            "op-13 nifler wrote its .p.nif into the shared memfs"
        );
        assert_eq!(
            c_inline, c_op13,
            "exit codes agree (inline {c_inline} vs op-13 {c_op13})"
        );
        assert_eq!(
            nif_inline, nif_op13,
            ".p.nif byte-identical between inline and op-13 nifler"
        );
        assert_eq!(
            deps_inline, deps_op13,
            ".p.deps.nif byte-identical between inline and op-13 nifler"
        );
    }

    /// End-to-end: the full in-browser compile with the phase-1 crawl on the op-13 path
    /// ([`compile_nim_ce`] with the child-entry nifler) produces the same program output as the inline
    /// path. Slow (~5 min) — the interpreter runs the whole front-end — so it shares `io_hello`'s gate.
    #[test]
    fn io_hello_op13_nifler() {
        let Some((nifler, nimsem, hexer, mut files)) = seed() else {
            return;
        };
        let Some(ce) = inflate_nifler_ce() else {
            eprintln!("SKIP: nifler_ce.temen.gz unavailable");
            return;
        };
        files.push((
            "prog.nim".into(),
            b"import std/syncio\n\nproc greet(name: string): string =\n  \"hello, \" & name & \"\\n\"\n\nwrite(stdout, greet(\"Nim\"))\nwrite(stdout, greet(\"the Temen\"))\n".to_vec(),
        ));
        let r = compile_nim_ce(&nifler, Some(&ce), &nimsem, &hexer, files, "prog.nim");
        assert_eq!(
            r.as_deref(),
            Ok("hello, Nim\nhello, the Temen\n"),
            "op-13-crawl in-browser compile matches the inline path"
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
