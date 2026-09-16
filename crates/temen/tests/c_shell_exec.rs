//! Stage 1 (STAGE1.md) — **a compiled-C shell execs a compiled-C command**: the shell (not
//! hand-written IR) spawns a *separate*, unmodified compiled-C command with inherited stdout and
//! collects its status. The shell parses its own `argv` (the powerbox args buffer), looks the command
//! up, seeds the command's `argv` into a carve, and spawns + `join`s through `posix_libc/spawn.c`
//! (#1509) — `vm_spawn` fills the `Instantiator.instantiate_rec` (op 17) record and the by-name grant
//! list (`{"stdout" -> out}`) that used to be laid out by hand here — the whole external-command path
//! emitted by the frontend.
//!
//! Both the shell and the command are ordinary C. Capability wiring: `stdout` is a re-grantable
//! `Stream` (shared sink, so the command's output and any shell output unify); `exec_stdout`/
//! `exec_lookup` are a tiny host fn (the embedder's PATH → `Module` map); the helper's
//! `vm_instantiate_rec`/`vm_instantiate_join` externs bind through the one shared name table
//! (`temen_ir::default_cap_resolver`, `Resolved::Cap`, link-time symbol resolution) and dispatch on
//! the `Instantiator`/host-fn handles the guest discovers itself via `cap.self` reflection.
//! Differential interp==JIT — the JIT is given the module resolver *and* the named-grant hooks the
//! record spawn needs.
//!
//! The second test is the **per-child attenuation witness**: one parent spawns the same command
//! twice with different grant lists, and only the child that was handed `stdout` can resolve it.
//!
//! This is the frontend-drives-exec proof. Folding it into the full `c_shell.rs` builtin dispatch (its
//! personality-heap-at-`win/2` layout vs. a 128 KiB command carve) is the follow-up.
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain).
#![cfg(unix)]

#[path = "support/grant_hooks.rs"]
mod grant_hooks_mod;
#[path = "support/repo_root.rs"]
mod repo_root_mod;
use grant_hooks_mod::grant_hooks;

use repo_root_mod::repo_root;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use temen_interp::{run_capture_reserved_with_host, GuestMem, Host, StreamRole, Trap};
use temen_ir::{Resolved, ResolvedCap};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};
use temen_text::parse_module as parse_module_raw;
use temen_verify::verify_module;

fn chibicc() -> &'static Path {
    static CC: OnceLock<PathBuf> = OnceLock::new();
    CC.get_or_init(|| {
        let dir = repo_root().join("frontend/chibicc");
        let status = Command::new("make")
            .arg("-s")
            .current_dir(&dir)
            .status()
            .unwrap();
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}

/// Compile `src` to text IR; `child_entry` selects the §14 spawnable entry ABI.
fn c_to_ir(src: &str, child_entry: bool) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("temen_cshx_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("temen");
    std::fs::write(&cfile, src).unwrap();
    let mut args = vec!["-cc1", "--emit-ir"];
    if child_entry {
        args.push("--child-entry");
    }
    let cin = cfile.to_str().unwrap().to_string();
    let cout = irfile.to_str().unwrap().to_string();
    args.extend(["-cc1-input", &cin, "-cc1-output", &cout, &cin]);
    let status = Command::new(chibicc()).args(&args).status().unwrap();
    assert!(status.success(), "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// The command: echo every `argv[i]` on its own line (ambient `write(1, …)`), return `argc`.
const CMD: &str = r#"
long write(long fd, void *buf, long n);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
int main(int argc, char **argv){
  for (int i = 0; i < argc; i++){ write(1, argv[i], slen(argv[i])); write(1, "\n", 1); }
  return argc;
}
"#;

/// The guest-side spawn helper (#1509), concatenated ahead of each C program that spawns.
const SPAWN_C: &str = include_str!("../../temen-run/demos/posix_libc/spawn.c");

/// The shell: `main(argc, argv)` — exec `argv[1]` as an external command, passing it `argv[1..]`.
/// `pool` (a big writable global) both forces a window large enough for the command's carve and holds
/// the spawn record + the aligned carve. Names link via imports (see [`link_shim`]); handles come
/// from the guest's own cap.self reflection (`vm_cap_of`).
///
/// chibicc widens every scalar to an i64 slot, so the helper's `call.cap 6 17`/`6 1` are declared
/// `(i64…) -> (i64)` even though the Instantiator contract's canonical child handle is i32. Both
/// backends reconcile that width: the interp reads args as i64 slots and coerces the result to the
/// declared type; the JIT's `lower_instantiator` does the matching `slot_i64`/`slot_i32`/`result_as`
/// coercions.
const SHELL_MAIN: &str = r#"
long exec_stdout(int h);
long exec_lookup(int h, char *name, long len);
long stream_write(int h, void *buf, long n);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
/* 384 KiB: room for the spawn record low, a 128 KiB-aligned 128 KiB carve, all below the SP. */
static char pool[393216];
int main(int argc, char **argv){
  int hf = vm_cap_of(13);   /* HOST_PROC = 13: the embedder's exec host fn */
  long out = exec_stdout(hf);
  if (argc < 2) return 1;
  long mod = exec_lookup(hf, argv[1], slen(argv[1]));
  if (mod < 0){ stream_write(out, "not found\n", 10); return 127; }
  long carve = ((long)pool + 131071) & ~131071;
  /* the command's args buffer at carve + guard + 128 (#1059: chibicc reads argv one 16 KiB NULL
     guard up, module_args_base): {argc-1, envc=0} then packed argv[1..] */
  char *ab = (char *)(carve + 16384 + 128);
  int *hdr = (int *)ab;
  hdr[0] = argc - 1;
  hdr[1] = 0;
  char *p = ab + 8;
  for (int i = 1; i < argc; i++){ char *s = argv[i]; long L = slen(s); for (long k=0;k<L;k++) *p++ = s[k]; *p++ = 0; }
  vm_grant g[1];
  g[0].name = "stdout";
  g[0].handle = (int)out;
  long child = vm_spawn(mod, 0, carve, 17, 0, g, 1, pool);
  return (int)vm_join(child);
}
"#;

/// The attenuation-witness command: returns 1 and prints if it was handed `stdout`, else 0. It
/// writes through the handle `self.resolve` returns (`__vm_write`, a `call.cap` on that handle) rather
/// than the ambient `write`: an ambient `write` is a manifest import, and a child whose manifest names
/// a capability its powerbox cannot bind is refused at spawn (`-EINVAL`, fail closed) — which is also
/// attenuation, but the witness wants the ungranted child to *run* and find nothing.
const PROBE: &str = r#"
long __vm_resolve(const char *name, long len);
long __vm_write(int h, void *buf, long len);
int main(int argc, char **argv){
  long h = __vm_resolve("stdout", 6);
  if (h < 0) return 0;
  __vm_write((int)h, "granted\n", 8);
  return 1;
}
"#;

/// The witness parent: spawns `PROBE` twice from one module — child A with `{"stdout" -> out}`,
/// child B with an empty grant list — and returns `A*10 + B`.
const TWO_CHILDREN: &str = r#"
long exec_stdout(int h);
long exec_lookup(int h, char *name, long len);
/* 640 KiB: the spawn record low, then two 128 KiB-aligned 128 KiB carves. */
static char pool[655360];
int main(int argc, char **argv){
  int hf = vm_cap_of(13);
  long out = exec_stdout(hf);
  long mod = exec_lookup(hf, "echo", 4);
  long ca = ((long)pool + 131071) & ~131071;
  long cb = ca + 131072;
  vm_grant g[1];
  g[0].name = "stdout";
  g[0].handle = (int)out;
  long a = vm_spawn(mod, 0, ca, 17, 0, g, 1, pool);
  long ra = vm_join(a);
  long b = vm_spawn(mod, 0, cb, 17, 0, g, 0, pool);
  long rb = vm_join(b);
  return (int)(ra * 10 + rb);
}
"#;

/// The embedder's PATH → `Module` map + stdout handle, as one host fn (op 0 = stdout handle, op 1 =
/// look a command name up). Returns handle values valid in the shell's own cap table.
fn exec_host(out_h: i32, echo_h: i32) -> temen_interp::HostProc {
    Box::new(
        move |op: u32,
              args: &[i64],
              mem: Option<&mut dyn GuestMem>,
              _minter: Option<&mut dyn temen_interp::RegionMinter>| match op {
            0 => Ok(vec![out_h as i64]),
            1 => {
                let mem = mem.ok_or(Trap::Malformed)?;
                let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
                let len = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
                let name = mem.read_bytes(ptr, len).ok_or(Trap::Malformed)?;
                Ok(vec![if name == b"echo" { echo_h as i64 } else { -1 }])
            }
            _ => Err(Trap::CapFault),
        },
    )
}

/// Link the shell's import names to their interfaces — link-time symbol resolution (the phase-4
/// linker-only `resolve_imports_with`; IMPORTS.md §2.5): `exec_stdout`/`exec_lookup` are the embedder
/// host fn's ops (0 / 1); `stream_write` is `Stream.write`; everything else (the spawn helper's
/// `vm_instantiate_rec`/`vm_instantiate_join`) comes from the one shared name table. No handle is
/// baked at link: each lowered `call.cap` dispatches on the guest's own handle operand, discovered at
/// run time via `__vm_cap_count`/`__vm_cap_at` reflection (§3c protection at the boundary,
/// IMPORTS.md §2.3 dynamic mode).
fn link_shim(name: &str) -> Option<Resolved> {
    let cap = match name {
        "stream_write" => ResolvedCap { type_id: 0, op: 1 },
        "exec_stdout" => ResolvedCap { type_id: 13, op: 0 },
        "exec_lookup" => ResolvedCap { type_id: 13, op: 1 },
        _ => temen_ir::default_cap_resolver(name)?,
    };
    Some(Resolved::Cap(cap))
}

/// The §3e args blob for `argv` (the shell's own args): `{argc, envc}` + packed NUL-terminated strings.
fn args_blob(argv: &[&str]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(argv.len() as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    for a in argv {
        b.extend_from_slice(a.as_bytes());
        b.push(0);
    }
    b
}

/// Run the shell with powerbox `argv`; return (status, stdout).
fn run(
    shell: &temen_ir::Module,
    cmd: &temen_ir::Module,
    argv: &[&str],
    jit: bool,
) -> (i64, Vec<u8>) {
    let win = 1usize << shell.memory.expect("shell window").size_log2;
    let mut host = Host::new();
    let _sink = host.shared_stdout(); // route the stdout Stream + re-granted child streams to one sink
    let out_h = host.grant_stream(StreamRole::Out);
    let _inst_h = host.grant_instantiator(0, win as u64);
    let echo_h = host.grant_module(cmd);
    let _exec_h = host.grant_host_proc(exec_host(out_h, echo_h));
    // Link the shell's imports to their interfaces; the guest discovers the handles by reflection.
    let m = temen_ir::resolve_imports_with(shell, link_shim).expect("resolve");
    verify_module(&m).expect("verify shell");
    // Seed the shell's own args buffer where its guard-shifted `_start` reads it (#1059/#1094:
    // `module_args_base` = guard + POWERBOX_ARGS_BASE, the unconditional guarded layout).
    let mut init = vec![0u8; win];
    let blob = args_blob(argv);
    let args_base = temen_ir::module_args_base() as usize;
    init[args_base..args_base + blob.len()].copy_from_slice(&blob);

    if jit {
        let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
            &m,
            0,
            &[],
            &init,
            0,
            temen_run::cap_thunk,
            &mut host as *mut Host as *mut c_void,
            Some(temen_run::module_resolver),
            Some(grant_hooks(&mut host as *mut Host)),
        )
        .expect("jit");
        let code = match jo {
            JitOutcome::Returned(ref s) => s.first().copied().unwrap_or(0),
            JitOutcome::Exited(c) => c as i64,
            ref o => panic!("jit ended abnormally: {o:?}"),
        };
        (code, host.stdout_bytes())
    } else {
        let mut fuel = 200_000_000u64;
        let res = run_capture_reserved_with_host(&m, 0, &[], &mut fuel, &init, 0, &mut host);
        let code = match res.0 {
            Ok(ref v) => match v.first() {
                Some(temen_interp::Value::I32(x)) => *x as i64,
                Some(temen_interp::Value::I64(x)) => *x,
                _ => 0,
            },
            Err(Trap::Exit(c)) => c as i64,
            Err(e) => panic!("interp trapped: {e:?}"),
        };
        (code, host.stdout_bytes())
    }
}

/// The compiled shell execs the compiled `echo` command with inherited stdout, identically on both
/// backends: the command's argv reaches its stdout (the shell's sink) and its `argc` is the shell's
/// exit status. This is the record spawn (op 17) driven end to end by the frontend.
#[test]
fn compiled_shell_execs_command_via_vm_spawn() {
    // Parse the shell raw — its imports (`vm_instantiate_*`/`exec_*`/`stream_write`) are resolved per-run by
    // `resolver` against that run's handles, so the names must survive parsing.
    let shell = parse_module_raw(&c_to_ir(&format!("{SPAWN_C}\n{SHELL_MAIN}"), false))
        .expect("parse shell");
    // Phase 3: keep the manifest — the spawn binds the child's slots by name.
    let cmd = parse_module_raw(&c_to_ir(CMD, true)).expect("parse cmd");
    verify_module(&cmd).expect("verify cmd");

    for argv in [&["sh", "echo", "hi"][..], &["sh", "echo", "a", "bb"][..]] {
        let (ic, iout) = run(&shell, &cmd, argv, false);
        let (jc, jout) = run(&shell, &cmd, argv, true);
        // The command echoes its argv (`argv[1..]` of the shell) and returns that count.
        let cmd_argv = &argv[1..];
        let expect: Vec<u8> = cmd_argv
            .iter()
            .flat_map(|a| [a.as_bytes(), b"\n"].concat())
            .collect();
        assert_eq!(
            iout, expect,
            "interp: command echoed its argv to the shell's sink"
        );
        assert_eq!(
            ic,
            cmd_argv.len() as i64,
            "interp: shell status = command argc"
        );
        assert_eq!(jout, iout, "jit: exec output must match interp");
        assert_eq!(jc, ic, "jit: exec status must match interp");
    }
}

/// Two children of one parent, two powerboxes: the child handed `stdout` resolves it and prints; its
/// twin, spawned from the same module with an empty grant list, cannot. Attenuation is the grant
/// list — nothing else distinguishes the two spawns. Identically on both backends.
#[test]
fn siblings_get_the_powerboxes_their_grant_lists_say() {
    let parent = parse_module_raw(&c_to_ir(&format!("{SPAWN_C}\n{TWO_CHILDREN}"), false))
        .expect("parse parent");
    let probe = parse_module_raw(&c_to_ir(PROBE, true)).expect("parse probe");
    verify_module(&probe).expect("verify probe");

    let (ic, iout) = run(&parent, &probe, &["p"], false);
    assert_eq!(iout, b"granted\n", "interp: only the granted child printed");
    assert_eq!(ic, 10, "interp: A (granted) = 1, B (not granted) = 0");
    let (jc, jout) = run(&parent, &probe, &["p"], true);
    assert_eq!(jout, iout, "jit: output must match interp");
    assert_eq!(jc, ic, "jit: status must match interp");
}
