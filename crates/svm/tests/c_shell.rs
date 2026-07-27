//! A minimal **Stage-0 shell** (PROCESS.md §10 / S7) compiled through chibicc onto the POSIX
//! personality — a real read-eval loop over stdin with builtin commands, no `fork`/`exec`.
//!
//! This is the playground target in miniature: it proves a genuine command interpreter runs end to
//! end on `svm-posix` (the libc-as-host-caps personality), and it's the scaffold BusyBox `ash` slots
//! into once the fork/exec surface lands. The shell's libc calls reach the personality **by name**:
//! `write`/`read`/`exit` are *defined* by the guest shim (shadowing chibicc's Stream/Exit builtins,
//! S15b) and forward — fd preserved — to `__px_`-prefixed generic imports; `getcwd`/`chdir`/`getenv`
//! are ordinary generic imports. The linker maps each name to its interface `(HOST_FN, op)`
//! (`svm_ir::Resolved::Cap`, link-time symbol resolution); the guest discovers the granted handles
//! itself via `cap.self` reflection, so there is no positional powerbox anywhere.
//!
//! The shell runs either a **script from preloaded stdin** (the personality's `read(0, …)` drains it)
//! or a single `sh -c "<command>"` — its `argv` delivered by the personality's host-side argument
//! vector (`argc`/`argv`, the symmetric analogue of `getenv`). It reaches the fs surface too:
//! `ls` drives `opendir`/`readdir`. It runs on **both** backends under identical personalities,
//! asserting they agree on the captured stdout — a cross-backend differential.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_run::cap_thunk;
use svm_text::parse_module as parse_module_raw;
use svm_verify::verify_module;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Build the chibicc fork once per test binary.
fn chibicc() -> &'static Path {
    static CC: OnceLock<PathBuf> = OnceLock::new();
    CC.get_or_init(|| {
        let dir = repo_root().join("frontend/chibicc");
        let status = Command::new("make")
            .arg("-s")
            .current_dir(&dir)
            .status()
            .expect("run `make` to build the chibicc fork");
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}

/// Compile a C source string to text IR via the frontend.
fn c_to_ir(src: &str) -> String {
    c_to_ir_with(src, &[])
}

/// [`c_to_ir`] with extra chibicc `-cc1` flags (e.g. `--data-page 65536` for the 64 KiB-page browser).
fn c_to_ir_with(src: &str, extra: &[&str]) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_cshell_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();
    let mut args: Vec<&str> = vec!["-cc1", "--emit-ir"];
    args.extend_from_slice(extra);
    args.extend_from_slice(&[
        "-cc1-input",
        cfile.to_str().unwrap(),
        "-cc1-output",
        irfile.to_str().unwrap(),
        cfile.to_str().unwrap(),
    ]);
    let status = Command::new(chibicc())
        .args(&args)
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// Compile a C source string to text IR with the `--child-entry` spawnable §14 child ABI — how an
/// external command the shell `exec`s (STAGE1.md §5) is built.
fn c_to_ir_child(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_cshcmd_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();
    let status = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
            "--child-entry",
            "-cc1-input",
            cfile.to_str().unwrap(),
            "-cc1-output",
            irfile.to_str().unwrap(),
            cfile.to_str().unwrap(),
        ])
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc --child-entry failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// The op-13 named-grant hooks the JIT needs to spawn a separate-module child with a by-name powerbox.
fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

/// Link the shim's import names to their interfaces — link-time symbol resolution (the phase-4
/// linker-only `resolve_imports_with`; IMPORTS.md §2.5): `__px_*` names strip the prefix and map
/// through [`svm_posix::resolve`] to `(HOST_FN, op)`; `__spawn`/`__join` are the shell's own
/// `Instantiator` ops (13 / 1, STAGE1.md §5). No handle is baked at link: each lowered `cap.call`
/// dispatches on the guest's own handle operand, discovered at run time via
/// `__vm_cap_count`/`__vm_cap_at` reflection (§3c protection at the boundary, IMPORTS.md §2.3
/// dynamic mode).
fn link_shim(name: &str) -> Option<svm_ir::Resolved> {
    let cap = match name {
        "__spawn" => svm_ir::ResolvedCap { type_id: 6, op: 13 },
        "__join" => svm_ir::ResolvedCap { type_id: 6, op: 1 },
        // The ring-pipeline surface (STAGE1.md item 6): mint a region (`AddressSpace` op 5) and
        // alias/query it (`SharedRegion` ops 0/1/3) — the shell pumps stage-0 output into a mapped
        // ring; the `__stage` filter runner maps its granted rings the same way.
        "__as_region" => svm_ir::ResolvedCap { type_id: 5, op: 5 },
        "__rg_map" => svm_ir::ResolvedCap { type_id: 4, op: 0 },
        "__rg_unmap" => svm_ir::ResolvedCap { type_id: 4, op: 1 },
        "__rg_granule" => svm_ir::ResolvedCap { type_id: 4, op: 3 },
        n => svm_posix::resolve(n.strip_prefix("__px_")?)?,
    };
    Some(svm_ir::Resolved::Cap(cap))
}

/// The guest libc shim (guest code): standard libc names, adapting C's NUL-terminated `char*` calls
/// to the personality's explicit-length `(ptr, len)` ABI (POSIX.md §4). `write`/`read`/`exit` are
/// *defined* here so their definitions shadow chibicc's builtins (S15b).
const SHIM: &str = include_str!("../../svm-run/demos/shell/shim.c");

/// The SPSC byte ring over a mapped `SharedRegion` (guest code, STAGE1.md item 6) — shared verbatim
/// by the shell (the stage-0 producer side) and the `__stage` filter runner (both sides). Layout at
/// ring base `b`: `[0]` head (bytes produced), `[4]` tail (bytes consumed), `[8]` done (writer
/// finished), `[12]` rclosed (reader gone — SIGPIPE-lite, so `| head -n 1` never wedges its
/// producer), data at `[64, 64+rcap)`. `rcap` = map granule − 64, set by each side after mapping.
/// The producer parks on the **tail** word when full, the consumer parks on the **head** word when
/// empty (`__vm_wait32`/`__vm_notify` — real futexes, canonical keys across windows); waits carry a
/// 5 s timeout and bail after 6, so a lost wake poisons the status loudly instead of hanging.
const RING: &str = include_str!("../../svm-run/demos/shell/ring.c");

/// The Stage-0 shell itself (guest code). `run_line` first strips `< file`, `> file`, and `>> file`
/// redirects (pointing globals `in_fd`/`out_fd` at the targets via `open`, restored after), then
/// `exec_line` tokenizes the remainder into `argv[]`, sets a shell variable for a lone `NAME=VALUE`,
/// then expands `$NAME`/`$?` tokens (shell vars shadow the environment) and glob tokens (`*`/`?`
/// matched against the memfs, `dir/name` results, literal if no match) before running one builtin —
/// `echo`, `export`, `pwd`, `cd`, `cat`, `wc`, `grep` (`-v`/`-c`), `head`/`tail` (`-n N`), `sort`, `uniq`, `rm`,
/// `ls`, `true`/`false`, `test`/`[ … ]`, `exit`; unknown → `<cmd>: not found`. Every command yields an exit status
/// (`grep` no-match → 1, unknown → 127, `test` per its predicate); the last is kept in `last_status`
/// and surfaced as `$?`. The text filters (`cat`/`wc`/`grep`/`head`/`tail`) read a path arg or the
/// redirected `in_fd`; together with `>`/`>>` and `rm` (`unlink`) this exercises the real file
/// surface (`open`/`read`/`write`/`close`/`unlink`). `run_list` (splitting on `;`/`&&`/`||`, short-
/// circuiting on `$?`) sits above `run_pipeline` (splitting on `|`, staging each stage's stdout
/// through a memfs temp the next stage reads as stdin) above `run_line`. `run_top` routes a line to
/// the single-line `if COND; then …; [else …;] fi` construct (`run_if`) or to a command list. `main`
/// supports two invocations: `sh -c "…"` (read via the personality's `argc`/`argv`) runs one line;
/// otherwise it's a read-eval loop over stdin. `exit` calls the personality `exit`.
const SHELL_MAIN: &str = include_str!("../../svm-run/demos/shell/shell_main.c");

/// Compile the shell, grant the personality (with `stdin` preloaded as the script) on two identical
/// hosts, resolve libc by name, and run on **both** backends. `env` seeds the personality environment
/// and `files` seeds the memfs before the run. Returns each backend's captured stdout (asserted equal
/// for the differential).
fn run_shell(
    stdin: &[u8],
    env: &[(&str, &str)],
    files: &[&str],
    args: &[&str],
) -> (Vec<u8>, Vec<u8>) {
    run_shell_ex(stdin, env, files, args, &[])
}

/// As [`run_shell`], plus a **PATH registry** of external commands `(name, C source)`: each is compiled
/// `--child-entry`, granted as a `Module`, and registered so an unknown command name in the script is
/// `exec`'d as an external child (STAGE1.md §5) instead of `<cmd>: not found`. With no `cmds` (the
/// [`run_shell`] case) `exec_lookup` always misses, so the `not found` path is unchanged.
fn run_shell_ex(
    stdin: &[u8],
    env: &[(&str, &str)],
    files: &[&str],
    args: &[&str],
    cmds: &[(&str, &str)],
) -> (Vec<u8>, Vec<u8>) {
    let src = format!("{SHIM}\n{RING}\n{SHELL_MAIN}");
    let ir = c_to_ir(&src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1usize << raw.memory.expect("frontend declares a window").size_log2;

    // The external command `Module`s (shared by both hosts), compiled with the spawnable child ABI.
    let cmd_mods: Vec<(&str, svm_ir::Module)> = cmds
        .iter()
        .map(|&(name, csrc)| {
            // Phase 3: keep the manifest — the op-13 spawn binds the child's slots. (The `__stage`
            // ring runner needs no special linking: its region ops are chibicc `__vm_region_*`
            // builtins, inline `cap.call`s dispatched on the runtime-minted region handles.)
            let m = parse_module_raw(&c_to_ir_child(csrc)).expect("parse cmd");
            verify_module(&m).expect("verify cmd");
            // The shell spawns `__stage` children with `size_log2 = 18` (a carve must equal the
            // child's declared memory); the runner pins itself there with a pad, and this assert
            // makes any drift loud instead of a probeable-but-silent spawn `-EINVAL`.
            if name == "__stage" {
                assert_eq!(
                    m.memory.map(|mm| mm.size_log2),
                    Some(18),
                    "__stage runner must declare memory 18 (the spawn's carve size)"
                );
            }
            (name, m)
        })
        .collect();

    // Grant a personality + the spawn caps on one host; identical grant order across the two hosts keeps
    // the handles equal (so the guest's reflection scan discovers the same handles on both, keeping the
    // differential exact). The `Instantiator` (over the whole window) and a
    // forwardable stdout `Stream` back the shell's `__spawn`/`exec_stdout`; the personality's fd-1 writes
    // route to the same shared sink as the child's re-granted `Stream`, unifying their output.
    let setup = |host: &mut Host| -> (svm_posix::Posix, i32, i32) {
        // Ring pipelines (STAGE1.md item 6): regions minted by the shell are real OS shared-memory
        // objects, so the JIT (parent map + child maps) gets hardware aliasing; the interpreter's
        // software aliasing is backing-agnostic, so both hosts get the same factory (differential
        // symmetry).
        host.set_region_factory(svm_run::new_shared_region);
        let sink = host.shared_stdout();
        let out_h = host.grant_stream(StreamRole::Out);
        let inst_h = host.grant_instantiator(0, win as u64);
        let _as_h = host.grant_address_space(0, win as u64);
        let cmd_handles: Vec<(&str, i32)> = cmd_mods
            .iter()
            .map(|(n, m)| (*n, host.grant_module(m)))
            .collect();
        // The shell never `malloc`s, so the personality heap (top 64 KiB) is never touched — it just
        // stays clear of the command carve (inside `pool`, low) and the shell's stack.
        let (px_h, posix) =
            svm_posix::grant(host, (win - (64 << 10)) as u64, win as u64, stdin.to_vec());
        posix.set_stdout_sink(sink);
        posix.set_exec_stdout(out_h);
        for (n, h) in &cmd_handles {
            posix.register_command(n, *h);
        }
        for (k, v) in env {
            posix.set_env(k, v);
        }
        for path in files {
            posix.write_file(path, b"");
        }
        if !args.is_empty() {
            posix.set_args(args);
        }
        (posix, px_h, inst_h)
    };

    let mut ih = Host::new();
    let (iposix, ipx, iinst) = setup(&mut ih);
    let mut jh = Host::new();
    let (jposix, jpx, jinst) = setup(&mut jh);
    assert_eq!(
        (ipx, iinst),
        (jpx, jinst),
        "identical grant order → identical handles"
    );

    let m = svm_ir::resolve_imports_with(&raw, link_shim)
        .unwrap_or_else(|e| panic!("resolve imports: {e:?}\n--- IR ---\n{ir}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    let init = vec![0u8; win];

    // Interpreter: the shell loops to EOF and returns 0 (or `exit`s, a `Trap::Exit`). The reserved
    // window backs the command carve op 13 spawns into.
    let mut fuel = 200_000_000u64;
    match run_capture_reserved_with_host(&m, 0, &[], &mut fuel, &init, 0, &mut ih).0 {
        Ok(_) | Err(Trap::Exit(_)) => {}
        Err(e) => panic!(
            "interp trapped: {e:?}\n--- stdout so far ---\n{}\n--- IR (head) ---\n{}",
            String::from_utf8_lossy(&iposix.stdout()),
            &ir[..ir.len().min(400)]
        ),
    }
    // JIT — given the module resolver + named-grant hooks op 13 needs.
    let (jout, _) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[],
        &init,
        0,
        cap_thunk,
        &mut jh as *mut Host as *mut c_void,
        Some(svm_run::module_resolver),
        Some(grant_hooks()),
    )
    .expect("jit compiles");
    assert!(
        matches!(jout, JitOutcome::Returned(_) | JitOutcome::Exited(_)),
        "jit ended abnormally: {jout:?}\n--- IR ---\n{ir}"
    );
    (iposix.stdout(), jposix.stdout())
}

/// The headline milestone: a real script runs through the shell loop end to end on the personality,
/// identically on both backends. `echo` (literal + `$VAR`), `pwd` after `cd`, and an unknown command
/// — every line's output is the personality's captured stdout.
#[test]
fn stage0_shell_runs_a_script() {
    let script = b"echo hello, shell\n\
                   echo $HOME\n\
                   cd /tmp\n\
                   pwd\n\
                   frobnicate\n\
                   exit\n";
    let (iout, jout) = run_shell(script, &[("HOME", "/root")], &[], &[]);
    assert_eq!(
        iout, b"hello, shell\n/root\n/tmp\nfrobnicate: not found\n",
        "interp: the shell ran the script (echo, $VAR, cd+pwd, unknown cmd)"
    );
    assert_eq!(jout, iout, "jit: shell output must match interp");
}

/// The `__stage` filter runner (guest code, STAGE1.md item 6) — the program a ring pipeline spawns
/// once per stage after the first. An ordinary `--child-entry` command: it discovers its grants by
/// `cap.self` reflection (regions in grant order — input ring first, output ring second when
/// present; a `Stream` = the shell's stdout), maps each ring into its **own** window
/// (`SharedRegion` op 0 — real aliasing on the JIT, separate windows per stage), and runs the one
/// filter `argv[0]` names, reading its input ring and writing its output ring — or, as the final
/// stage, the granted stdout. Statuses match the shell's builtins (`grep` no-match → 1); a wait
/// bail poisons the status to 99. It holds no memfs/personality capability at all, which is exactly
/// why only pure filters ride the ring path.
const STAGE_RUNNER_MAIN: &str = include_str!("../../svm-run/demos/shell/stage_runner_main.c");

/// The complete `__stage` runner source: the shared ring protocol + the filter dispatch.
fn stage_runner_src() -> String {
    format!("{RING}\n{STAGE_RUNNER_MAIN}")
}

/// An external command: echo every `argv[i]` on its own line, return `argc` (a non-zero status that
/// tracks the argument count, so `$?` is observable).
const CMD_ECHO: &str = r#"
long write(long fd, void *buf, long n);
static long slen(char *s){ long n=0; while(s[n]) n++; return n; }
int main(int argc, char **argv){
  for (int i = 0; i < argc; i++){ write(1, argv[i], slen(argv[i])); write(1, "\n", 1); }
  return argc;
}
"#;

/// An external command that succeeds: print `ok\n`, return `0` — so `&&`/`||` see a success status.
const CMD_OK: &str = r#"
long write(long fd, void *buf, long n);
int main(int argc, char **argv){ write(1, "ok\n", 3); return 0; }
"#;

/// STAGE1.md §5 — the real Stage-0 shell **spawns an external command**. A command name that is not a
/// builtin is looked up in the personality's PATH registry and, if found, run as a separate compiled-C
/// child via `Instantiator` op 13 + `join`: its `argv` is delivered, its stdout interleaves with the
/// shell's own output in the one shared sink, and its status threads into `$?`. An unregistered name is
/// still `<cmd>: not found` (status 127). Differential interp==JIT.
#[test]
fn stage0_shell_spawns_external_command() {
    let script = b"echo start\n\
                   say hi there\n\
                   echo rc $?\n\
                   bogus\n\
                   echo rc $?\n";
    let (iout, jout) = run_shell_ex(script, &[], &[], &[], &[("say", CMD_ECHO)]);
    assert_eq!(
        iout,
        b"start\nsay\nhi\nthere\nrc 3\nbogus: not found\nrc 127\n".as_slice(),
        "interp: builtin + spawned external (argv echoed, status = argc) + not-found, all in one sink"
    );
    assert_eq!(jout, iout, "jit: shell output must match interp");
}

/// A spawned command's status participates in `&&`/`||` short-circuiting exactly like a builtin's, and
/// the PATH registry holds more than one command. `ok` returns 0 (success); `say` returns its argc
/// (non-zero, a failure). Differential interp==JIT.
#[test]
fn stage0_shell_external_command_status_in_control_flow() {
    let script = b"ok && echo yes\n\
                   say a || echo fallback\n\
                   ok || echo skipped\n";
    let (iout, jout) = run_shell_ex(script, &[], &[], &[], &[("say", CMD_ECHO), ("ok", CMD_OK)]);
    assert_eq!(
        iout,
        b"ok\nyes\nsay\na\nfallback\nok\n".as_slice(),
        "interp: `ok`(0)&&echo → yes; `say a`(2, fail)||echo → fallback; `ok`(0)||echo → skipped"
    );
    assert_eq!(jout, iout, "jit: shell output must match interp");
}

/// EOF (no trailing `exit`) cleanly ends the loop — the personality's `read(0, …)` returns `0` at the
/// end of the preloaded script, and `main` returns. Also checks a bare `pwd` at the default cwd `/`.
#[test]
fn stage0_shell_handles_eof_and_default_cwd() {
    let (iout, jout) = run_shell(b"pwd\necho done", &[], &[], &[]);
    assert_eq!(
        iout, b"/\ndone\n",
        "interp: default cwd is / then echo, then EOF ends it"
    );
    assert_eq!(jout, iout, "jit: must match interp");
}

/// The `ls` builtin drives the personality's `opendir`/`readdir`/`closedir` from compiled C: with a
/// memfs staged, `ls /tmp` lists the immediate children (files and the subdir once), sorted; `ls` of
/// a missing directory reports `not found`. Proves the fs-metadata surface (S7 item 2) end to end.
#[test]
fn stage0_shell_ls_lists_a_directory() {
    let (iout, jout) = run_shell(
        b"ls /tmp\nls /nope\n",
        &[],
        &["/tmp/a.txt", "/tmp/b.txt", "/tmp/sub/c"],
        &[],
    );
    assert_eq!(
        iout, b"a.txt\nb.txt\nsub\n/nope: not found\n",
        "interp: ls lists sorted children (subdir once), then a miss"
    );
    assert_eq!(jout, iout, "jit: ls output must match interp");
}

/// `sh -c "<command>"` — the standard non-interactive shell invocation, delivered through the
/// personality's host-side argument vector (`argc`/`argv`, S7 item 1). No stdin script; the command
/// comes from `argv[2]`. Runs one line (`echo $HOME`) and returns, differential on both backends.
#[test]
fn stage0_shell_dash_c_runs_one_command() {
    let (iout, jout) = run_shell(
        b"", // no stdin script — the command is in argv
        &[("HOME", "/home/user")],
        &[],
        &["sh", "-c", "echo $HOME"],
    );
    assert_eq!(iout, b"/home/user\n", "interp: sh -c ran the argv command");
    assert_eq!(jout, iout, "jit: sh -c output must match interp");
}

/// I/O redirection + `cat` end to end (S7 item 3): `echo … > f` opens/truncates a memfs file and
/// writes there instead of stdout (so the redirected lines are absent from captured stdout); `>>`
/// appends; `cat f` reads it back to stdout. Only the final `cat`s reach stdout, proving the
/// `open`/`write`/`read`/`close` round-trip through the personality on both backends.
#[test]
fn stage0_shell_redirection_and_cat() {
    let (iout, jout) = run_shell(
        b"echo first > /out\n\
          echo second >> /out\n\
          cat /out\n\
          echo only-stdout\n\
          cat /missing\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"first\nsecond\nonly-stdout\n/missing: not found\n",
        "interp: `>`/`>>` divert to the file; only the cats + bare echo hit stdout"
    );
    assert_eq!(jout, iout, "jit: redirection/cat output must match interp");
}

/// A truncating redirect (`>`) replaces the file's prior contents rather than appending: after two
/// separate `>` writes, `cat` sees only the second. Confirms `O_TRUNC` on re-open.
#[test]
fn stage0_shell_redirect_truncates() {
    let (iout, jout) = run_shell(
        b"echo one > /f\n\
          echo two > /f\n\
          cat /f\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"two\n",
        "interp: the second `>` truncated the first write"
    );
    assert_eq!(jout, iout, "jit: truncation output must match interp");
}

/// Input redirection (`<`) + `wc` (S7 item 3, cont.): write a two-line file, then `wc < /f` reads it
/// through the redirected `in_fd` and reports `lines words bytes`; `cat < /f` streams the same file
/// back to stdout. Proves `<` binds a file to the command's input and that arg-less `cat`/`wc`
/// consume it. `wc` with an explicit path arg matches the redirected form.
#[test]
fn stage0_shell_input_redirection_and_wc() {
    let (iout, jout) = run_shell(
        b"echo hello world > /f\n\
          echo again >> /f\n\
          wc < /f\n\
          wc /f\n\
          cat < /f\n",
        &[],
        &[],
        &[],
    );
    // "hello world\nagain\n" = 2 lines, 3 words, 18 bytes.
    assert_eq!(
        iout, b"2 3 18\n2 3 18\nhello world\nagain\n",
        "interp: `< /f` feeds wc/cat; path arg and redirect agree"
    );
    assert_eq!(
        jout, iout,
        "jit: input-redirection/wc output must match interp"
    );
}

/// Both redirections at once: `wc < in > out` reads one file and writes the counts to another, so
/// nothing reaches stdout; `cat out` then reveals the diverted result. Exercises the multi-redirect
/// path in `run_line`.
#[test]
fn stage0_shell_input_and_output_redirection() {
    let (iout, jout) = run_shell(
        b"echo a b c > /in\n\
          wc < /in > /out\n\
          cat /out\n",
        &[],
        &[],
        &[],
    );
    // "a b c\n" = 1 line, 3 words, 6 bytes; the wc line itself is diverted to /out.
    assert_eq!(
        iout, b"1 3 6\n",
        "interp: wc's output went to /out, surfaced by cat"
    );
    assert_eq!(
        jout, iout,
        "jit: combined-redirection output must match interp"
    );
}

/// `grep` over a redirected file: only lines containing the pattern survive. Exercises the argv
/// tokenizer (pattern in argv[1], file via `<`) and line-buffered reading (`read_line`).
#[test]
fn stage0_shell_grep_filters_lines() {
    let (iout, jout) = run_shell(
        b"echo alpha > /f\n\
          echo beta >> /f\n\
          echo alps >> /f\n\
          grep al < /f\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"alpha\nalps\n",
        "interp: grep keeps lines containing `al`"
    );
    assert_eq!(jout, iout, "jit: grep output must match interp");
}

/// `grep -v` inverts the match and `grep -c` prints only the count. Both read the redirected file.
#[test]
fn stage0_shell_grep_flags() {
    let (iout, jout) = run_shell(
        b"echo alpha > /f\n\
          echo beta >> /f\n\
          echo alps >> /f\n\
          grep -v al < /f\n\
          grep -c al < /f\n",
        &[],
        &[],
        &[],
    );
    // -v al → lines without "al" → "beta"; -c al → count of matching lines → 2.
    assert_eq!(
        iout, b"beta\n2\n",
        "interp: grep -v inverts, grep -c counts"
    );
    assert_eq!(jout, iout, "jit: grep-flags output must match interp");
}

/// `head -n N` / `tail -n N` with an explicit path argument select the first / last N lines of a
/// six-line file. Exercises `-n` flag parsing (`atoi_`) and tail's ring buffer.
#[test]
fn stage0_shell_head_and_tail() {
    let (iout, jout) = run_shell(
        b"echo l1 > /f\n\
          echo l2 >> /f\n\
          echo l3 >> /f\n\
          echo l4 >> /f\n\
          echo l5 >> /f\n\
          echo l6 >> /f\n\
          head -n 2 /f\n\
          tail -n 2 /f\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"l1\nl2\nl5\nl6\n",
        "interp: head -n 2 → first two, tail -n 2 → last two"
    );
    assert_eq!(jout, iout, "jit: head/tail output must match interp");
}

/// `rm` removes a memfs file (`unlink`, op 8): after `rm /f`, `cat /f` reports not-found, and
/// removing an absent file reports not-found too. Multi-arg `rm` deletes each argument.
#[test]
fn stage0_shell_rm_removes_files() {
    let (iout, jout) = run_shell(
        b"echo x > /a\n\
          echo y > /b\n\
          rm /a /b\n\
          cat /a\n\
          rm /gone\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"/a: not found\n/gone: not found\n",
        "interp: rm unlinks; cat of a removed/absent file is not-found"
    );
    assert_eq!(jout, iout, "jit: rm output must match interp");
}

/// `echo` now joins multiple argv tokens with single spaces (argv tokenizer), collapsing the runs of
/// spaces in the source line. A `$VAR` token still expands mid-line.
#[test]
fn stage0_shell_echo_joins_argv() {
    let (iout, jout) = run_shell(
        b"echo  a   b    c\n\
          echo hi $WHO !\n",
        &[("WHO", "bob")],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"a b c\nhi bob !\n",
        "interp: argv tokens rejoin with single spaces; $WHO expands"
    );
    assert_eq!(jout, iout, "jit: echo-join output must match interp");
}

/// Exit status via `$?`: `true`/`false` set 0/1, an unknown command sets 127, `grep` with no match
/// sets 1, and `echo $?` reports the previous command's status. Proves `exec_line` returns a status
/// that `main` threads into `last_status`.
#[test]
fn stage0_shell_exit_status() {
    let (iout, jout) = run_shell(
        b"true\n\
          echo $?\n\
          false\n\
          echo $?\n\
          nope\n\
          echo $?\n\
          echo hit > /f\n\
          grep zzz < /f\n\
          echo $?\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"0\n1\nnope: not found\n127\n1\n",
        "interp: $? tracks true/false/unknown/grep-miss"
    );
    assert_eq!(jout, iout, "jit: exit-status output must match interp");
}

/// `test` / `[ … ]`: string equality, numeric comparison, and file/dir predicates over the memfs.
/// Each result is read back through `$?`.
#[test]
fn stage0_shell_test_builtin() {
    let (iout, jout) = run_shell(
        b"test a = a\n\
          echo $?\n\
          [ 3 -gt 5 ]\n\
          echo $?\n\
          echo hi > /f\n\
          test -f /f\n\
          echo $?\n\
          test -d /nodir\n\
          echo $?\n\
          [ -n hello ]\n\
          echo $?\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"0\n1\n0\n1\n0\n",
        "interp: test string/numeric/-f/-d/-n predicates"
    );
    assert_eq!(jout, iout, "jit: test-builtin output must match interp");
}

/// Command sequencing: `;` runs unconditionally; `&&` runs the next only after success; `||` only
/// after failure. Short-circuiting is driven by `$?` and threaded through `run_list`.
#[test]
fn stage0_shell_sequencing_and_short_circuit() {
    let (iout, jout) = run_shell(
        b"echo a ; echo b\n\
          true && echo yes\n\
          false && echo no\n\
          false || echo fallback\n\
          true || echo skip\n\
          false && echo x || echo y\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"a\nb\nyes\nfallback\ny\n",
        "interp: ; always, && on success, || on failure, with chaining"
    );
    assert_eq!(jout, iout, "jit: sequencing output must match interp");
}

/// Pipelines: a multi-stage `cat FILE | grep P | wc` streams each stage's full output into the next
/// via memfs temps, and the final stage's result reaches stdout. Also checks a per-stage redirect
/// inside a pipeline (`| grep P > out`) overrides the pipe. This is the shell's process-driven core
/// (emulated in-process, no fork yet).
#[test]
fn stage0_shell_pipelines() {
    let (iout, jout) = run_shell(
        b"echo apple > /f\n\
          echo apricot >> /f\n\
          echo banana >> /f\n\
          echo cherry >> /f\n\
          cat /f | grep ap | wc\n\
          cat /f | grep ap > /hits\n\
          cat /hits\n",
        &[],
        &[],
        &[],
    );
    // grep ap → "apple\napricot\n" (2 lines, 2 words, 14 bytes); the redirected pipeline writes the
    // same two lines to /hits, surfaced by cat.
    assert_eq!(
        iout, b"2 2 14\napple\napricot\n",
        "interp: pipeline stages chain; a stage redirect overrides the pipe"
    );
    assert_eq!(jout, iout, "jit: pipeline output must match interp");
}

/// A pipeline reading real (redirected) stdin at its head: `grep b < /f | wc -l`-style chain, here
/// `cat < /f | tail -n 1` — the first stage consumes the `<` file, the last emits to stdout.
#[test]
fn stage0_shell_pipeline_from_stdin_redirect() {
    let (iout, jout) = run_shell(
        b"echo one > /f\n\
          echo two >> /f\n\
          echo three >> /f\n\
          cat < /f | tail -n 1\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"three\n",
        "interp: `<` feeds stage 0; tail -n 1 ends the pipe"
    );
    assert_eq!(
        jout, iout,
        "jit: pipeline-from-stdin output must match interp"
    );
}

/// Shell variables: `NAME=VALUE` sets a shell var, `$NAME` (a whole token) expands it in any argument
/// position (not just echo), a shell var shadows an environment var of the same name, and a `$NAME`
/// RHS composes. An unset variable token expands to nothing (an empty line here).
#[test]
fn stage0_shell_variables() {
    let (iout, jout) = run_shell(
        b"X=hello\n\
          echo $X world\n\
          Y=$X\n\
          echo $Y\n\
          echo $UNSET\n\
          echo $X > /vf\n\
          cat /vf\n\
          WHO=shellvar\n\
          echo $WHO\n",
        &[("WHO", "envvar")],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"hello world\nhello\n\nhello\nshellvar\n",
        "interp: assignment, expansion everywhere, shadowing, unset->empty"
    );
    assert_eq!(jout, iout, "jit: variable output must match interp");
}

/// `export` promotes a shell variable into the personality environment (`setenv`). Both
/// `export NAME=VALUE` and `export NAME` (of an existing shell var) make the value observable —
/// expansion confirms it round-trips.
#[test]
fn stage0_shell_export_to_env() {
    let (iout, jout) = run_shell(
        b"export FOO=fooval\n\
          echo $FOO\n\
          BAR=barval\n\
          export BAR\n\
          echo $BAR\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"fooval\nbarval\n",
        "interp: export NAME=VALUE and export NAME both reach env"
    );
    assert_eq!(jout, iout, "jit: export output must match interp");
}

/// `sort` and `uniq` as a pipeline: `cat f | sort | uniq` orders the lines and collapses adjacent
/// duplicates — the canonical Unix idiom, here proving three-stage piping of real filters.
#[test]
fn stage0_shell_sort_uniq_pipeline() {
    let (iout, jout) = run_shell(
        b"echo banana > /f\n\
          echo apple >> /f\n\
          echo cherry >> /f\n\
          echo apple >> /f\n\
          echo banana >> /f\n\
          cat /f | sort | uniq\n",
        &[],
        &[],
        &[],
    );
    // sorted: apple, apple, banana, banana, cherry → uniq → apple, banana, cherry.
    assert_eq!(
        iout, b"apple\nbanana\ncherry\n",
        "interp: sort orders, uniq collapses adjacent dups, over a 3-stage pipe"
    );
    assert_eq!(jout, iout, "jit: sort/uniq output must match interp");
}

/// Globbing: `*` expands against the memfs into sorted `dir/name` matches, feeding multi-file
/// builtins (`echo`, `cat`, `rm`); a pattern with no match stays literal (nullglob-off). Exercises
/// `fnmatch_` + `glob_expand` driving `opendir`/`readdir`.
#[test]
fn stage0_shell_globbing() {
    let (iout, jout) = run_shell(
        b"echo one > /a1\n\
          echo two > /a2\n\
          echo three > /b1\n\
          echo /a*\n\
          cat /a*\n\
          echo /z*\n\
          rm /a*\n\
          cat /a1\n",
        &[],
        &[],
        &[],
    );
    // `/a*` → /a1 /a2 (sorted); cat concatenates both; `/z*` has no match so stays literal; rm /a*
    // removes both, so the final cat misses.
    assert_eq!(
        iout, b"/a1 /a2\none\ntwo\n/z*\n/a1: not found\n",
        "interp: glob expands, feeds cat/rm, and is literal on no match"
    );
    assert_eq!(jout, iout, "jit: globbing output must match interp");
}

/// Single-line `if/then/else/fi`: the condition's exit status picks the branch, both the taken and
/// not-taken branches behave, and multiple body commands run. Uses `test -f` over the memfs and a
/// multi-command then-body.
#[test]
fn stage0_shell_if_then_else() {
    let (iout, jout) = run_shell(
        b"echo hi > /f\n\
          if test -f /f; then echo present; echo again; else echo absent; fi\n\
          if test -f /nope; then echo present; else echo absent; fi\n\
          if false; then echo t; fi\n\
          if true; then echo taken; fi\n",
        &[],
        &[],
        &[],
    );
    assert_eq!(
        iout, b"present\nagain\nabsent\ntaken\n",
        "interp: if picks the branch by $?, runs multi-command bodies, no-else is a no-op"
    );
    assert_eq!(jout, iout, "jit: if/then/else output must match interp");
}

/// `if` composes with the rest of the shell: the condition can be a pipeline (`grep` sets the status)
/// and a body command can redirect. Proves `run_if` delegates each part back through `run_list`.
#[test]
fn stage0_shell_if_with_pipeline_condition() {
    let (iout, jout) = run_shell(
        b"echo apple > /f\n\
          echo banana >> /f\n\
          if cat /f | grep ban; then echo found > /r; else echo missing > /r; fi\n\
          cat /r\n",
        &[],
        &[],
        &[],
    );
    // grep prints its match (to stdout) and succeeds, so the then-branch writes "found" to /r.
    assert_eq!(
        iout, b"banana\nfound\n",
        "interp: pipeline condition drives if; redirected body writes the result"
    );
    assert_eq!(jout, iout, "jit: if-with-pipeline output must match interp");
}

/// STAGE1.md item 6 — the shell's `|` runs **concurrent** stages over `SharedRegion` rings. With the
/// `__stage` runner on PATH, every stage after the first is a spawned child of its own window; the
/// stages stream through ring futexes (stage 0 pumps from inside the shell, the last child writes
/// the granted stdout). The output is byte-identical to the sequential temp-file staging — the
/// concurrency is the point, not a semantics change. Differential interp==JIT.
#[test]
fn stage0_shell_pipeline_over_rings() {
    let runner = stage_runner_src();
    let script = b"echo b > f\n\
                   echo a >> f\n\
                   echo b >> f\n\
                   cat f | sort | uniq\n";
    let (iout, jout) = run_shell_ex(script, &[], &[], &[], &[("__stage", &runner)]);
    assert_eq!(
        iout,
        b"a\nb\n".as_slice(),
        "interp: cat f | sort | uniq over rings — three concurrent stages, sorted + deduped"
    );
    assert_eq!(jout, iout, "jit: ring-pipeline output must match interp");
}

/// The ring path end to end: a 4-stage pipeline (two ring→ring middles), `grep` status flowing into
/// `$?` from a ring child, `grep -c` counting, and `head`'s early exit closing its input ring so the
/// producer stops (the SIGPIPE-lite) instead of wedging to a timeout. Differential interp==JIT.
#[test]
fn stage0_shell_ring_pipeline_status_and_early_exit() {
    let runner = stage_runner_src();
    let script = b"echo one > f\n\
                   echo two >> f\n\
                   echo one >> f\n\
                   echo three >> f\n\
                   cat f | grep -v two | sort | uniq\n\
                   cat f | grep -c one\n\
                   cat f | grep zzz\n\
                   echo rc $?\n\
                   cat f | head -n 2\n\
                   echo rc $?\n";
    let (iout, jout) = run_shell_ex(script, &[], &[], &[], &[("__stage", &runner)]);
    assert_eq!(
        iout,
        b"one\nthree\n2\nrc 1\none\ntwo\nrc 0\n".as_slice(),
        "interp: 4-stage ring pipeline, -c count, no-match status 1, head early-exit status 0"
    );
    assert_eq!(jout, iout, "jit: ring-pipeline statuses must match interp");
}

/// The fallback contract: a later stage that the ring path cannot serve (a `>` redirect — the
/// runner has no filesystem) silently takes the sequential memfs-temp path with identical
/// semantics, and the two paths coexist in one script. Differential interp==JIT.
#[test]
fn stage0_shell_ring_pipeline_falls_back_on_redirect() {
    let runner = stage_runner_src();
    let script = b"echo x > f\n\
                   echo y >> f\n\
                   cat f | grep x > out\n\
                   cat out\n\
                   cat f | grep y\n";
    let (iout, jout) = run_shell_ex(script, &[], &[], &[], &[("__stage", &runner)]);
    assert_eq!(
        iout,
        b"x\ny\n".as_slice(),
        "interp: redirected stage falls back to temps (x lands in `out`), plain stage rides rings"
    );
    assert_eq!(jout, iout, "jit: fallback + ring outputs must match interp");
}

/// **Browser fixture generator** (run explicitly). Compiles the real shell (`shim + ring +
/// shell_main`), resolves its libc/personality imports by name (the same `link_shim` the differential
/// uses), verifies, and encodes it to `browser/tests/fixtures/shell.svmb` — the exact module bytes the
/// browser playground's POSIX-personality entry runs (STAGE1.md, playground-shell epic). `#[ignore]`d
/// because it writes into the tree and needs the chibicc build; regenerate with:
///   cargo test -p svm --test c_shell -- --ignored --exact gen_browser_shell_fixture
#[test]
#[ignore = "writes browser/tests/fixtures/shell.svmb; run explicitly to (re)generate the fixture"]
fn gen_browser_shell_fixture() {
    // The **sequential** subset (`SVM_SHELL_SEQUENTIAL`): no external-command spawn, no concurrent ring
    // pipelines — so the module carries no `Instantiator`/`SharedRegion` cap.calls and compiles on the
    // browser's bytecode engine (`compile_inst` rejects those; the tree-walk/JIT engines that run the
    // full shell use OS threads + a wall clock, absent under wasm). `RING` is dropped with it.
    let src = format!("#define SVM_SHELL_SEQUENTIAL 1\n{SHIM}\n{SHELL_MAIN}");
    // `--data-page 65536`: the playground runs on a 64 KiB wasm page, so the read-only string data
    // must share no host page with a writable global (else the shell's own write to a global faults
    // under D40). Native chibicc defaults to 16 KiB, which is why the differential above never hit it.
    let ir = c_to_ir_with(&src, &["--data-page", "65536"]);
    let raw = parse_module_raw(&ir).expect("parse shell IR");
    let m = svm_ir::resolve_imports_with(&raw, link_shim).expect("resolve shell imports");
    verify_module(&m).expect("verify shell");
    let bytes = svm_encode::encode_module(&m);
    let out = repo_root().join("browser/tests/fixtures/shell.svmb");
    std::fs::create_dir_all(out.parent().unwrap()).expect("create fixtures dir");
    std::fs::write(&out, &bytes).expect("write shell.svmb");
    eprintln!("wrote {} ({} bytes)", out.display(), bytes.len());
}
