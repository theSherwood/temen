//! The **`exec` capability** (EXEC.md): shell-out as granted, attenuated authority. One
//! interface a guest resolves by name (`"exec"`); *what a spawn is* — a real host process, a
//! scripted table entry, or (later) a child temen domain — is the wirer's choice, invisible to
//! the guest. This module holds the real-subprocess backend, [`host_exec`]: the capability
//! *is* the program allowlist, the way [`crate::fs::host_fs`]'s capability is the rooted
//! directory. The deterministic [`scripted_exec`] backend and the shared wire protocol live in
//! the wasm-safe `temen-exec` crate, re-exported here so `temen_run::exec::*` is one surface.
//!
//! No subprocess authority exists un-granted: without a grant, `self.resolve("exec")` is
//! negative and the guest's fallback runs. An allowlist miss is a refused op (`-EPERM`), never
//! a trap — the same failure shape as a `scripted_exec` table miss, so a guest cannot tell the
//! backends apart by how they refuse (interposition invisibility, per EXEC.md).

pub use temen_exec::*;
use temen_ir::errno::*; // the one shared negative-errno table (#905)

use crate::Descriptor;
use crate::{Backend, HostCap, IfaceShape, Instance, Limits, Outcome, RunConfig};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;
use temen_interp::{
    CapState, ForkedProc, HostBuffer, HostProcFork, HostProcGrant, ProcHandler, Value,
};

/// The `exec` interface's [`IfaceShape`], [`EXEC_IFACE`] parsed.
fn exec_shape() -> IfaceShape {
    EXEC_IFACE
        .iter()
        .fold(IfaceShape::new(), |shape, (name, sig)| {
            shape.op(
                *name,
                temen_text::parse_func_type(sig).expect("EXEC_IFACE parses"),
            )
        })
}

/// An `exec` [`HostCap`] whose jobs `runner` makes (one [`RunJob`] per handler it mints): a grouped
/// provider of [`EXEC_IFACE`], published on the handler as its vtable, so a guest's grouped import
/// binds it by op name, in a child as in the root. Its grant is the wirer's whole act (#2962): for
/// the slot `name` it mints the message region, charged to the grantee as the embedder's grant,
/// grants it as `<name>.msg`, and grants the handler, which serves every byte from the region
/// ([`serve`]) and never touches the guest's window. A `fork()` twin re-mints the handler over a
/// fresh job table and the same region (#2970). A region the host cannot make leaves the slot
/// unbound: its handle is the errno, so a resolve of the slot is negative and the guest's fallback
/// runs.
fn exec_cap(runner: impl Fn() -> RunJob + Send + Sync + 'static) -> HostCap {
    let runner = Arc::new(runner);
    let shape = exec_shape();
    let vtable: (Vec<String>, Vec<temen_ir::FuncType>) = shape.ops.iter().cloned().unzip();
    HostCap::iface_named(&shape, "exec", move |h, _, name| {
        let msg = match h.mint_region(EXEC_MSG_LEN, true) {
            Ok(msg) => msg,
            Err(errno) => return errno as i32,
        };
        let region = h.grant_shared_region_backed(Arc::clone(&msg));
        h.register_cap_name(&msg_name(name), region);
        let payer = h.own_node();
        let fork: HostProcFork = Arc::new({
            let (runner, msg, payer) = (Arc::clone(&runner), Arc::clone(&msg), payer.clone());
            move |_pid| {
                let handler = serve(Arc::clone(&msg), &payer, runner());
                // Each handler keeps a job table the guest drains output from: `Uncaptured` (#1699).
                Some(ForkedProc::shared(
                    ProcHandler::Sync(handler),
                    CapState::Uncaptured,
                ))
            }
        });
        let handler = serve(msg, &payer, runner());
        let h_exec = h.grant_host_proc(
            HostProcGrant::new(ProcHandler::Sync(handler), CapState::Uncaptured).forkable(fork),
        );
        h.set_host_proc_vtable(h_exec, vtable.0.clone(), vtable.1.clone());
        h_exec
    })
}

/// The deterministic **scripted** backend as a [`HostCap`]: a `(argv-prefix → {stdout, stderr,
/// exit})` table, no host processes. What differential tests and wasm/browser embedders grant.
pub fn scripted_exec(table: Vec<ScriptedEntry>) -> HostCap {
    let table = Arc::new(table);
    exec_cap(move || scripted_run(Arc::clone(&table)))
}

/// The **real subprocess** backend: spawn via the host, attenuated by an explicit program
/// **allowlist** — `run`'s `argv[0]` must match an entry exactly, or the op is refused
/// (`-EPERM`). An **empty allowlist means "any program"** and is a choice the embedder must
/// spell out at the grant site. v1 is blocking one-shot: `run` feeds `stdin`, waits for exit,
/// and captures stdout/stderr; reads then drain the captured bytes. The exit code is collapsed
/// the POSIX-shell way (0-255; signal death as `128 + signo`).
pub fn host_exec(allowlist: &[&str]) -> HostCap {
    let allow: Arc<Vec<String>> = Arc::new(allowlist.iter().map(|s| s.to_string()).collect());
    exec_cap(move || {
        let allow = Arc::clone(&allow);
        Box::new(move |argv, stdin, jobs| {
            if !allow.is_empty() && !allow.iter().any(|a| a == &argv[0]) {
                return Err(EPERM); // outside the allowlist: refused, not a trap
            }
            spawn_blocking(&argv, &stdin, jobs)
        })
    })
}

/// One program in a [`domain_exec`] registry: the name a `run`'s `argv[0]` selects (exact
/// match — the registry *is* the attenuation, like `host_exec`'s allowlist), the instantiated
/// module to run, and the per-run [`Limits`] (a fuel bound is what makes a domain run
/// deterministic — EXEC.md).
#[derive(Clone)]
pub struct DomainProgram {
    pub name: String,
    pub instance: Arc<Instance>,
    pub limits: Limits,
}

/// The **child-domain** backend (EXEC.md's third row): a spawn is a fresh temen domain — its own
/// window, powerbox, and fuel — run to completion with the wire `stdin` seeded and both output
/// streams captured, no OS process anywhere. `argv[0]` resolves through the program registry
/// (a miss is `-EPERM`, the same refusal shape as an allowlist or table miss); the full argv
/// rides the §3e args buffer, so a `main(int, char**)` program reads it exactly as it would
/// standalone. The exit code is the child's entry result **verbatim** (`Exit(code)` or the
/// first returned value; nothing collapsed) per the contract's exit-code table; a child that
/// **traps** is a failed `run` (`-EINVAL`, probeable) — v1 does not invent an exit code for a
/// crash. Runs on the reference interpreter (deterministic under fuel; a serving module would
/// fold there anyway). Blocking profile: v1 `run` executes the child inline, exactly as
/// `host_exec` blocks on the process — one synchronous op either way.
pub fn domain_exec(programs: Vec<DomainProgram>) -> HostCap {
    domain_exec_inner(programs, None)
}

/// Like [`domain_exec`], but each spawned child additionally inherits `child_fs` — an explicitly
/// provided `fs` capability, granted to the child under the name `"fs"`. Back it with a
/// [`crate::fs::mem_fs_shared_factory`]-derived cap and every phase shares **one** working
/// filesystem, so phase N writes `x.nif` and phase N+1 reads it — the faithful file-based `nifmake`
/// hand-off (NIM.md §3c). This is a **deliberate widening of a child's authority** and lives at the
/// grant site by design: plain [`domain_exec`] gives a child only stdin/stdout, and a child never
/// gains `fs` unless the embedder hands it one here. Even then the child gets *only* this fs (the
/// in-memory, seeded store the cap wraps) — no host filesystem, no ambient authority; a child cannot
/// widen its own grant. The children share the store the caller chose to share; isolation between
/// unrelated pipelines is the caller granting them *different* factories.
pub fn domain_exec_with_fs(programs: Vec<DomainProgram>, child_fs: HostCap) -> HostCap {
    domain_exec_inner(programs, Some(child_fs))
}

fn domain_exec_inner(programs: Vec<DomainProgram>, child_fs: Option<HostCap>) -> HostCap {
    let programs = Arc::new(programs);
    let child_fs = Arc::new(child_fs);
    exec_cap(move || {
        let (programs, child_fs) = (Arc::clone(&programs), Arc::clone(&child_fs));
        Box::new(move |argv, stdin, jobs| {
            domain_run(&programs, (*child_fs).as_ref(), argv, stdin, jobs)
        })
    })
}

fn domain_run(
    programs: &[DomainProgram],
    child_fs: Option<&HostCap>,
    argv: Vec<String>,
    stdin: Vec<u8>,
    jobs: &JobTable,
) -> Result<Job, i64> {
    let Some(p) = programs.iter().find(|p| p.name == argv[0]) else {
        return Err(EPERM); // outside the registry: refused, not a trap
    };
    let config = RunConfig {
        limits: p.limits.clone(),
        stdin,
        memory_size_log2: None,
        args: argv.into_iter().map(String::into_bytes).collect(),
        env: Vec::new(),
        ..RunConfig::default()
    };
    // The child inherits only what the parent granted: stdin/stdout always, and `fs` iff the embedder
    // built this backend with `domain_exec_with_fs`. A fresh grant from the (shared) fs cap per spawn,
    // so all phases reach one store.
    let child_run = match child_fs {
        Some(fs) => p
            .instance
            .run_with_caps(Backend::TreeWalk, &config, &[("fs", fs.clone())]),
        None => p.instance.run(Backend::TreeWalk, &config),
    };
    // A child that trapped or ran out of fuel is a failed run, probeable.
    let run = child_run.map_err(|_| EINVAL)?;
    let exit = match run.outcome {
        Outcome::Exited(c) => c as i64,
        Outcome::Returned(vals) => vals.first().map_or(0, |v| match v {
            Value::I64(x) => *x,
            Value::I32(x) => *x as i64,
            _ => 0,
        }),
    };
    jobs.job_of(&run.stdout, &run.stderr, exit)
}

/// Read `from` to its end into `into`: `false`, having stopped, once `into`'s payer has no room for
/// more (#2306). A read error ends the stream, as its end would.
fn drain(from: Option<impl Read>, into: &mut HostBuffer) -> bool {
    let Some(mut from) = from else {
        return true;
    };
    let mut piece = [0u8; 8192];
    loop {
        match from.read(&mut piece) {
            Ok(0) => return true,
            Ok(n) if !into.push(&piece[..n]) => return false,
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return true,
        }
    }
}

/// Run `argv` to completion with `stdin` fed to it; capture both output streams and collapse
/// the wait status the POSIX-shell way. #2306: the output is charged as it arrives to the payer of
/// `jobs`, and a child that writes more than it has room for is killed and its run refused
/// (`-ENOMEM`).
fn spawn_blocking(argv: &[String], stdin: &[u8], jobs: &JobTable) -> Result<Job, i64> {
    // #2306 — the three pipes to the child are this process's descriptors while it runs.
    let _held = [
        Descriptor::take()?,
        Descriptor::take()?,
        Descriptor::take()?,
    ];
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| ENOENT)?;
    // Feed stdin and close it so the child sees EOF. v1 payloads are one-shot and bounded by
    // the guest window, so a straight write before reaping is fine (a child that blocks writing
    // output before draining stdin could wedge only if the payload exceeds the OS pipe buffer —
    // acceptable for v1's shell-out shape; incremental stdin is the reserved extension).
    if let Some(mut si) = child.stdin.take() {
        let _ = si.write_all(stdin);
    }
    let (mut stdout, mut stderr) = (jobs.output(), jobs.output());
    let err = child.stderr.take();
    let reading = std::thread::spawn(move || (drain(err, &mut stderr), stderr));
    let out_fits = drain(child.stdout.take(), &mut stdout);
    if !out_fits {
        let _ = child.kill();
    }
    let (err_fits, stderr) = reading.join().map_err(|_| EINVAL)?;
    let status = child.wait().map_err(|_| EINVAL)?;
    if !(out_fits && err_fits) {
        return Err(ENOMEM);
    }
    let exit = {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            match (status.code(), status.signal()) {
                (Some(c), _) => (c & 0xff) as i64,
                (None, Some(sig)) => 128 + sig as i64,
                (None, None) => EINVAL,
            }
        }
        #[cfg(not(unix))]
        {
            status.code().map_or(EINVAL, |c| c as i64)
        }
    };
    Ok(Job {
        stdout,
        stderr,
        out_pos: 0,
        err_pos: 0,
        exit,
    })
}
