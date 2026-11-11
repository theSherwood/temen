//! **The `exec` capability's wire protocol, its shared dispatch, and the deterministic scripted
//! backend.**
//!
//! This crate holds the wasm-safe half of the subprocess capability (EXEC.md): the op / argv / errno
//! protocol every backend speaks, the one dispatcher every host backend serves through ([`serve`]),
//! and the deterministic **scripted** backend ([`scripted_run`]): a `(argv-prefix → {stdout, stderr,
//! exit})` table, no processes at all. It depends only on `temen-interp`, so it builds for **wasm**.
//! `temen-run` keeps the real subprocess and child-domain backends (`crates/temen-run/src/exec.rs`)
//! and the `HostCap`s, and re-exports this crate so `temen_run::exec::*` is one surface.
//!
//! One interface, wirer-chosen backend (EXEC.md): a guest cannot tell which backend serves it,
//! a host one or a guest one. Two things make that true (#2962):
//!
//! - **One call form.** `exec` is a single entry, `exec(op, a, b, c) -> i64`, with the op in the
//!   first argument. A flat import of that shape binds the same way to a host procedure and to a
//!   guest's live offer, which a call naming the op in its immediate cannot (`call.cap HOST_PROC op`
//!   never reaches a guest servicer).
//! - **One data plane: the message region.** Every byte the protocol moves crosses a `SharedRegion`
//!   the wirer grants beside the slot, named [`EXEC_MSG`]. Offsets are region-relative, so a guest
//!   servicer, which cannot read its caller's window, reads the same bytes a host servicer does.
//!   A host servicer never touches the caller's window at all.
//!
//! v1 is **blocking one-shot**: `run` executes to completion with the given stdin;
//! `read_out`/`read_err` drain the captured output in chunks; `status` reports the exit code.
//! The contract *reserves* streaming (a future backend may yield output before exit).

use std::sync::Arc;
use temen_interp::{
    GuestMem, HostBuffer, HostCharge, HostProc, NodeRef, RegionBacking, RegionMinter, TABLE_ENTRY,
};
use temen_ir::errno::*;

/// `exec(RUN, off, argv_len, stdin_len) -> job | -errno` — spawn with the argv at `msg[off..]`
/// (NUL-separated, `argv[0]` the program) and the stdin right after it; v1 runs the job to
/// completion before returning. A program the capability does not permit (allowlist miss / no
/// table entry) is `-EPERM`, a refused op, never a trap.
pub const EXEC_RUN: i64 = 0;
/// `exec(READ_OUT, job, off, cap) -> n | 0 = EOF | -errno` — copy the next chunk of the job's
/// stdout, at most `cap` bytes, to `msg[off..]`.
pub const EXEC_READ_OUT: i64 = 1;
/// `exec(READ_ERR, job, off, cap) -> n | 0 = EOF | -errno` — the same for stderr.
pub const EXEC_READ_ERR: i64 = 2;
/// `exec(STATUS, job) -> exit code | -errno` — the job's exit code (POSIX-shell collapse on the
/// host backend: 0-255, signal death as 128+signo; verbatim on scripted/domain backends).
pub const EXEC_STATUS: i64 = 3;
/// `exec(CLOSE, job) -> 0 | -errno` — release the job handle.
pub const EXEC_CLOSE: i64 = 4;

/// The name the message region is granted under, beside the `exec` slot (#2962): the wirer grants
/// both, at the same time.
pub const EXEC_MSG: &str = "exec.msg";
/// The message region's size: a `run` whose argv and stdin together exceed it is `-E2BIG`; output
/// drains through it in chunks, so it never limits a job's output.
pub const EXEC_MSG_LEN: usize = 64 << 10;

/// The bytes `msg[off..off + len)`: `-EINVAL` for a negative offset or length, `-EFAULT` for a
/// range past the region.
fn msg_bytes(msg: &RegionBacking, off: i64, len: i64) -> Result<Vec<u8>, i64> {
    if off < 0 || len < 0 {
        return Err(EINVAL);
    }
    let mut bytes = vec![0u8; len as usize];
    if !msg.copy_out(off as u64, &mut bytes) {
        return Err(EFAULT);
    }
    Ok(bytes)
}

/// Parse the wire `argv`: NUL-separated byte strings, no trailing NUL required, each a non-empty
/// UTF-8 token. `argv[0]` is the program. Empty argv or a non-UTF-8 token is `-EINVAL`.
pub fn parse_argv(bytes: &[u8]) -> Result<Vec<String>, i64> {
    let argv: Vec<String> = bytes
        .split(|&b| b == 0)
        .filter(|t| !t.is_empty())
        .map(|t| String::from_utf8(t.to_vec()).map_err(|_| EINVAL))
        .collect::<Result<_, i64>>()?;
    if argv.is_empty() {
        return Err(EINVAL);
    }
    Ok(argv)
}

/// Decode `run`'s `(off, argv_len, stdin_len)` into `(argv, stdin)`, read from the message region.
/// A payload past the region's end is `-E2BIG`: the region is the bound on what one `run` carries.
pub fn run_args(args: &[i64], msg: &RegionBacking) -> Result<(Vec<String>, Vec<u8>), i64> {
    let a = |i: usize| args.get(i).copied().unwrap_or(0);
    let (off, argv_len, stdin_len) = (a(1), a(2), a(3));
    if off < 0 || argv_len < 1 || stdin_len < 0 {
        return Err(EINVAL);
    }
    let end = off
        .checked_add(argv_len)
        .and_then(|e| e.checked_add(stdin_len));
    if end.is_none_or(|e| e as u64 > msg.size()) {
        return Err(E2BIG);
    }
    let argv = parse_argv(&msg_bytes(msg, off, argv_len)?)?;
    let stdin = msg_bytes(msg, off + argv_len, stdin_len)?;
    Ok((argv, stdin))
}

/// One completed job: captured output with read cursors + the exit code. v1 jobs are complete
/// at creation (`run` is synchronous-to-completion), so reads only ever drain buffers. Its output
/// is charged to the table's payer until the job is closed (#2306, [`JobTable::output`]).
pub struct Job {
    pub stdout: HostBuffer,
    pub stderr: HostBuffer,
    pub out_pos: usize,
    pub err_pos: usize,
    pub exit: i64,
}

/// Shared per-grant job table + the non-`run` ops: everything except *how a job is produced*.
/// Every backend routes through it ([`serve`]), so read/status/close semantics are
/// backend-identical by construction.
pub struct JobTable {
    jobs: Vec<Option<Job>>,
    /// #2306 — the jobs the guest holds, a [`TABLE_ENTRY`] each, charged with their output to the
    /// domain the capability was granted to.
    slots: HostCharge,
}

impl JobTable {
    /// An empty table, whose jobs `payer` pays for.
    pub fn new(payer: &NodeRef) -> JobTable {
        JobTable {
            jobs: Vec::new(),
            slots: HostCharge::new(Some(payer.clone())),
        }
    }

    /// An empty buffer for a job's output, charged to the table's payer as it fills.
    pub fn output(&self) -> HostBuffer {
        HostBuffer::new(self.slots.payer().cloned())
    }

    /// A job's output holding `bytes`: `-ENOMEM` when the payer has no room for them.
    pub fn output_of(&self, bytes: &[u8]) -> Result<HostBuffer, i64> {
        let mut out = self.output();
        if out.push(bytes) {
            Ok(out)
        } else {
            Err(ENOMEM)
        }
    }

    /// A completed job holding `stdout`, `stderr` and `exit`: `-ENOMEM` when the payer has no room
    /// for its output.
    pub fn job_of(&self, stdout: &[u8], stderr: &[u8], exit: i64) -> Result<Job, i64> {
        Ok(Job {
            stdout: self.output_of(stdout)?,
            stderr: self.output_of(stderr)?,
            out_pos: 0,
            err_pos: 0,
            exit,
        })
    }

    /// Install a completed job, returning its handle: `-ENOMEM`, the job dropped, when the payer has
    /// no room for another.
    pub fn push(&mut self, job: Job) -> i64 {
        if !self.slots.resize(self.slots.bytes() + TABLE_ENTRY) {
            return ENOMEM;
        }
        match self.jobs.iter().position(Option::is_none) {
            Some(i) => {
                self.jobs[i] = Some(job);
                i as i64
            }
            None => {
                self.jobs.push(Some(job));
                (self.jobs.len() - 1) as i64
            }
        }
    }

    /// A non-`run` op (`read_out`/`read_err`/`status`/`close`), `args` the entry's `[op, job, …]`.
    fn handle(&mut self, args: &[i64], msg: &RegionBacking) -> i64 {
        let a = |i: usize| args.get(i).copied().unwrap_or(0);
        let job_idx = match a(1) {
            j if j >= 0 && (j as usize) < self.jobs.len() => j as usize,
            _ => return EBADF,
        };
        let Some(job) = self.jobs[job_idx].as_mut() else {
            return EBADF;
        };
        match a(0) {
            op @ (EXEC_READ_OUT | EXEC_READ_ERR) => {
                let (off, cap) = (a(2), a(3));
                if off < 0 || cap < 0 {
                    return EINVAL;
                }
                let (buf, pos) = if op == EXEC_READ_OUT {
                    (&job.stdout, &mut job.out_pos)
                } else {
                    (&job.stderr, &mut job.err_pos)
                };
                let n = (buf.len() - *pos).min(cap as usize);
                if n == 0 {
                    return 0; // EOF
                }
                if !msg.copy_in(off as u64, &buf[*pos..*pos + n]) {
                    return EFAULT;
                }
                *pos += n;
                n as i64
            }
            EXEC_STATUS => job.exit,
            EXEC_CLOSE => {
                self.jobs[job_idx] = None;
                self.slots.resize(self.slots.bytes() - TABLE_ENTRY);
                0
            }
            _ => EINVAL,
        }
    }
}

/// How a backend produces a job from a `run`'s argv and stdin: the one thing backends differ in.
/// The table is the grant's, for charging the job's output to its payer.
pub type RunJob = Box<dyn FnMut(Vec<String>, Vec<u8>, &JobTable) -> Result<Job, i64> + Send>;

/// The `exec` handler every host backend serves through: one entry (the call's own op must be 0;
/// the protocol's op is `args[0]`), every byte through `msg`, every non-`run` op through one
/// [`JobTable`] whose jobs `payer` pays for. Only `run_job` is the backend's.
pub fn serve(msg: RegionBacking, payer: &NodeRef, mut run_job: RunJob) -> HostProc {
    let mut jobs = JobTable::new(payer);
    Box::new(
        move |op: u32,
              args: &[i64],
              out: &mut [i64],
              _mem: Option<&mut dyn GuestMem>,
              _minter: Option<&mut dyn RegionMinter>| {
            let r = if op != 0 {
                EINVAL
            } else if args.first() == Some(&EXEC_RUN) {
                match run_args(args, &msg) {
                    Ok((argv, stdin)) => match run_job(argv, stdin, &jobs) {
                        Ok(job) => jobs.push(job),
                        Err(e) => e,
                    },
                    Err(e) => e,
                }
            } else {
                jobs.handle(args, &msg)
            };
            temen_interp::put_results(out, &[r])
        },
    )
}

/// One scripted entry: an argv **prefix** and the result any matching `run` yields.
#[derive(Clone)]
pub struct ScriptedEntry {
    pub argv_prefix: Vec<String>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit: i64,
}

/// The deterministic **scripted** backend's [`RunJob`] (EXEC.md): `run` matches the argv against
/// the table (longest matching argv-prefix wins) and yields that entry's `{stdout, stderr, exit}`;
/// no host process is ever created. A miss is `-EPERM` (the same refused-op shape as a `host_exec`
/// allowlist miss, so a guest cannot tell backends apart by their failure mode). The table is
/// shared and immutable, so re-runs are deterministic.
pub fn scripted_run(table: Arc<Vec<ScriptedEntry>>) -> RunJob {
    Box::new(move |argv, _stdin, jobs| {
        let e = table
            .iter()
            .filter(|e| argv.starts_with(&e.argv_prefix))
            .max_by_key(|e| e.argv_prefix.len())
            .ok_or(EPERM)?;
        jobs.job_of(&e.stdout, &e.stderr, e.exit)
    })
}
