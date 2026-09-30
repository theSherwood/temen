//! temen process spawning: `std::process::Command` over the POSIX personality's `posix_spawn`
//! (temen-posix `OP_PSPAWN`). A child is a real process running beside its parent: its program starts
//! at its entry in a fresh window, over the parent's descriptors as the spawn's file actions leave them,
//! and the parent reaps it with `OP_WAITPID`. The program is a path the embedder registered, or a
//! program the process built (under a `ModuleLoader`); a name without a `/` is looked up along `PATH`,
//! as `posix_spawnp` does.
//!
//! Stdio: a child stream becomes the file action that sets the child's descriptor up — a `dup2` from a
//! fresh pipe's end, a `ChildPipe`, a file, or the parent's stdout/stderr — or none, for an inherited
//! one. Pipes are core pipes, so a parent can stream into a live child's stdin and read its output to
//! EOF. The memfs has no `/dev/null`: a null stdin is a pipe nobody writes (EOF at once), and a null
//! stdout or stderr is closed in the child (its writes fail with `EBADF`).
//!
//! Where it differs from unix: the personality has no close-on-exec, so a child inherits every
//! descriptor its parent holds but the ones its file actions close — a pipe end the parent holds for
//! another live child keeps that pipe open until this child exits too. An exec carries a process's
//! environment, so `Command::env` changes do not reach the child. And `output` reads the child's
//! stdout to EOF before its stderr: a child that fills its stderr pipe (64 KiB) before it closes its
//! stdout waits on its parent, which waits on it.
#![deny(unsafe_op_in_unsafe_fn)]
use super::env::{CommandEnv, CommandEnvs, CommandResolvedEnvs};
pub use crate::ffi::OsString as EnvKey;
use crate::ffi::{OsStr, OsString};
use crate::num::NonZero;
use crate::path::Path;
use crate::process::StdioPipes;
use crate::sys::fs::File;
use crate::sys::pal::host;
use crate::sys::pipe::{self, Pipe};
use crate::sys::unsupported_err;
use crate::{fmt, io};

pub type ChildPipe = Pipe;

/// temen-posix's `pspawn` file actions: close `fd`; `dup2(fd, arg)`; change the working directory to
/// `path`.
const PSPAWN_CLOSE: u64 = 1;
const PSPAWN_DUP2: u64 = 2;
const PSPAWN_CHDIR: u64 = 3;

const ENOENT: i64 = -2;
const ENOSYS: i64 = -38;
/// `waitpid`'s option: answer `0` for a child still running rather than wait for it.
const WNOHANG: i64 = 1;
const SIGKILL: i64 = 9;

/// Map a negative errno from a spawn/wait op to an `io::Error`: `-ENOENT` (nothing at any path the
/// program names) and `-ENOSYS` (no `posix_spawn` on this route) get the kinds programs match; the
/// rest fall back to the raw code.
fn err(code: i64) -> io::Error {
    match code {
        ENOENT => io::const_error!(io::ErrorKind::NotFound, "spawn: no such program"),
        ENOSYS => io::const_error!(io::ErrorKind::Unsupported, "spawn: posix_spawn is unavailable here"),
        _ => io::Error::from_raw_os_error((-code) as i32),
    }
}

/// `bytes` NUL-terminated, as the spawn's paths and argv strings are.
fn cstr(bytes: &[u8]) -> io::Result<Vec<u8>> {
    if bytes.contains(&0) {
        return Err(io::const_error!(io::ErrorKind::InvalidInput, "nul byte found in provided data"));
    }
    let mut v = Vec::with_capacity(bytes.len() + 1);
    v.extend_from_slice(bytes);
    v.push(0);
    Ok(v)
}

/// The paths a spawn of `program` tries, in order: `program` itself when it holds a `/`, else each
/// `PATH` directory joined with it (`/bin:/usr/bin` without a `PATH`), as `posix_spawnp` looks it up.
fn candidates(program: &[u8]) -> Vec<Vec<u8>> {
    if program.contains(&b'/') {
        return vec![program.to_vec()];
    }
    let path = crate::env::var_os("PATH");
    let dirs = path.as_ref().map_or(&b"/bin:/usr/bin"[..], |p| p.as_encoded_bytes());
    dirs.split(|&b| b == b':')
        .map(|dir| {
            let mut p = if dir.is_empty() { b".".to_vec() } else { dir.to_vec() };
            p.push(b'/');
            p.extend_from_slice(program);
            p
        })
        .collect()
}

////////////////////////////////////////////////////////////////////////////////
// Command — the platform-agnostic builder (mirrors the `unsupported` PAL).
////////////////////////////////////////////////////////////////////////////////

pub struct Command {
    program: OsString,
    args: Vec<OsString>,
    env: CommandEnv,

    cwd: Option<OsString>,
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
}

#[derive(Debug)]
pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
    ParentStdout,
    ParentStderr,
    InheritFile(File),
    Fd(Pipe),
}

impl Command {
    pub fn new(program: &OsStr) -> Command {
        Command {
            program: program.to_owned(),
            args: vec![program.to_owned()],
            env: Default::default(),
            cwd: None,
            stdin: None,
            stdout: None,
            stderr: None,
        }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(arg.to_owned());
    }

    pub fn env_mut(&mut self) -> &mut CommandEnv {
        &mut self.env
    }

    pub fn cwd(&mut self, dir: &OsStr) {
        self.cwd = Some(dir.to_owned());
    }

    pub fn stdin(&mut self, stdin: Stdio) {
        self.stdin = Some(stdin);
    }

    pub fn stdout(&mut self, stdout: Stdio) {
        self.stdout = Some(stdout);
    }

    pub fn stderr(&mut self, stderr: Stdio) {
        self.stderr = Some(stderr);
    }

    pub fn get_program(&self) -> &OsStr {
        &self.program
    }

    pub fn get_args(&self) -> CommandArgs<'_> {
        let mut iter = self.args.iter();
        iter.next();
        CommandArgs { iter }
    }

    pub fn get_envs(&self) -> CommandEnvs<'_> {
        self.env.iter()
    }

    pub fn get_env_clear(&self) -> bool {
        self.env.does_clear()
    }

    pub fn get_resolved_envs(&self) -> CommandResolvedEnvs {
        CommandResolvedEnvs::new(self.env.capture())
    }

    pub fn get_current_dir(&self) -> Option<&Path> {
        self.cwd.as_ref().map(|cs| Path::new(cs))
    }

    /// Start the command as a new process, returning it and the parent's ends of whatever pipes its
    /// stdio asked for. `default` fills in an unset stdout/stderr disposition (`MakePipe` for `output`,
    /// `Inherit` for `status`/`spawn`), and an unset stdin too when `needs_stdin`; an unset stdin is
    /// null otherwise, as on unix.
    pub fn spawn(
        &mut self,
        default: Stdio,
        needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        if !host::have_posix() {
            return Err(unsupported_err());
        }
        let null = Stdio::Null;
        let stdin = self.stdin.as_ref().unwrap_or(if needs_stdin { &default } else { &null });
        let stdout = self.stdout.as_ref().unwrap_or(&default);
        let stderr = self.stderr.as_ref().unwrap_or(&default);

        // The file actions, applied in order to the child's copy of this process's descriptors: each
        // stream's `dup2` (or close), then the close of every end of this spawn's pipes, which the
        // child holds as 0–2 now. (A process's 0–2 are always open — std gives no way to close them —
        // so a pipe end is never one of them.)
        let mut actions: Vec<[u64; 4]> = Vec::new();
        let streams = [
            ChildStream::setup(stdin, 0, &mut actions)?,
            ChildStream::setup(stdout, 1, &mut actions)?,
            ChildStream::setup(stderr, 2, &mut actions)?,
        ];
        for (ours, theirs) in streams.iter().filter_map(|s| s.pipe.as_ref()) {
            actions.push([PSPAWN_CLOSE, ours.fd() as u64, 0, 0]);
            actions.push([PSPAWN_CLOSE, theirs.fd() as u64, 0, 0]);
        }
        let cwd = match &self.cwd {
            Some(dir) => Some(cstr(dir.as_encoded_bytes())?),
            None => None,
        };
        if let Some(dir) = &cwd {
            actions.push([PSPAWN_CHDIR, 0, 0, dir.as_ptr() as u64]);
        }

        // argv: the args as NUL-terminated strings (`argv[0]` is the program, set by `Command::new`),
        // then NULL. The environment is the one this process carries (see the module header).
        let args = self
            .args
            .iter()
            .map(|a| cstr(a.as_encoded_bytes()))
            .collect::<io::Result<Vec<_>>>()?;
        let mut argv: Vec<*const u8> = args.iter().map(|a| a.as_ptr()).collect();
        argv.push(crate::ptr::null());

        let mut pid = ENOENT;
        for path in candidates(self.program.as_encoded_bytes()) {
            let path = cstr(&path)?;
            let req = [
                path.as_ptr() as u64,
                argv.as_ptr() as u64,
                0,
                actions.as_ptr() as u64,
                actions.len() as u64,
            ];
            pid = host::pspawn(req.as_ptr());
            if pid != ENOENT {
                break;
            }
        }
        // The child holds its own ends now: the parent keeps only the ones its streams hand back.
        let [stdin, stdout, stderr] = streams.map(ChildStream::ours);
        if pid < 0 {
            return Err(err(pid));
        }
        Ok((Process { pid: pid as i32, status: None }, StdioPipes { stdin, stdout, stderr }))
    }
}

/// How one child stream is set up: a fresh pipe when it needs one, `(the parent's end, the child's
/// end)`, and whether the parent keeps its end (the stream `StdioPipes` hands back).
struct ChildStream {
    pipe: Option<(Pipe, Pipe)>,
    keep: bool,
}

impl ChildStream {
    /// Push the file action that makes the child's descriptor `fd` what `cfg` asks for, minting the
    /// pipe it needs. An inherited stream needs no action.
    fn setup(cfg: &Stdio, fd: u64, actions: &mut Vec<[u64; 4]>) -> io::Result<ChildStream> {
        let mut dup2 = |from: i32| actions.push([PSPAWN_DUP2, from as u64, fd, 0]);
        let mut pipe = None;
        let mut keep = false;
        match cfg {
            Stdio::Inherit => {}
            Stdio::ParentStdout | Stdio::ParentStderr => {
                let from = if matches!(cfg, Stdio::ParentStdout) { 1 } else { 2 };
                if from != fd {
                    dup2(from as i32);
                }
            }
            Stdio::Fd(p) => dup2(p.fd()),
            Stdio::InheritFile(f) => dup2(f.fd()),
            // A null stdin reads EOF at once: a pipe whose write end nobody keeps.
            Stdio::MakePipe | Stdio::Null if fd == 0 => {
                let (read, write) = pipe::pipe()?;
                dup2(read.fd());
                keep = matches!(cfg, Stdio::MakePipe);
                pipe = Some((write, read));
            }
            Stdio::MakePipe => {
                let (read, write) = pipe::pipe()?;
                dup2(write.fd());
                keep = true;
                pipe = Some((read, write));
            }
            Stdio::Null => actions.push([PSPAWN_CLOSE, fd, 0, 0]),
        }
        Ok(ChildStream { pipe, keep })
    }

    /// The parent's end, once the child holds its own: kept if the stream hands it back, else closed.
    fn ours(self) -> Option<Pipe> {
        let (ours, _theirs) = self.pipe?;
        self.keep.then_some(ours)
    }
}

/// `Command::output`: run the command with its stdout and stderr piped, read both to EOF (stdout
/// first — see the module header), and reap it.
pub fn output(cmd: &mut Command) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
    let (mut process, mut pipes) = cmd.spawn(Stdio::MakePipe, false)?;
    drop(pipes.stdin.take());
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    match (pipes.stdout.take(), pipes.stderr.take()) {
        (Some(out), Some(err)) => read_output(out, &mut stdout, err, &mut stderr)?,
        (Some(out), None) => {
            out.read_to_end(&mut stdout)?;
        }
        (None, Some(err)) => {
            err.read_to_end(&mut stderr)?;
        }
        (None, None) => {}
    }
    let status = process.wait()?;
    Ok((status, stdout, stderr))
}

impl From<ChildPipe> for Stdio {
    fn from(pipe: ChildPipe) -> Stdio {
        Stdio::Fd(pipe)
    }
}

impl From<io::Stdout> for Stdio {
    fn from(_: io::Stdout) -> Stdio {
        Stdio::ParentStdout
    }
}

impl From<io::Stderr> for Stdio {
    fn from(_: io::Stderr) -> Stdio {
        Stdio::ParentStderr
    }
}

impl From<File> for Stdio {
    fn from(file: File) -> Stdio {
        Stdio::InheritFile(file)
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            let mut debug_command = f.debug_struct("Command");
            debug_command.field("program", &self.program).field("args", &self.args);
            if !self.env.is_unchanged() {
                debug_command.field("env", &self.env);
            }
            if self.cwd.is_some() {
                debug_command.field("cwd", &self.cwd);
            }
            if self.stdin.is_some() {
                debug_command.field("stdin", &self.stdin);
            }
            if self.stdout.is_some() {
                debug_command.field("stdout", &self.stdout);
            }
            if self.stderr.is_some() {
                debug_command.field("stderr", &self.stderr);
            }
            debug_command.finish()
        } else {
            if let Some(ref cwd) = self.cwd {
                write!(f, "cd {cwd:?} && ")?;
            }
            if self.env.does_clear() {
                write!(f, "env -i ")?;
            } else {
                let mut any_removed = false;
                for (key, value_opt) in self.get_envs() {
                    if value_opt.is_none() {
                        if !any_removed {
                            write!(f, "env ")?;
                            any_removed = true;
                        }
                        write!(f, "-u {} ", key.to_string_lossy())?;
                    }
                }
            }
            for (key, value_opt) in self.get_envs() {
                if let Some(value) = value_opt {
                    write!(f, "{}={value:?} ", key.to_string_lossy())?;
                }
            }
            if self.program != self.args[0] {
                write!(f, "[{:?}] ", self.program)?;
            }
            write!(f, "{:?}", self.args[0])?;
            for arg in &self.args[1..] {
                write!(f, " {arg:?}")?;
            }
            Ok(())
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Process + exit status
////////////////////////////////////////////////////////////////////////////////

pub struct Process {
    pid: i32,
    /// Its wait status once reaped: `OP_WAITPID` consumes the child, so `wait`/`try_wait` keep it.
    status: Option<i32>,
}

impl Process {
    pub fn id(&self) -> u32 {
        self.pid as u32
    }

    pub fn kill(&mut self) -> io::Result<()> {
        // A reaped child is gone, and nothing is signalled — unix answers the same.
        if self.status.is_some() {
            return Ok(());
        }
        let r = host::kill(self.pid as i64, SIGKILL);
        if r < 0 { Err(err(r)) } else { Ok(()) }
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        match self.reap(0)? {
            Some(status) => Ok(ExitStatus(status)),
            None => Err(io::const_error!(io::ErrorKind::Other, "waitpid: the child was not reaped")),
        }
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        Ok(self.reap(WNOHANG)?.map(ExitStatus))
    }

    /// Reap the child with `waitpid(pid, options)`: its wait status, or `None` when `options` holds
    /// `WNOHANG` and it is still running.
    fn reap(&mut self, options: i64) -> io::Result<Option<i32>> {
        if let Some(s) = self.status {
            return Ok(Some(s));
        }
        let mut sb = [0u8; 4];
        let r = host::waitpid(self.pid as i64, sb.as_mut_ptr(), options);
        if r < 0 {
            return Err(err(r));
        }
        if r == 0 {
            return Ok(None);
        }
        let s = i32::from_le_bytes(sb);
        self.status = Some(s);
        Ok(Some(s))
    }
}

/// A wait-encoded exit status: `WEXITSTATUS` in bits 8–15 (the host records normal exits only).
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub struct ExitStatus(i32);

impl ExitStatus {
    pub fn exit_ok(&self) -> Result<(), ExitStatusError> {
        // A zero wait-status is a clean `exit(0)`; anything else carries a non-zero code.
        if self.0 == 0 { Ok(()) } else { Err(ExitStatusError(self.0)) }
    }

    pub fn code(&self) -> Option<i32> {
        Some((self.0 >> 8) & 0xff)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exit status: {}", (self.0 >> 8) & 0xff)
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct ExitStatusError(i32);

impl fmt::Debug for ExitStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExitStatusError").field(&((self.0 >> 8) & 0xff)).finish()
    }
}

impl Into<ExitStatus> for ExitStatusError {
    fn into(self) -> ExitStatus {
        ExitStatus(self.0)
    }
}

impl ExitStatusError {
    pub fn code(self) -> Option<NonZero<i32>> {
        NonZero::new((self.0 >> 8) & 0xff)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitCode(u8);

impl ExitCode {
    pub const SUCCESS: ExitCode = ExitCode(0);
    pub const FAILURE: ExitCode = ExitCode(1);

    pub fn as_i32(&self) -> i32 {
        self.0 as i32
    }
}

impl From<u8> for ExitCode {
    fn from(code: u8) -> Self {
        Self(code)
    }
}

////////////////////////////////////////////////////////////////////////////////
// CommandArgs + free fns
////////////////////////////////////////////////////////////////////////////////

pub struct CommandArgs<'a> {
    iter: crate::slice::Iter<'a, OsString>,
}

impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;
    fn next(&mut self) -> Option<&'a OsStr> {
        self.iter.next().map(|os| &**os)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> ExactSizeIterator for CommandArgs<'a> {
    fn len(&self) -> usize {
        self.iter.len()
    }
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

impl<'a> fmt::Debug for CommandArgs<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter.clone()).finish()
    }
}

/// Read a child's stdout to EOF, then its stderr (see the module header).
pub fn read_output(
    out: ChildPipe,
    stdout: &mut Vec<u8>,
    err: ChildPipe,
    stderr: &mut Vec<u8>,
) -> io::Result<()> {
    out.read_to_end(stdout)?;
    err.read_to_end(stderr)?;
    Ok(())
}

pub fn getpid() -> u32 {
    // The personality's pid for this process; `1`, a run's root, without a posix grant.
    if host::have_posix() { host::getpid() as u32 } else { 1 }
}
