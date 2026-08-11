//! svm process spawning: `std::process::Command` over the POSIX personality's **fork-free** spawn
//! (svm-posix `OP_SPAWN`/`OP_WAITPID`). A spawn runs the named command *to completion* synchronously —
//! there is no fork-returns-twice — so a `Process` is already-exited by the time `spawn` returns, and
//! `wait`/`try_wait` just reap the recorded status (`OP_WAITPID`). The command is resolved by the
//! embedder's spawn delegate (`Posix::set_spawn`); without one, spawning is `Unsupported`.
//!
//! Stdio: the child inherits the caller's fd 0 (stdin) and fd 1 (stdout). To **capture** stdout
//! (`Command::output`), we bracket the spawn with `dup(1)`/`dup2(pipe_w, 1)`/restore so the child's
//! output lands in an in-personality pipe we drain afterwards (the FIFO is unbounded, so the whole
//! output is buffered by the time the synchronous spawn returns). Because the model is synchronous,
//! there is no live child to stream *into*: child stdin is whatever fd 0 already holds, so `StdioPipes`
//! never yields a writable stdin, and stderr is not separately captured (the host routes only stdout).
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

/// Map a negative errno from a spawn/wait op to an `io::Error`: `-ENOSYS` (no spawn delegate wired) and
/// `-ENOENT` (unknown command) get the kinds programs match; the rest fall back to the raw code.
fn err(code: i64) -> io::Error {
    match code {
        -2 => io::const_error!(io::ErrorKind::NotFound, "spawn: no such command"),
        -38 => io::const_error!(io::ErrorKind::Unsupported, "spawn: no posix spawn delegate is wired"),
        _ => io::Error::from_raw_os_error((-code) as i32),
    }
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
    #[allow(dead_code)] // only reachable via `From<File>`
    InheritFile(File),
    #[allow(dead_code)] // only reachable via `From<ChildPipe>`
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

    /// Run the command to completion on the personality's spawn delegate, returning the (already-exited)
    /// `Process` and whatever stdio pipes the disposition asked for. `_default` fills in an unset stdout
    /// disposition (`MakePipe` for `output`, `Inherit` for `status`/`spawn`); `_needs_stdin` is unused —
    /// the synchronous model has no live child to stream stdin into (see the module header).
    pub fn spawn(
        &mut self,
        default: Stdio,
        _needs_stdin: bool,
    ) -> io::Result<(Process, StdioPipes)> {
        if !host::have_posix() {
            return Err(unsupported_err());
        }

        // Resolve the stdout disposition and, if it redirects fd 1, bracket the spawn so the parent's
        // fd 1 is restored afterwards. `captured` is the read end handed back for `output` to drain.
        let stdout_cfg = self.stdout.as_ref().unwrap_or(&default);
        let mut captured: Option<Pipe> = None;
        let mut saved_fd1: Option<i32> = None;
        let mut discard: Option<Pipe> = None;
        match stdout_cfg {
            Stdio::MakePipe => {
                let (read_end, write_end) = pipe::pipe()?;
                saved_fd1 = redirect_fd1(write_end.fd());
                captured = Some(read_end);
                // `write_end` drops here; `dup2` gave fd 1 its own reference to the shared FIFO.
            }
            Stdio::Null => {
                let (read_end, write_end) = pipe::pipe()?;
                saved_fd1 = redirect_fd1(write_end.fd());
                discard = Some(read_end); // drained-then-dropped below: the child's output is discarded
            }
            Stdio::Fd(p) => {
                saved_fd1 = redirect_fd1(p.fd());
            }
            Stdio::InheritFile(_) => {
                return Err(io::const_error!(
                    io::ErrorKind::Unsupported,
                    "file-backed child stdio is not supported on svm"
                ));
            }
            // Inherit / ParentStdout / ParentStderr: the child writes to the parent's fd 1.
            Stdio::Inherit | Stdio::ParentStdout | Stdio::ParentStderr => {}
        }

        // argv: the args as a NUL-separated blob (`argv[0]` is the program, set by `Command::new`).
        let mut argv: Vec<u8> = Vec::new();
        for (i, a) in self.args.iter().enumerate() {
            if i > 0 {
                argv.push(0);
            }
            argv.extend_from_slice(a.as_encoded_bytes());
        }
        let name = self.program.as_encoded_bytes();
        let pid = host::spawn(name.as_ptr(), name.len() as i64, argv.as_ptr(), argv.len() as i64);

        // Restore the parent's fd 1 before returning, whatever happened.
        if let Some(s) = saved_fd1 {
            host::dup2(s as i64, 1);
            let _ = host::close(s as i64);
        }
        // For `Null`, drop the discarded output so the FIFO/read-end fd is released.
        drop(discard);

        if pid < 0 {
            return Err(err(pid));
        }
        let pipes = StdioPipes { stdin: None, stdout: captured, stderr: None };
        Ok((Process { pid: pid as i32, status: None }, pipes))
    }
}

/// Save the parent's fd 1 (via `dup`) and point fd 1 at `fd`, returning the saved fd to restore later
/// (`None` if the save failed — then fd 1 is not restored, which is acceptable for a discard sink).
fn redirect_fd1(fd: i32) -> Option<i32> {
    let saved = host::dup(1);
    host::dup2(fd as i64, 1);
    (saved >= 0).then_some(saved as i32)
}

/// `Command::output`: capture stdout (stderr is not separately captured by the host spawn model, so it
/// comes back empty). Mirrors the generic `output` in `sys::process` but for the synchronous spawn.
pub fn output(cmd: &mut Command) -> io::Result<(ExitStatus, Vec<u8>, Vec<u8>)> {
    let (mut process, mut pipes) = cmd.spawn(Stdio::MakePipe, false)?;
    drop(pipes.stdin.take());
    let mut stdout = Vec::new();
    if let Some(out) = pipes.stdout.take() {
        out.read_to_end(&mut stdout)?;
    }
    let status = process.wait()?;
    Ok((status, stdout, Vec::new()))
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
    /// Cached once reaped, so `wait`/`try_wait` are idempotent (`OP_WAITPID` consumes the child).
    status: Option<i32>,
}

impl Process {
    pub fn id(&self) -> u32 {
        self.pid as u32
    }

    pub fn kill(&mut self) -> io::Result<()> {
        // The child already ran to completion (synchronous spawn); there is nothing to signal.
        Ok(())
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        if let Some(s) = self.status {
            return Ok(ExitStatus(s));
        }
        let mut sb = [0u8; 4];
        let r = host::waitpid(self.pid as i64, sb.as_mut_ptr(), 0);
        if r < 0 {
            return Err(err(r));
        }
        let s = i32::from_le_bytes(sb);
        self.status = Some(s);
        Ok(ExitStatus(s))
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        // A spawned child has already run, so its status is always immediately available.
        Ok(Some(self.wait()?))
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

pub fn read_output(
    out: ChildPipe,
    stdout: &mut Vec<u8>,
    err: ChildPipe,
    stderr: &mut Vec<u8>,
) -> io::Result<()> {
    // The FIFOs are already fully populated (synchronous spawn) and non-blocking, so a straight
    // drain of each in turn cannot deadlock.
    out.read_to_end(stdout)?;
    err.read_to_end(stderr)?;
    Ok(())
}

pub fn getpid() -> u32 {
    // Single-process personality: a fixed, stable pid (there is no `getpid` op).
    1
}
