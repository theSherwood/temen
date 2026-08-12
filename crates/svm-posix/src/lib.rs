//! A **POSIX personality** as an embedder host capability (POSIX.md, §7).
//!
//! The "host provides libc as a capability; the §7 named-import mechanism carries it" story — the
//! same shape as [`svm-wasi`](../svm_wasi), generalized from a WASI subset to the POSIX/libc surface a
//! fork-less shell (BusyBox `ash` → Bash) links against. [`resolve`] binds libc symbol names to a
//! single [`svm_interp::cap_id::HOST_PROC`] capability; [`handler`] implements the ops over the guest
//! window. All libc *semantics* live **here** — outside the interp escape-TCB — reached only through a
//! granted, masked, type-checked handle (DESIGN.md §7).
//!
//! **State model (POSIX.md §3):** the bytes a libc call touches — a `malloc`'d buffer, a `write`
//! source — live in the **guest window** (native-speed access; `malloc` returns a window offset). The
//! *bookkeeping* — the allocator's cursor, captured stdout/stderr, the stdin cursor — lives host-side
//! in [`Inner`], never in the guest's address space, so the guest cannot corrupt it.
//!
//! Scope: `write` / `read` / `malloc` / `free` / `exit`, plus `open` / `close` / `lseek` / `unlink`
//! over an in-memory filesystem (a `path → bytes` memfs) with a host-side fd table, and
//! `getcwd` / `chdir` / `getenv` / `setenv` over a host-side cwd + environment. `malloc` is a
//! first-fit free list over a configured window-heap region. A minimal **`exec` surface** (STAGE1.md
//! §5) lets a shell on this personality launch an external command: `exec_lookup` resolves a name
//! against a `name → Module` PATH registry and `exec_stdout` hands back the `Stream` to forward — the
//! spawn itself is the shell's own `Instantiator` `cap.call` (op 13), not a personality op. Still to
//! come (POSIX.md §6): signals and `fork`/`clone`. Pure computation (`strlen`, `snprintf`, `math`, …)
//! is **guest code**, not a cap — it needs no authority (POSIX.md §1).
#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use svm_interp::{cap_id, GuestMem, Host, HostProc, SignalSource, Trap};
use svm_ir::ResolvedCap;

/// Op numbers on the shared `HOST_PROC` handle; [`resolve`] maps libc names to these.
pub const OP_WRITE: u32 = 0;
pub const OP_READ: u32 = 1;
pub const OP_MALLOC: u32 = 2;
pub const OP_FREE: u32 = 3;
pub const OP_EXIT: u32 = 4;
pub const OP_OPEN: u32 = 5;
pub const OP_CLOSE: u32 = 6;
pub const OP_LSEEK: u32 = 7;
pub const OP_UNLINK: u32 = 8;
pub const OP_GETCWD: u32 = 9;
pub const OP_CHDIR: u32 = 10;
pub const OP_GETENV: u32 = 11;
pub const OP_SETENV: u32 = 12;
pub const OP_STAT: u32 = 13;
pub const OP_OPENDIR: u32 = 14;
pub const OP_READDIR: u32 = 15;
pub const OP_CLOSEDIR: u32 = 16;
pub const OP_ARGC: u32 = 17;
pub const OP_ARGV: u32 = 18;
/// **Personality `exec` surface** (STAGE1.md §5) — how a shell running on this personality launches an
/// external command. `exec_lookup(name_ptr, name_len) -> module_handle | -1` resolves a command name
/// against the [`Inner::commands`] PATH registry (a `name → Module` handle map the embedder seeds); the
/// shell then drives `Instantiator.instantiate_module_named` (op 13) + `join` on the returned handle.
/// `exec_stdout() -> stream_handle` returns the `Stream` the shell should re-grant to the child under
/// the name `"stdout"` so the command's `write(1, …)` reaches the shell's sink. `exec_stdin(ptr, len)
/// -> stream_handle` is the input counterpart (a **filter** command): the personality pushes the
/// `[ptr, len)` bytes — the shell's current input, e.g. a `< file` redirect it drained — into a
/// read-only pipe FIFO and returns the read end for the shell to re-grant as `"stdin"`, so the
/// command's `read(0, …)` drains them (then `0` = EOF). Neither op *is* the spawn — op 13 is a guest
/// `cap.call` (the compiled shell holds the `Instantiator`); the personality only supplies the registry
/// lookup and the forwardable stdout/stdin handles.
pub const OP_EXEC_LOOKUP: u32 = 19;
pub const OP_EXEC_STDOUT: u32 = 20;
pub const OP_EXEC_STDIN: u32 = 21;
/// `exec_win(module_handle) -> size_log2`: the granted command `Module`'s declared window, so the shell
/// carves each spawn to match it (a §14 child's carve must equal its declared memory, and commands vary
/// in size). Returns `-1` for an unregistered handle. The embedder records it in [`Posix::register_command`].
pub const OP_EXEC_WIN: u32 = 22;

/// **POSIX process/fd surface** (STAGE1.md slice 1 — the ABI a real shell links against, replacing the
/// hand-shell's bespoke `open`/`close` fd juggling). `pipe(fds_ptr) -> 0` creates an in-personality byte
/// FIFO and stores `[read_fd, write_fd]` (two `i32`s) at `fds_ptr`. `dup2(oldfd, newfd) -> newfd`
/// re-points `newfd` at `oldfd`'s object (closing whatever `newfd` was), the primitive a shell uses to
/// wire a redirect (`dup2(pipe_w, 1)`) before launching a command. `dup(oldfd) -> fd` clones `oldfd`
/// onto the lowest free fd. `fcntl(fd, cmd, arg)` covers `F_DUPFD`/`F_DUPFD_CLOEXEC` (dup ≥ `arg`) and
/// accepts `F_GETFD`/`F_SETFD`/`F_GETFL`/`F_SETFL` as no-ops (there is no exec-in-place, so `FD_CLOEXEC`
/// has nothing to act on yet). These are **intra-personality** pipes: a single guest's write end and read
/// end share one buffer, non-blocking (an empty pipe reads `0`/EOF). Handing a pipe end to a *spawned
/// child* as its stdin/stdout is `spawn`'s job (below); this group lands the fd surface.
pub const OP_PIPE: u32 = 23;
pub const OP_DUP2: u32 = 24;
pub const OP_DUP: u32 = 25;
pub const OP_FCNTL: u32 = 26;

/// **POSIX spawn/wait surface** (STAGE1.md slice 2). The fork-free process primitive: `spawn` launches a
/// registered command as a child, runs it to completion (sequential — there is no fork-returns-twice),
/// and `waitpid`/`wait` reap its exit status. Because a spawn is *authority* the libc personality does
/// not itself hold (children are born destitute; the shell mints and grants), the actual instantiate+run
/// is an **embedder-wired delegate** ([`Posix::set_spawn`]) — opt-in, exactly like the stdout `Stream`.
/// Absent a delegate, `spawn` is `-ENOSYS` (a program links and its fork-free paths run; spawning fails
/// closed). The child **inherits the caller's fd 0 and fd 1**: `spawn` drains the current fd-0 binding
/// (preloaded stdin, a file, or a pipe) as the child's input and routes the child's captured stdout to
/// the current fd-1 binding — so a `dup2(pipe_w, 1)` / `dup2(file, 1)` redirect before the spawn lands
/// the child's output exactly where POSIX would. `fork`/`vfork`/`execve` (return-twice / image-replace)
/// remain parked on the durable-clone capstone.
///
/// `spawn(name_ptr, name_len, argv_ptr, argv_len) -> pid | -errno`: look up the command by name; `argv`
/// is the `argv_len` bytes at `argv_ptr` as a NUL-separated blob (empty ⇒ `[name]`). Returns a synthetic
/// pid. `waitpid(pid, status_ptr, options) -> pid | -errno`: reap `pid` (or any child if `pid == -1`),
/// writing the wait-encoded status (`WEXITSTATUS` in bits 8–15) to `status_ptr` when non-null;
/// `-ECHILD` for an unknown pid. `wait(status_ptr)` is `waitpid(-1, status_ptr, 0)`.
pub const OP_SPAWN: u32 = 27;
pub const OP_WAITPID: u32 = 28;
pub const OP_WAIT: u32 = 29;

/// **POSIX signal surface — L0 doorbell** (STAGE1.md slice 3 / PROCESS.md §9). A signal a shell traps
/// (SIGINT/SIGTERM/…) becomes a **pending bit** the guest polls at a safe point (a command boundary) and
/// dispatches itself — no asynchronous interruption of running guest code (that is L1/L2, parked). This
/// is exact for `trap`: a *caught* signal (a handler installed) is delivered; an *ignored* one is dropped;
/// a *default*-disposition one is dropped in L0 (default actions — terminate a running loop — are L1/L2).
///
/// `signal(signum, handler) -> prev_handler | -errno`: record the disposition for `signum` (`SIG_DFL = 0`,
/// `SIG_IGN = 1`, else a guest handler pointer), returning the previous. `kill(pid, sig) -> 0 | -errno`
/// raises `sig` (sets its pending bit; `pid` is advisory in this single-process model — `raise(s)` is the
/// guest one-liner `kill(0, s)`). `sigcheck(_) -> handler | 0`: the doorbell poll — clear and return the
/// handler pointer of the lowest-numbered pending **caught** signal (skipping/dropping ignored and default
/// ones), or `0` when none is deliverable, so the guest runtime runs `((void(*)(void))handler)()` at its
/// safe point. The embedder raises an external signal (a terminal `^C`) via [`Posix::raise_signal`].
pub const OP_SIGNAL: u32 = 30;
pub const OP_KILL: u32 = 31;
pub const OP_SIGCHECK: u32 = 32;
/// `clock(clock_id) -> nanos` (POSIX.md — the `Clock` surface; `std::time` reaches it via the svm
/// `std` PAL). `clock_id == 1` is monotonic (nanos since this personality started), anything else is
/// realtime (nanos since the Unix epoch). An embedder/test can pin the value with [`Posix::set_clock`]
/// for determinism (the differential harness wants a reproducible clock).
pub const OP_CLOCK: u32 = 33;
/// `getenv_r(name, nlen, buf, cap) -> nbytes | -1` — a **buffer-writing** `getenv` for callers that
/// own the destination (Rust's `std::env::var` copies into an `OsString`), unlike op 11 which
/// materializes a stable `char*` in the personality arena. Returns the value's byte length (writing it
/// into `[buf, cap)` when it fits — the two-call size-then-fetch shape of `getcwd`/`readdir`), or `-1`
/// if unset. Because it writes into guest-owned memory it needs **no arena**, so it never contends
/// with the guest's own heap — the reason the svm `std` PAL uses it.
pub const OP_GETENV_R: u32 = 34;
/// `unsetenv(name, nlen) -> 0` — remove an environment variable (`std::env::remove_var`). Absent name
/// is a success no-op; a non-UTF-8 name is `-EINVAL`.
pub const OP_UNSETENV: u32 = 35;
/// `environ(index, buf, cap) -> len | -1` — enumerate the environment for `std::env::vars`. Writes the
/// `index`-th `KEY=VALUE` (keys **sorted**, so the order is deterministic) into `[buf, cap)` and
/// returns its byte length (size-then-fetch like `getenv_r`); `-1` once `index` is past the last var.
pub const OP_ENVIRON: u32 = 36;
/// `mkdir(path, plen, mode) -> 0 | -errno` — create an explicit **empty** directory (`std::fs::create_dir`).
/// The memfs otherwise infers dirs from file prefixes, so an empty dir needs recording. `mode` is
/// ignored (no perm model). `-EEXIST` if the path already exists (file or dir), `-ENOENT` if the parent
/// isn't a directory, `-EINVAL` for a non-UTF-8 path.
pub const OP_MKDIR: u32 = 37;
/// `rename(old, olen, new, nlen) -> 0 | -errno` — rename a file or directory (`std::fs::rename`). A file
/// key moves (overwriting any existing target file); a directory re-keys every file/subdir under it.
/// `-ENOENT` if `old` doesn't exist, `-EINVAL` for a non-UTF-8 path.
pub const OP_RENAME: u32 = 38;
/// `rmdir(path, plen) -> 0 | -errno` — remove an **empty** directory (`std::fs::remove_dir`). `-ENOTDIR`
/// if the path is a file, `-ENOENT` if it isn't a directory, `-ENOTEMPTY` if it still has children,
/// `-EINVAL` for the root or a non-UTF-8 path.
pub const OP_RMDIR: u32 = 39;
/// `sigprocmask(how, set, oldset) -> 0 | -errno` (#796 — PROCESS.md §9) — examine/change the blocked
/// signal set. `how` is `SIG_BLOCK`(0) / `SIG_UNBLOCK`(1) / `SIG_SETMASK`(2); `set`/`oldset` (either may be
/// null) point at an 8-byte `sigset_t` in this personality's ABI — a `u64` bitset, signal `s` = bit `s`.
/// A pending **blocked** signal is held (not delivered by `sigcheck`) until unblocked. `SIGKILL`/`SIGSTOP`
/// can never be blocked. Bad `how` (with a non-null `set`) is `-EINVAL`.
pub const OP_SIGPROCMASK: u32 = 40;
/// `sigaction(signum, act, oldact) -> 0 | -errno` (#796) — the richer `signal`. `act`/`oldact` (either may
/// be null) point at a 24-byte `struct sigaction` in this personality's ABI: `{ sa_handler: i64@0,
/// sa_mask: u64@8, sa_flags: i64@16 }`. Records the disposition (like op 30 `signal`) plus `sa_mask` and
/// `sa_flags` for round-trip fidelity. Out-of-range `signum` is `-EINVAL`.
pub const OP_SIGACTION: u32 = 41;
/// `sigaltstack(sp, size) -> 0` (#796 L2 async signals) — register the guest's dedicated **signal-handler
/// stack**: the data-stack pointer an async handler runs on. The interp can't reuse the interrupted
/// frame's stack (a frame's size is baked into the IR, not known to the interp), so async delivery
/// requires this. `sp == 0` disables it (poll-only — the safe default). `size` is advisory (window masking
/// bounds every access anyway). This simplified ABI takes two `long`s, not POSIX's `stack_t*` structs.
pub const OP_SIGALTSTACK: u32 = 42;

/// **`net` capability ops** (POSIX.md §5a) — a **separate named handle** (`"net"`), not entries in the
/// libc table above: authority is its own granted capability (the WASI 0.2 lesson), while the data
/// plane rides the ordinary fd `read`/`write`/`close`/`dup2` ops. `connect`/`bind` take a socket-address
/// **blob** — `[family u8 (4|6), port u16 LE, addr 4|16 bytes]` — and write the resulting local/bound
/// address back through an out-blob. Loopback is served by the in-personality **memnet**; anything
/// beyond routes to the embedder's [`NetDelegate`] ([`Posix::set_net`]) or fails closed.
pub const NET_CONNECT: u32 = 1;
pub const NET_BIND: u32 = 2;
pub const NET_ACCEPT: u32 = 3;
pub const NET_SHUTDOWN: u32 = 4;
pub const NET_RESOLVE: u32 = 5;

/// `signal` dispositions (the low, non-pointer handler values): default action, or ignore.
const SIG_DFL: i64 = 0;
const SIG_IGN: i64 = 1;
/// `sigprocmask` `how` values (Linux ABI, so a guest's `<signal.h>` agrees).
const SIG_BLOCK: i64 = 0;
const SIG_UNBLOCK: i64 = 1;
const SIG_SETMASK: i64 = 2;
/// The signals POSIX forbids blocking — `SIGKILL` (9) and `SIGSTOP` (19). Cleared out of any requested
/// mask so a guest can never wedge them.
const UNMASKABLE: u64 = (1 << 9) | (1 << 19);

/// Negative errnos this personality returns (Linux values, so a guest's `<errno.h>` agrees).
const ENOENT: i64 = -2; // no such file (open without O_CREAT; stat/opendir of an absent path)
const ECHILD: i64 = -10; // waitpid on a pid that is not a live child
const EBADF: i64 = -9; // an op on an fd this personality does not serve
const EINVAL: i64 = -22; // bad argument (whence, non-UTF-8 path, negative seek)
const ENOTDIR: i64 = -20; // opendir on a path that is a regular file, not a directory
const ESPIPE: i64 = -29; // lseek on a pipe/stdio fd (not seekable)
const ERANGE: i64 = -34; // result won't fit the caller's buffer (getcwd)
const ENOSYS: i64 = -38; // spawn with no embedder-wired delegate (fail closed)
const EEXIST: i64 = -17; // mkdir/rename onto a path that already exists
const ENOTEMPTY: i64 = -39; // rmdir on a directory that still has children
const EAGAIN: i64 = -11; // read/accept on an empty memnet socket (would block a cooperative guest)
const EACCES: i64 = -13; // bind beyond loopback (no delegate-granted listener path yet, POSIX.md §5a)
const EPIPE: i64 = -32; // write on a socket whose write side is shut down
const ENOTSOCK: i64 = -88; // a net op on an fd that is not a socket/listener
const EADDRINUSE: i64 = -98; // bind on a loopback port another listener holds
const ECONNREFUSED: i64 = -111; // connect with no listener (loopback) or no delegate (beyond)

/// `fcntl` commands this personality serves (Linux `<fcntl.h>` values). `F_DUPFD`/`F_DUPFD_CLOEXEC`
/// duplicate to the lowest free fd `>= arg`; `F_GETFD`/`F_SETFD`/`F_GETFL`/`F_SETFL` are accepted no-ops
/// (there is no exec-in-place here, so `FD_CLOEXEC` and status flags have nothing to gate yet).
const F_DUPFD: i64 = 0;
const F_GETFD: i64 = 1;
const F_SETFD: i64 = 2;
const F_GETFL: i64 = 3;
const F_SETFL: i64 = 4;
const F_DUPFD_CLOEXEC: i64 = 1030;

/// `struct stat` **mode** bits this personality reports (Linux `<sys/stat.h>` `S_IFMT` values). The
/// personality's `struct stat` is a deliberately minimal **`{ i64 st_mode; i64 st_size; }`** (16
/// bytes) — the two fields a shell actually reads (`S_ISDIR`/`S_ISREG` on `st_mode`, `st_size`); a
/// guest `<sys/stat.h>` agrees on that layout (POSIX.md §5). A memfs file is a regular file; a path
/// that is a prefix of some file key (or `"/"`) is a directory.
const S_IFREG: i64 = 0o100000; // regular file (| 0o644 perms)
const S_IFDIR: i64 = 0o040000; // directory (| 0o755 perms)

// The ABI is **explicit-length**, syscall-style: a string argument is `(ptr, len)`, not a
// NUL-terminated `char*`. This avoids an unbounded window scan (safer) and matches `read`/`write`;
// a thin guest libc adapts C's NUL-terminated conventions to it (POSIX.md §4, "one ABI, two
// bindings" — the shim is guest code). `getcwd`/`getenv` *write* NUL-terminated results (C's
// contract) since the caller consumes them as `char*`.

/// `open` flags (Linux `<fcntl.h>` values). The low two bits are the access mode.
const O_ACCMODE: i64 = 3;
const O_WRONLY: i64 = 1;
const O_RDWR: i64 = 2;
const O_CREAT: i64 = 0o100;
const O_TRUNC: i64 = 0o1000;
const O_APPEND: i64 = 0o2000;

/// `lseek` whence values (`SEEK_SET`/`SEEK_CUR`/`SEEK_END`).
const SEEK_SET: i64 = 0;
const SEEK_CUR: i64 = 1;
const SEEK_END: i64 = 2;

/// One open **memfs file** entry: which file it refers to, the current offset, and whether it was opened
/// for writing. Independent offsets per fd, shared file contents (POSIX file semantics).
struct OpenFile {
    path: String,
    pos: usize,
    writable: bool,
}

/// A shared, in-personality **pipe buffer** — a byte FIFO both ends of a `pipe()` hold via `Arc`.
/// Non-blocking: a `read` on an empty buffer returns `0` (EOF), since a single cooperative guest cannot
/// block on itself. Cross-process pipe semantics (a spawned child draining a parent's write end) arrive
/// with the `execve`/spawn slice; this type gives the fd surface its buffering.
type PipeBuf = Arc<Mutex<VecDeque<u8>>>;

/// The result of one embedder-wired [`spawn`](Posix::set_spawn): the child's captured `stdout` and
/// `stderr` (which the personality routes to the caller's current fd-1 / fd-2 bindings) and its `status`
/// (an exit code, `0`–`255`, which `waitpid` returns wait-encoded). A crash/abnormal exit is out of
/// scope for the sequential fork-free primitive — model it as a nonzero code (`128 + signal`, the shell
/// convention). `Default` lets a delegate that produces no `stderr` build one with `..Default::default()`.
#[derive(Default)]
pub struct SpawnResult {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub status: i32,
}

/// The embedder's **spawn delegate**: `(command_name, argv, stdin_bytes) -> SpawnResult`. This is the
/// authority the libc personality does not itself hold — the embedder wires it ([`Posix::set_spawn`])
/// with whatever *running a child* means in its world (an `Instantiator` op-13 instantiate + `join`, a
/// scripted table, a real subprocess). Runs to completion synchronously (the sequential, no-fork model).
type SpawnFn = Box<dyn FnMut(&str, &[String], &[u8]) -> SpawnResult + Send>;

// ---- net (POSIX.md §5a): the memnet + the embedder delegate ------------------------------------

/// A socket address, parsed from / encoded to the wire blob `[family u8 (4|6), port u16 LE,
/// addr 4|16 bytes]` — sized so an address fits the 4-arg host-call ABI as one `(ptr, len)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NetAddr {
    pub v6: bool,
    pub port: u16,
    /// The address bytes; a v4 address uses the first 4.
    pub addr: [u8; 16],
}

impl NetAddr {
    /// The loopback address for `port` (v4 `127.0.0.1`).
    pub fn loopback(port: u16) -> NetAddr {
        let mut addr = [0u8; 16];
        addr[0] = 127;
        addr[3] = 1;
        NetAddr {
            v6: false,
            port,
            addr,
        }
    }

    fn parse(blob: &[u8]) -> Option<NetAddr> {
        let (&family, rest) = blob.split_first()?;
        let port = u16::from_le_bytes(rest.get(0..2)?.try_into().ok()?);
        let mut addr = [0u8; 16];
        match family {
            4 => addr[..4].copy_from_slice(rest.get(2..6)?),
            6 => addr.copy_from_slice(rest.get(2..18)?),
            _ => return None,
        }
        Some(NetAddr {
            v6: family == 6,
            port,
            addr,
        })
    }

    fn encode(&self) -> Vec<u8> {
        let n = if self.v6 { 16 } else { 4 };
        let mut out = Vec::with_capacity(3 + n);
        out.push(if self.v6 { 6 } else { 4 });
        out.extend_from_slice(&self.port.to_le_bytes());
        out.extend_from_slice(&self.addr[..n]);
        out
    }

    fn is_loopback(&self) -> bool {
        if self.v6 {
            self.addr[..15] == [0; 15] && self.addr[15] == 1
        } else {
            self.addr[0] == 127
        }
    }
}

/// The embedder's **network delegate** — the authority for anything beyond the loopback memnet
/// (the [`Posix::set_spawn`] analog; POSIX.md §5a). The personality itself holds no network
/// authority: absent a delegate, a non-loopback `connect` is `-ECONNREFUSED` and `resolve` of a
/// non-`localhost` name is `-ENOENT` — fail closed. Policy (allowlists, remapping, scripting)
/// lives here, host-side; the guest never sees a raw socket.
pub trait NetDelegate: Send {
    /// Connect to a non-loopback destination; return a live byte stream, or a negative errno
    /// (e.g. `-ECONNREFUSED`).
    fn connect(&mut self, addr: &NetAddr) -> Result<Box<dyn NetStream>, i64>;

    /// Resolve a host name (no port) to addresses. Default: refuse (`-ENOENT`).
    fn resolve(&mut self, _host: &str) -> Result<Vec<NetAddr>, i64> {
        Err(ENOENT)
    }
}

/// One delegate-backed connected stream (the embedder owns the real I/O). `recv` **may block
/// host-side** — the guest is suspended inside the host call, like a spawn running its child.
/// Return the byte count, `0` for EOF, or a negative errno.
pub trait NetStream: Send {
    fn send(&mut self, buf: &[u8]) -> i64;
    fn recv(&mut self, buf: &mut [u8]) -> i64;
    fn shutdown(&mut self, _how: i64) -> i64 {
        0
    }
}

/// A memnet write-side liveness token: dropped when the **last** fd-table entry holding this end
/// (the original plus every `dup`) closes, flipping the peer's empty reads from `-EAGAIN` ("no data
/// *yet*") to `0` (EOF). `shutdown(SHUT_WR)` sets the flag directly, ahead of the drop.
struct WriteToken {
    closed: Arc<AtomicBool>,
}

impl Drop for WriteToken {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
    }
}

/// One end of a memnet connection: crosswise-shared byte FIFOs (my `tx` is the peer's `rx`) plus the
/// liveness flags that give reads correct EOF-vs-would-block semantics. `Clone` shares everything —
/// a `dup` is another reference to the same connection end.
#[derive(Clone)]
struct MemSock {
    rx: PipeBuf,
    tx: PipeBuf,
    /// Set when the **peer's** write side is gone (its token dropped or it shut down writing):
    /// an empty `rx` then reads `0` (EOF) instead of `-EAGAIN`.
    peer_write_closed: Arc<AtomicBool>,
    /// Set by our own `shutdown(SHUT_RD)`: reads return `0` regardless of buffered data.
    read_shut: Arc<AtomicBool>,
    /// Our write-side token (shared by dups) — see [`WriteToken`].
    write_token: Arc<WriteToken>,
    local: NetAddr,
    peer: NetAddr,
}

/// A memnet listener: the bound address and the queue of not-yet-accepted server-side ends
/// (each pushed by a `connect` to this port). `Clone` shares the queue (`dup`).
#[derive(Clone)]
struct MemListener {
    addr: NetAddr,
    pending: Arc<Mutex<VecDeque<MemSock>>>,
}

/// Build a connected memnet pair for `client_addr → server_addr`: two FIFOs wired crosswise, with
/// each side's write-liveness flag observed by the other. Returns `(client_end, server_end)`.
fn mem_pair(client_addr: NetAddr, server_addr: NetAddr) -> (MemSock, MemSock) {
    let c2s: PipeBuf = Arc::new(Mutex::new(VecDeque::new()));
    let s2c: PipeBuf = Arc::new(Mutex::new(VecDeque::new()));
    let client_wclosed = Arc::new(AtomicBool::new(false));
    let server_wclosed = Arc::new(AtomicBool::new(false));
    let client = MemSock {
        rx: Arc::clone(&s2c),
        tx: Arc::clone(&c2s),
        peer_write_closed: Arc::clone(&server_wclosed),
        read_shut: Arc::new(AtomicBool::new(false)),
        write_token: Arc::new(WriteToken {
            closed: Arc::clone(&client_wclosed),
        }),
        local: client_addr,
        peer: server_addr,
    };
    let server = MemSock {
        rx: c2s,
        tx: s2c,
        peer_write_closed: client_wclosed,
        read_shut: Arc::new(AtomicBool::new(false)),
        write_token: Arc::new(WriteToken {
            closed: server_wclosed,
        }),
        local: server_addr,
        peer: client_addr,
    };
    (client, server)
}

/// One entry in the host-side fd table. The three stdio streams start as sentinels (`Stdin`/`Stdout`/
/// `Stderr`) so `dup2`/`dup`/`close` treat fds `0`/`1`/`2` uniformly with the rest; `open` adds `File`;
/// `pipe` adds a `PipeRead`/`PipeWrite` pair sharing one [`PipeBuf`]. `dup`/`dup2` clone an entry —
/// pipe ends clone the `Arc` (shared buffer); a `File` clones its description (independent offset — the
/// POSIX shared-offset nuance is a follow-up, irrelevant to the shell redirect pattern).
enum FdEntry {
    Stdin,
    Stdout,
    Stderr,
    File(OpenFile),
    PipeRead(PipeBuf),
    PipeWrite(PipeBuf),
    /// A connected memnet socket end (loopback; POSIX.md §5a).
    NetSock(MemSock),
    /// A delegate-backed connected stream (beyond loopback; the embedder owns the I/O).
    NetStream(Arc<Mutex<Box<dyn NetStream>>>),
    /// A memnet listener (`bind`+listen folded); `accept` pops its pending queue.
    NetListener(MemListener),
}

impl FdEntry {
    /// Clone this entry for `dup`/`dup2`: pipe ends share the buffer (`Arc` clone), a file copies its
    /// (independent) description, stdio sentinels are trivial.
    fn dup_clone(&self) -> FdEntry {
        match self {
            FdEntry::Stdin => FdEntry::Stdin,
            FdEntry::Stdout => FdEntry::Stdout,
            FdEntry::Stderr => FdEntry::Stderr,
            FdEntry::File(of) => FdEntry::File(OpenFile {
                path: of.path.clone(),
                pos: of.pos,
                writable: of.writable,
            }),
            FdEntry::PipeRead(p) => FdEntry::PipeRead(Arc::clone(p)),
            FdEntry::PipeWrite(p) => FdEntry::PipeWrite(Arc::clone(p)),
            // Socket ends and listeners share their connection state (`Arc` clones throughout) —
            // a dup is another reference to the same socket, and the write-liveness token's
            // last-drop EOF accounts for every dup.
            FdEntry::NetSock(s) => FdEntry::NetSock(s.clone()),
            FdEntry::NetStream(d) => FdEntry::NetStream(Arc::clone(d)),
            FdEntry::NetListener(l) => FdEntry::NetListener(l.clone()),
        }
    }
}

/// One open directory stream: the immediate child names under the opened path, snapshotted at
/// `opendir` (so a concurrent `open`/`unlink` during iteration doesn't perturb it — POSIX permits
/// either), plus the `readdir` cursor.
struct DirStream {
    entries: Vec<String>,
    pos: usize,
}

/// The allocator's alignment (bytes). 16 covers `max_align_t` (doubles / SIMD) so a `malloc`'d buffer
/// is suitably aligned for anything the guest stores into it.
const ALIGN: u64 = 16;

/// Host-side bookkeeping for one POSIX personality: captured output, the stdin cursor, and the
/// window-heap allocator cursor. Lives outside the guest window (POSIX.md §3), shared (`Arc<Mutex>`)
/// so an embedder/test can read the captured output back after a run.
struct Inner {
    stdout: Vec<u8>,
    /// When set, fd-1 writes go **here** instead of [`Inner::stdout`], and [`Posix::stdout`] reads it
    /// back. This unifies the shell's own output with a spawned child's: the embedder points it at the
    /// `Host`'s shared stdout sink (`Host::shared_stdout`), the same buffer a re-granted `Stream` writes
    /// to, so the shell's `write(1, …)` and the command's `write(1, …)` interleave in one stream
    /// (STAGE1.md §5). `None` keeps the self-contained captured-`Vec` behaviour (unchanged for every
    /// existing embedder).
    stdout_sink: Option<Arc<Mutex<Vec<u8>>>>,
    stderr: Vec<u8>,
    /// Preloaded standard input; `read(0, …)` drains it from `stdin_pos`.
    stdin: Vec<u8>,
    stdin_pos: usize,
    /// High-water mark: the window offset fresh (never-freed) allocations bump upward from.
    heap_next: u64,
    /// One past the last window byte the allocator may hand out.
    heap_end: u64,
    /// Live allocations, `ptr → size` — so `free` knows a block's length (the size header lives
    /// host-side, out of the guest's reach, rather than in a window prefix the guest could clobber).
    allocated: HashMap<u64, u64>,
    /// Freed blocks available for reuse (`offset, size`), first-fit. No coalescing yet — adjacent
    /// frees stay separate (a fragmentation follow-up, POSIX.md §6); reuse of a same-or-larger block
    /// works regardless.
    free_list: Vec<(u64, u64)>,
    /// The **in-memory filesystem**: path → contents. A memfs keeps the personality self-contained and
    /// deterministic (the playground has no disk); a native embedder routing to a real `fs` cap is a
    /// follow-up. Shared file bytes; per-fd offsets live in [`Inner::fds`].
    files: HashMap<String, Vec<u8>>,
    /// Explicitly-created **empty** directories (`mkdir`). The memfs otherwise infers directories as
    /// prefixes of file keys, which can't represent a dir with no files under it; this set carries those.
    /// A path is a directory if it is the root, appears here, or is a proper prefix of some file key.
    explicit_dirs: HashSet<String>,
    /// The host-side fd table (indexed by fd). Seeded with the three stdio sentinels at `0`/`1`/`2`
    /// (`FdEntry::Stdin`/`Stdout`/`Stderr`), so `dup2`/`dup`/`close`/`fcntl` treat every fd uniformly.
    /// `open`/`pipe`/`dup` allocate the lowest free slot; a closed fd (including a closed stdio fd) is
    /// reused, matching POSIX "lowest available".
    fds: Vec<Option<FdEntry>>,
    /// Loopback memnet listeners: bound port → the pending-connection queue its `accept` pops and a
    /// loopback `connect` pushes into. (The queue is shared with the listener's `FdEntry`; this index
    /// exists so `connect` can find it by port and `bind` can detect `-EADDRINUSE`.)
    net_listeners: HashMap<u16, Arc<Mutex<VecDeque<MemSock>>>>,
    /// The next ephemeral port a `bind :0` (or a connect's synthesized source) is assigned from.
    net_next_port: u16,
    /// The embedder's network delegate ([`Posix::set_net`]) — authority beyond loopback. `None` ⇒
    /// non-loopback fails closed.
    net_delegate: Option<Box<dyn NetDelegate>>,
    /// Open directory streams (`opendir`/`readdir`/`closedir`), indexed by the `DIR*`-analog handle
    /// `opendir` returns. Each holds the immediate child names snapshotted at `opendir` time and a
    /// read cursor. Separate from [`Inner::fds`] (a directory stream is not a file fd here).
    dirs: Vec<Option<DirStream>>,
    /// The program's argument vector (`args[0]` is the program name), delivered **host-side** — the
    /// symmetric analogue of the environment: `argc`/`argv` read it, the embedder sets it. This is how
    /// a personality program gets `sh -c "…"` without the window args buffer (POSIX.md §5); a guest
    /// crt that wants a standard `main(int, char**)` builds `argv[]` from these ops.
    args: Vec<String>,
    /// The current working directory `getcwd` reports and `chdir` updates. A plain string — the memfs
    /// is flat (paths are used as-given), so `cwd` is not validated against it; path normalization/
    /// resolution is a follow-up (POSIX.md §6).
    cwd: String,
    /// The environment: `name → value`. `getenv`/`setenv` read and update it; host-side, out of the
    /// guest's reach, like the rest of the bookkeeping (POSIX.md §3).
    env: HashMap<String, String>,
    /// The monotonic-clock base — `clock(1)` reports nanos elapsed since this. Captured at creation.
    clock_base: std::time::Instant,
    /// A pinned clock value (`Some(nanos)`) for determinism: when set, `clock(_)` returns it verbatim
    /// so a differential run is reproducible. `None` reads the real host clock.
    clock_fixed: Option<i64>,
    /// Cache of `getenv` results already materialized into the window: `name → ptr`. C's `getenv`
    /// returns a stable `char*` into libc-owned storage, so a repeated `getenv("X")` must return the
    /// **same** pointer; we allocate a NUL-terminated copy in the arena once and reuse it. `setenv`
    /// invalidates the entry so the next `getenv` re-materializes the new value.
    env_ptrs: HashMap<String, u64>,
    /// The **PATH registry** (STAGE1.md §5): command name → `(granted Module handle, declared window
    /// size_log2)`. `exec_lookup` returns the handle; `exec_win` the size_log2 (so the shell carves each
    /// spawn to the command's own window). The embedder seeds it with [`Posix::register_command`] after
    /// granting each command `Module`. A plain `Vec` (a shell's PATH is short); first match wins.
    commands: Vec<(String, i32, u8)>,
    /// The `Stream` handle `exec_stdout` returns — the stdout the shell re-grants to a spawned child
    /// under the name `"stdout"`. Set by [`Posix::set_exec_stdout`]; `0` until then (a shell that never
    /// spawns never reads it).
    exec_stdout_handle: i32,
    /// The read-only-pipe handle `exec_stdin` returns (the child's `"stdin"`) and its shared FIFO. Set
    /// by [`Posix::set_exec_stdin`] from [`Host::grant_input_pipe`]; `exec_stdin(ptr, len)` pushes the
    /// guest bytes into `fifo` and returns `handle`. `None`/`0` until set (a shell with no filter
    /// commands never calls it). The FIFO is fully drained by the child before the synchronous spawn
    /// returns, so it is empty and ready for the next command.
    exec_stdin_handle: i32,
    exec_stdin_fifo: Option<Arc<Mutex<VecDeque<u8>>>>,
    /// The embedder-wired **spawn delegate** ([`Posix::set_spawn`]) — the authority `spawn` needs to run
    /// a child. `None` until wired, in which case `spawn` is `-ENOSYS` (fail closed).
    spawn_fn: Option<SpawnFn>,
    /// Reaped-pending children: synthetic `pid → wait-encoded status`. `spawn` runs the child to
    /// completion and records its status here; `waitpid`/`wait` remove and return it.
    children: HashMap<i32, i32>,
    /// The next synthetic pid `spawn` hands out. Starts at `1000` (well clear of small fd/int values, so
    /// a pid is never confused with an fd in a test).
    next_pid: i32,
    /// **Pending signals** — the L0 doorbell. Bit `s` set ⇒ signal `s` has been raised (`kill`/
    /// [`Posix::raise_signal`]) and not yet polled. Signals are `1..=63` (one `u64`).
    sig_pending: u64,
    /// **Signal dispositions**: `signum → handler` (`SIG_DFL`/`SIG_IGN`/a guest handler pointer), set by
    /// `signal`/`sigaction`. Absent ⇒ `SIG_DFL`. `sigcheck` consults this to deliver caught signals and
    /// drop the rest.
    sig_handler: HashMap<i32, i64>,
    /// **Signal mask** — the blocked set (`sigprocmask`, #796). Bit `s` set ⇒ signal `s` is blocked: a
    /// pending blocked signal is held (not delivered by `sigcheck`) until unblocked. `SIGKILL`/`SIGSTOP`
    /// are never blockable, so those bits stay clear.
    sig_mask: u64,
    /// **Per-signal `sigaction` extras** (`sa_mask` / `sa_flags`), kept for round-trip fidelity (`oldact`).
    /// The poll model does not yet auto-block `sa_mask` while a handler runs, nor act on `SA_RESTART` —
    /// those land with the L2 async-delivery / L1 EINTR slices of #796.
    sig_action_mask: HashMap<i32, u64>,
    sig_action_flags: HashMap<i32, i64>,
    /// #796 L2 — the guest's registered **signal-handler stack** base (`sigaltstack`), the data-SP an async
    /// handler runs on. `0` ⇒ no stack ⇒ async delivery is off (poll-only). Distinct from the guest's normal
    /// stack because the interp can't compute where the interrupted frame's stack ends.
    sig_stack_base: u64,
    /// #796 L2 — the **armed** flag shared with the interp ([`Host::sig_armed`]): set when a caught,
    /// unmasked signal may be deliverable, so the interp's per-op poll knows to ask [`SignalSource`]. The
    /// same `Arc` is handed to the `Host` at grant time.
    sig_armed: Arc<AtomicBool>,
    /// #799 L1 (embedder `^C`) — the interp's **scheduler-wake** closure, installed at run start via
    /// [`SignalSource::set_wake`] and cleared to a no-op at teardown. [`Posix::raise_signal`] invokes it
    /// after raising a **deliverable** signal, so an embedder `^C` interrupts a parked blocking syscall
    /// even when every fiber is parked (no per-op safepoint to notice the arm). `None` until a run installs
    /// it (and between runs).
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

/// A handle to a granted POSIX personality's shared state — read the captured output after a run.
/// Cheap to clone (shares one `Arc`).
#[derive(Clone)]
pub struct Posix {
    inner: Arc<Mutex<Inner>>,
}

impl Posix {
    /// Bytes the guest `write`-to-fd-1'd — from the shared sink when one is set ([`Posix::set_stdout_sink`]),
    /// else the personality's own captured buffer.
    pub fn stdout(&self) -> Vec<u8> {
        let st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match &st.stdout_sink {
            Some(sink) => sink.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            None => st.stdout.clone(),
        }
    }
    /// Bytes the guest `write`-to-fd-2'd.
    pub fn stderr(&self) -> Vec<u8> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stderr
            .clone()
    }

    /// Seed (or overwrite) a memfs file — how an embedder/test stages the filesystem a guest `open`s.
    pub fn write_file(&self, path: &str, bytes: &[u8]) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .files
            .insert(path.to_string(), bytes.to_vec());
    }

    /// Read a memfs file back — how an embedder/test inspects what the guest wrote.
    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .files
            .get(path)
            .cloned()
    }

    /// Seed (or overwrite) an environment variable — how an embedder/test stages the environment a
    /// guest `getenv`s. Invalidates any cached `getenv` pointer for the name.
    pub fn set_env(&self, name: &str, value: &str) {
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        st.env_ptrs.remove(name);
        st.env.insert(name.to_string(), value.to_string());
    }

    /// Pin the clock to a fixed `nanos` value (all `clock(_)` calls return it) — how a test makes
    /// `std::time` deterministic. Passing a value makes a differential run reproducible.
    pub fn set_clock(&self, nanos: i64) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clock_fixed = Some(nanos);
    }

    /// The current working directory — how an embedder/test observes a guest `chdir`.
    pub fn cwd(&self) -> String {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cwd
            .clone()
    }

    /// Set the program's argument vector (`args[0]` is conventionally the program name) — how an
    /// embedder hands a personality program its `argv` (e.g. `["sh", "-c", "echo hi"]`), read back by
    /// the guest through the `argc`/`argv` ops.
    pub fn set_args(&self, args: &[&str]) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).args =
            args.iter().map(|s| s.to_string()).collect();
    }

    /// Route fd-1 (`stdout`) writes to a **shared sink** instead of the personality's own buffer, so a
    /// spawned child's re-granted `Stream` output and the shell's own output land in one stream
    /// (STAGE1.md §5). Pass the `Host`'s shared stdout (`Host::shared_stdout()`). [`Posix::stdout`] then
    /// reads this sink back.
    pub fn set_stdout_sink(&self, sink: Arc<Mutex<Vec<u8>>>) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stdout_sink = Some(sink);
    }

    /// Register a command in the PATH registry (STAGE1.md §5): `name → module_handle`, where
    /// `module_handle` is the handle a granted command `Module` has in the shell's cap table. The
    /// shell's `exec_lookup(name)` returns it (or `-1` when absent). A later registration of the same
    /// name shadows the earlier (last wins), matching a `PATH` re-export.
    pub fn register_command(&self, name: &str, module_handle: i32, win_log2: u8) {
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        st.commands.retain(|(n, _, _)| n != name);
        st.commands
            .push((name.to_string(), module_handle, win_log2));
    }

    /// Set the `Stream` handle the shell re-grants to a spawned child as its `"stdout"` — what
    /// `exec_stdout()` returns. Grant the `Stream` on the same `Host`, routed to the shared sink
    /// (`set_stdout_sink`), so the child's output joins the shell's (STAGE1.md §5).
    pub fn set_exec_stdout(&self, handle: i32) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .exec_stdout_handle = handle;
    }

    /// Set the read-only-pipe `Stream` + FIFO the shell re-grants to a **filter** command as `"stdin"`
    /// — the input twin of [`Self::set_exec_stdout`]. `handle` and `fifo` come from one
    /// [`Host::grant_input_pipe`] call on the same `Host`; `exec_stdin(ptr, len)` pushes the guest bytes
    /// into `fifo` and returns `handle`, so the child's `read(0, …)` drains them (then EOF).
    pub fn set_exec_stdin(&self, handle: i32, fifo: Arc<Mutex<VecDeque<u8>>>) {
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        st.exec_stdin_handle = handle;
        st.exec_stdin_fifo = Some(fifo);
    }

    /// Wire the **spawn delegate** — the authority the `spawn` op needs to run a child (POSIX.md ops
    /// 27–29). `f(name, argv, stdin) -> SpawnResult` runs the named command to completion and returns its
    /// captured stdout + exit status; the personality routes the stdout to the caller's current fd 1 and
    /// records the status for `waitpid`. Opt-in like [`Self::set_exec_stdout`]: until it is set, `spawn`
    /// is `-ENOSYS`. The embedder supplies whatever *running a child* means (an `Instantiator` op-13
    /// instantiate + `join`, a scripted table, a real subprocess).
    pub fn set_spawn<F>(&self, f: F)
    where
        F: FnMut(&str, &[String], &[u8]) -> SpawnResult + Send + 'static,
    {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .spawn_fn = Some(Box::new(f));
    }

    /// Wire the **network delegate** — the authority for anything beyond the loopback memnet
    /// (POSIX.md §5a; the [`Self::set_spawn`] analog). Until one is set, a non-loopback `connect`
    /// is `-ECONNREFUSED` and a non-`localhost` `resolve` is `-ENOENT` — fail closed. The delegate
    /// is where policy lives: a real socket, a scripted table, an allowlisting proxy.
    pub fn set_net(&self, delegate: impl NetDelegate + 'static) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .net_delegate = Some(Box::new(delegate));
    }

    /// Raise a signal from the **embedder** — how a terminal `^C` (SIGINT) or a `kill(1)` reaches the
    /// guest (the L0 doorbell's external door, the twin of the guest's own `kill` op). Sets the pending
    /// bit; the guest delivers it at its next `sigcheck` poll if it has a handler installed. Out-of-range
    /// `signum` (`< 1` or `> 63`) is ignored.
    pub fn raise_signal(&self, signum: i32) {
        if !(1..=63).contains(&signum) {
            return;
        }
        // Set pending + arm under the lock, then decide (still the personality's policy) whether this is
        // *deliverable* — a caught, unmasked signal with async delivery on. If so, grab the interp's
        // scheduler-wake closure and invoke it **after releasing the lock** (it locks the scheduler + the
        // parked vCPUs' hosts, a distinct lock order), so a blocked syscall parked with no running fiber to
        // notice the arm gets interrupted → `-EINTR` (#799 L1, the terminal `^C`). An ignored/masked
        // signal is not deliverable, so `wake` stays untouched and nothing is interrupted.
        let wake = {
            let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            st.sig_pending |= 1 << signum;
            st.arm_signals(); // #796 L2 — deliverable async, not just at the next poll
            if st.deliverable_now() {
                st.wake.clone()
            } else {
                None
            }
        };
        if let Some(w) = wake {
            w();
        }
    }
}

/// The §7 import-name policy for the POSIX subset: maps libc symbol names to the
/// [`cap_id::HOST_PROC`] capability + op — the name vocabulary [`bind`] installs as slot bindings.
/// Unknown names return `None`, so binding fails closed. Both bare (`"write"`) and `"posix."`-
/// prefixed names resolve, so it works whether the frontend emits raw libc symbols or namespaced ones.
pub fn resolve(name: &str) -> Option<ResolvedCap> {
    let bare = name.strip_prefix("posix.").unwrap_or(name);
    let op = match bare {
        "write" => OP_WRITE,
        "read" => OP_READ,
        "malloc" => OP_MALLOC,
        "free" => OP_FREE,
        "exit" | "_exit" | "_Exit" => OP_EXIT,
        "open" => OP_OPEN,
        "close" => OP_CLOSE,
        "lseek" => OP_LSEEK,
        "unlink" | "remove" => OP_UNLINK,
        "mkdir" => OP_MKDIR,
        "rename" => OP_RENAME,
        "rmdir" => OP_RMDIR,
        "getcwd" => OP_GETCWD,
        "chdir" => OP_CHDIR,
        "getenv" => OP_GETENV,
        "setenv" => OP_SETENV,
        "getenv_r" => OP_GETENV_R,
        "unsetenv" => OP_UNSETENV,
        "environ" => OP_ENVIRON,
        "clock_gettime" | "clock" => OP_CLOCK,
        "stat" | "lstat" => OP_STAT,
        "opendir" => OP_OPENDIR,
        "readdir" => OP_READDIR,
        "closedir" => OP_CLOSEDIR,
        // Personality extensions (not standard libc functions): the host-side argument vector, the
        // symmetric analogue of `getenv`/`environ`. A guest crt reads these to build `main`'s argv.
        "argc" => OP_ARGC,
        "argv" => OP_ARGV,
        // Personality `exec` surface (STAGE1.md §5): PATH lookup + the forwardable stdout handle, so a
        // shell on this personality can spawn an external command via `Instantiator` op 13.
        "exec_lookup" => OP_EXEC_LOOKUP,
        "exec_stdout" => OP_EXEC_STDOUT,
        "exec_stdin" => OP_EXEC_STDIN,
        "exec_win" => OP_EXEC_WIN,
        "pipe" => OP_PIPE,
        "dup2" => OP_DUP2,
        "dup" => OP_DUP,
        "fcntl" => OP_FCNTL,
        "spawn" | "posix_spawn" | "posix_spawnp" => OP_SPAWN,
        "waitpid" => OP_WAITPID,
        "wait" => OP_WAIT,
        "signal" => OP_SIGNAL,
        "kill" => OP_KILL,
        "sigcheck" => OP_SIGCHECK,
        "sigprocmask" => OP_SIGPROCMASK,
        "sigaction" => OP_SIGACTION,
        "sigaltstack" => OP_SIGALTSTACK,
        _ => return None,
    };
    Some(ResolvedCap {
        type_id: cap_id::HOST_PROC,
        op,
    })
}

/// Bind a manifest module's import slots to this personality (IMPORTS.md): each import name maps
/// through [`resolve`] to its `(HOST_PROC, op)` on the granted personality `handle`, installed as
/// instance bindings ([`Host::set_import_bindings`]) — the module bytes are never modified and its
/// `call.import`s dispatch through the slots. The guest declares plain libc signatures and never
/// threads a handle argument; the import section is its capability manifest. Returns `false`
/// (nothing installed, fail-closed) on a non-POSIX import name. Call **after** [`grant`]
/// (binding needs the granted handle — the §7 "binding happens once, at instantiation" ordering).
pub fn bind(m: &svm_ir::Module, host: &mut Host, handle: i32) -> bool {
    bind_with_fork(m, host, handle, None)
}

/// FORK.md PR 5 — [`bind`] plus the **`fork` endpoint**: an import named `fork` (or `posix.fork`)
/// binds not to the shared libc `HOST_PROC` handle but to the supplied **live fork offer**
/// `(type_id, handle)` — an offer the domain's personality-provider/parent wired over its own serve
/// export, whose handler calls `clone_caller(0)` (pid mode: the caller's `fork()` returns the twin's
/// id in the parent copy and `0` in the child copy; `-EAGAIN` if the domain is not forkable). The
/// provider-side contract is exactly the interp's pid-mode `clone_caller`; this function is only the
/// name-binding half. A module importing `fork` with no offer supplied fails the bind closed.
pub fn bind_with_fork(
    m: &svm_ir::Module,
    host: &mut Host,
    handle: i32,
    fork: Option<(u32, i32)>,
) -> bool {
    let mut binds = Vec::with_capacity(m.imports.len());
    for i in &m.imports {
        let bare = i.name.strip_prefix("posix.").unwrap_or(&i.name);
        if bare == "fork" {
            let Some((tid, fh)) = fork else { return false };
            binds.push(svm_interp::BoundImport::required(tid, 0, fh));
        } else {
            let Some(c) = resolve(&i.name) else {
                return false;
            };
            binds.push(svm_interp::BoundImport::required(c.type_id, c.op, handle));
        }
    }
    host.set_import_bindings(binds);
    true
}

/// Grant a POSIX personality on `host`, returning the `HOST_PROC` handle and a [`Posix`] handle to its
/// captured state. `heap_base`/`heap_end` bound the window region `malloc` hands out (both window
/// offsets, `heap_base <= heap_end`, within the guest window and clear of the guest's static
/// data/stack). `stdin` preloads standard input for `read(0, …)`. Every libc import in a linked module
/// shares this **one** handle (svm-wasm/chibicc thread a single capability handle); the op number
/// distinguishes the call, so pass the handle as the entry's leading argument.
/// #796 L2 — the interp-facing async-signal door: the [`SignalSource`] the `Host` holds. It locks the
/// shared `Inner` on demand (the interp holds no personality lock at a safepoint) and returns the next
/// deliverable handler, keeping POSIX signal *policy* (dispositions, mask, pending) inside this
/// personality while the interp only performs the *mechanism* (the safepoint redirect, invariant 4).
struct SignalDoor(Arc<Mutex<Inner>>);

impl SignalSource for SignalDoor {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        let mut st = self.0.lock().unwrap_or_else(|e| e.into_inner());
        // No signal stack registered ⇒ async delivery is off (poll-only): leave pending signals for the
        // guest's own `sigcheck` loop, and disarm so the interp stops asking.
        if st.sig_stack_base == 0 {
            st.sig_armed.store(false, Ordering::Relaxed);
            return None;
        }
        let sp = st.sig_stack_base;
        loop {
            let deliverable = st.sig_pending & !st.sig_mask;
            if deliverable == 0 {
                st.sig_armed.store(false, Ordering::Relaxed);
                return None; // nothing pending, or all pending signals blocked (held)
            }
            let s = deliverable.trailing_zeros() as i32;
            st.sig_pending &= !(1u64 << s);
            let handler = st.sig_handler.get(&s).copied().unwrap_or(SIG_DFL);
            if handler > SIG_IGN {
                // A caught handler: deliver it. Re-arm iff another signal is already deliverable, so the
                // interp returns for it once this handler completes (its in-handler guard then clears).
                let more = (st.sig_pending & !st.sig_mask) != 0;
                st.sig_armed.store(more, Ordering::Relaxed);
                return Some((handler as i32, s, sp));
            }
            // SIG_DFL / SIG_IGN: dropped in L0 (no default actions yet), keep scanning for a caught one.
        }
    }

    /// #799 L1 — store the interp's scheduler-wake closure so [`Posix::raise_signal`] can interrupt a
    /// parked blocking syscall on an embedder `^C`. Installed at run start, cleared to a no-op at teardown.
    fn set_wake(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).wake = Some(wake);
    }
}

pub fn grant(host: &mut Host, heap_base: u64, heap_end: u64, stdin: Vec<u8>) -> (i32, Posix) {
    let inner = Arc::new(Mutex::new(new_inner(heap_base, heap_end, stdin)));
    let posix = Posix {
        inner: Arc::clone(&inner),
    };
    // #796 L2 — install the async-signal source + the shared `armed` flag (the *same* `Arc` the
    // personality mutates), so the interp can redirect into a handler at a safepoint (PROCESS.md §9 L2).
    let armed = inner
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .sig_armed
        .clone();
    host.set_signal_source(Arc::new(SignalDoor(Arc::clone(&inner))), armed);
    // FORK.md PR 5 — grant **forkable**: the factory re-mints the handler over the *same* shared
    // `Inner` (fd table + memfs + cwd/env + captured output), so a `fork()` twin inherits the whole
    // libc state, shared — POSIX fork-shares-open-file-descriptions. `Host::fork_powerbox` calls this
    // factory to carry libc into the twin, instead of failing closed on an opaque closure.
    let make = move || handler(Arc::clone(&inner));
    let handle = host.grant_host_proc_forkable(make(), Arc::new(make));
    (handle, posix)
}

/// Build the personality as a re-grantable **capability factory** for the powerbox model (svm-run's
/// `HostCap`), instead of granting it on a specific `Host` by name binding like [`grant`]. Returns a
/// shared [`Posix`] handle (read captured output, `set_spawn`, `raise_signal`) and a `make` closure that
/// produces the `HostProc` handler over the *same* shared state each time it is called (once per backend,
/// so the interp and JIT hosts share one personality state). This is how the **LLVM on-ramp** reaches
/// the personality: the embedder wraps `make` in a `HostCap` at [`cap_id::HOST_PROC`] and grants it under a
/// name (e.g. `"posix"`), and an on-ramp guest calls `__vm_host_call(__vm_cap_resolve("posix"), op, …)`.
pub fn cap(
    heap_base: u64,
    heap_end: u64,
    stdin: Vec<u8>,
) -> (Posix, impl Fn() -> HostProc + Send + Sync + 'static) {
    let inner = Arc::new(Mutex::new(new_inner(heap_base, heap_end, stdin)));
    let posix = Posix {
        inner: Arc::clone(&inner),
    };
    let make = move || handler(Arc::clone(&inner));
    (posix, make)
}

/// The **`net` capability** as a factory over an existing personality (POSIX.md §5a) — the same
/// shape as [`cap`], granted under its **own name** (e.g. `"net"`). Each call produces a `HostProc`
/// over the *same* shared state, so the socket fds it mints live in the same fd table the libc
/// `read`/`write`/`close`/`dup2` ops serve — the data plane needs no new surface.
pub fn net_cap_factory(posix: &Posix) -> impl Fn() -> HostProc + Send + Sync + 'static {
    let inner = Arc::clone(&posix.inner);
    move || net_handler(Arc::clone(&inner))
}

/// Build the `net` capability's [`HostProc`] handler over shared `inner`: the tiny authority surface
/// (connect / bind / accept / shutdown / resolve). An unknown op is a clean `CapFault`.
fn net_handler(inner: Arc<Mutex<Inner>>) -> HostProc {
    Box::new(
        move |op, args, mem, _minter: Option<&mut dyn svm_interp::RegionMinter>| {
            let mut st = inner.lock().unwrap_or_else(|e| e.into_inner());
            match op {
                NET_CONNECT => st.net_connect(args, mem),
                NET_BIND => st.net_bind(args, mem),
                NET_ACCEPT => st.net_accept(args, mem),
                NET_SHUTDOWN => Ok(vec![st.net_shutdown(args)]),
                NET_RESOLVE => st.net_resolve(args, mem),
                _ => Err(Trap::CapFault),
            }
        },
    )
}

/// A fresh personality state: preloaded `stdin`, the window-heap arena bounded by `[heap_base, heap_end)`,
/// and the three stdio sentinels seeded in the fd table. Shared by [`grant`] and [`cap`].
fn new_inner(heap_base: u64, heap_end: u64, stdin: Vec<u8>) -> Inner {
    Inner {
        stdout: Vec::new(),
        stdout_sink: None,
        stderr: Vec::new(),
        stdin,
        stdin_pos: 0,
        heap_next: heap_base,
        heap_end,
        allocated: HashMap::new(),
        free_list: Vec::new(),
        files: HashMap::new(),
        explicit_dirs: HashSet::new(),
        fds: vec![
            Some(FdEntry::Stdin),
            Some(FdEntry::Stdout),
            Some(FdEntry::Stderr),
        ],
        net_listeners: HashMap::new(),
        net_next_port: 49152, // the IANA ephemeral range start
        net_delegate: None,
        dirs: Vec::new(),
        args: Vec::new(),
        cwd: "/".to_string(),
        env: HashMap::new(),
        clock_base: std::time::Instant::now(),
        clock_fixed: None,
        env_ptrs: HashMap::new(),
        commands: Vec::new(),
        exec_stdout_handle: 0,
        exec_stdin_handle: 0,
        exec_stdin_fifo: None,
        spawn_fn: None,
        children: HashMap::new(),
        next_pid: 1000,
        sig_pending: 0,
        sig_handler: HashMap::new(),
        sig_mask: 0,
        sig_action_mask: HashMap::new(),
        sig_action_flags: HashMap::new(),
        sig_stack_base: 0,
        sig_armed: Arc::new(AtomicBool::new(false)),
        wake: None,
    }
}

/// Build the POSIX [`HostProc`] handler over shared `inner`. Dispatches on the op number; an unknown op
/// on this handle is a `CapFault` (as for any capability).
fn handler(inner: Arc<Mutex<Inner>>) -> HostProc {
    Box::new(
        move |op, args, mem, _minter: Option<&mut dyn svm_interp::RegionMinter>| {
            let mut st = inner.lock().unwrap_or_else(|e| e.into_inner());
            match op {
                OP_WRITE => st.write(args, mem),
                OP_READ => st.read(args, mem),
                OP_MALLOC => Ok(vec![st.malloc(args)]),
                OP_FREE => {
                    st.free(args);
                    Ok(vec![0])
                }
                OP_EXIT => Err(Trap::Exit(args.first().copied().unwrap_or(0) as i32)),
                OP_OPEN => st.open(args, mem),
                OP_CLOSE => Ok(vec![st.close(args)]),
                OP_LSEEK => Ok(vec![st.lseek(args)]),
                OP_UNLINK => st.unlink(args, mem),
                OP_MKDIR => st.mkdir(args, mem),
                OP_RENAME => st.rename(args, mem),
                OP_RMDIR => st.rmdir(args, mem),
                OP_STAT => st.stat(args, mem),
                OP_OPENDIR => st.opendir(args, mem),
                OP_READDIR => st.readdir(args, mem),
                OP_CLOSEDIR => Ok(vec![st.closedir(args)]),
                OP_ARGC => Ok(vec![st.args.len() as i64]),
                OP_ARGV => st.argv(args, mem),
                OP_EXEC_LOOKUP => st.exec_lookup(args, mem),
                OP_EXEC_STDOUT => Ok(vec![st.exec_stdout_handle as i64]),
                OP_EXEC_STDIN => st.exec_stdin(args, mem),
                OP_EXEC_WIN => st.exec_win(args),
                OP_PIPE => st.pipe(args, mem),
                OP_DUP2 => Ok(vec![st.dup2(args)]),
                OP_DUP => Ok(vec![st.dup(args)]),
                OP_FCNTL => Ok(vec![st.fcntl(args)]),
                OP_SPAWN => st.spawn(args, mem),
                OP_WAITPID => st.waitpid(args, mem),
                OP_WAIT => st.waitpid(&[-1, *args.first().unwrap_or(&0), 0], mem),
                OP_SIGNAL => Ok(vec![st.signal(args)]),
                OP_KILL => Ok(vec![st.kill(args)]),
                OP_SIGCHECK => Ok(vec![st.sigcheck()]),
                OP_SIGPROCMASK => st.sigprocmask(args, mem),
                OP_SIGACTION => st.sigaction(args, mem),
                OP_SIGALTSTACK => Ok(vec![st.sigaltstack(args)]),
                OP_GETCWD => st.getcwd(args, mem),
                OP_CHDIR => st.chdir(args, mem),
                OP_GETENV => st.getenv(args, mem),
                OP_SETENV => st.setenv(args, mem),
                OP_GETENV_R => st.getenv_r(args, mem),
                OP_UNSETENV => st.unsetenv(args, mem),
                OP_ENVIRON => st.environ(args, mem),
                OP_CLOCK => Ok(vec![st.clock(args)]),
                _ => Err(Trap::CapFault),
            }
        },
    )
}

impl Inner {
    /// `write(fd, buf, len) -> n | -errno`: the `Stdout`/`Stderr` sentinels append to the captured
    /// stdout/stderr; a `File` fd writes into its memfs file at the offset (extending it), advancing it;
    /// a `PipeWrite` fd appends to its shared buffer. `Stdin`, a `PipeRead`, a read-only file, and an
    /// unopened fd are `-EBADF`.
    fn write(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as u64;
        if len == 0 {
            return Ok(vec![0]);
        }
        let data = mem.read_bytes(buf, len).ok_or(Trap::Malformed)?;
        Ok(vec![self.sink_write(fd, &data)])
    }

    /// Write `data` to fd `fd`'s current binding, returning the count or `-EBADF`: the `Stdout`/`Stderr`
    /// sentinels append to captured stdout/stderr, a `File` writes at its offset, a `PipeWrite` appends to
    /// its shared buffer. Factored out of [`Inner::write`] so `spawn` can route a child's captured stdout
    /// to *whatever the caller's fd 1 currently is* (the fd-inheritance path). Empty `data` is a `0` no-op.
    fn sink_write(&mut self, fd: i64, data: &[u8]) -> i64 {
        if data.is_empty() {
            return 0;
        }
        // Decide the sink first (cloning the pipe `Arc`) so we don't hold a borrow of `self.fds` while
        // mutating `self.stdout`/`self.stderr`/the memfs.
        enum Sink {
            Stdout,
            Stderr,
            File,
            Pipe(PipeBuf),
            Net(MemSock),
            NetDelegate(Arc<Mutex<Box<dyn NetStream>>>),
            Bad,
        }
        let sink = match self.fd(fd) {
            Some(FdEntry::Stdout) => Sink::Stdout,
            Some(FdEntry::Stderr) => Sink::Stderr,
            Some(FdEntry::File(_)) => Sink::File,
            Some(FdEntry::PipeWrite(p)) => Sink::Pipe(Arc::clone(p)),
            Some(FdEntry::NetSock(s)) => Sink::Net(s.clone()),
            Some(FdEntry::NetStream(d)) => Sink::NetDelegate(Arc::clone(d)),
            _ => Sink::Bad,
        };
        match sink {
            Sink::Stdout => match &self.stdout_sink {
                Some(s) => s
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .extend_from_slice(data),
                None => self.stdout.extend_from_slice(data),
            },
            Sink::Stderr => self.stderr.extend_from_slice(data),
            Sink::File => return self.file_write(fd as usize, data),
            Sink::Pipe(p) => p
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(data.iter().copied()),
            // A memnet write appends to the peer-facing FIFO (unbounded); a shut-down write side
            // is `-EPIPE`, matching a closed peer.
            Sink::Net(s) => {
                if s.write_token.closed.load(Ordering::Acquire) {
                    return EPIPE;
                }
                s.tx.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .extend(data.iter().copied());
            }
            Sink::NetDelegate(d) => {
                return d.lock().unwrap_or_else(|e| e.into_inner()).send(data);
            }
            Sink::Bad => return EBADF,
        }
        data.len() as i64
    }

    /// `read(fd, buf, len) -> n | -errno`: the `Stdin` sentinel drains preloaded stdin; a `File` fd reads
    /// its memfs file from the offset, advancing it (`0` at EOF); a `PipeRead` fd drains its shared buffer
    /// (`0` when empty). `Stdout`/`Stderr`, a `PipeWrite`, and an unopened fd are `-EBADF`.
    fn read(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as usize;
        enum Src {
            Stdin,
            File,
            Pipe(PipeBuf),
            Net(MemSock),
            NetDelegate(Arc<Mutex<Box<dyn NetStream>>>),
            Bad,
        }
        let src = match self.fd(fd) {
            Some(FdEntry::Stdin) => Src::Stdin,
            Some(FdEntry::File(_)) => Src::File,
            Some(FdEntry::PipeRead(p)) => Src::Pipe(Arc::clone(p)),
            Some(FdEntry::NetSock(s)) => Src::Net(s.clone()),
            Some(FdEntry::NetStream(d)) => Src::NetDelegate(Arc::clone(d)),
            _ => Src::Bad,
        };
        let chunk: Vec<u8> = match src {
            Src::Stdin => {
                let avail = &self.stdin[self.stdin_pos.min(self.stdin.len())..];
                let n = len.min(avail.len());
                self.stdin_pos += n;
                avail[..n].to_vec()
            }
            Src::File => match self.file_read(fd as usize, len) {
                Ok(c) => c,
                Err(e) => return Ok(vec![e]),
            },
            Src::Pipe(p) => {
                let mut g = p.lock().unwrap_or_else(|e| e.into_inner());
                let n = len.min(g.len());
                g.drain(..n).collect()
            }
            // Memnet read: drain what's buffered; on empty, EOF (`0`) once the peer's write side is
            // gone or our read side is shut, else `-EAGAIN` (blocking would deadlock a cooperative
            // guest waiting on itself — POSIX.md §5a).
            Src::Net(s) => {
                if s.read_shut.load(Ordering::Acquire) {
                    Vec::new()
                } else {
                    let mut g = s.rx.lock().unwrap_or_else(|e| e.into_inner());
                    let n = len.min(g.len());
                    if n == 0 && !s.peer_write_closed.load(Ordering::Acquire) {
                        return Ok(vec![EAGAIN]);
                    }
                    g.drain(..n).collect()
                }
            }
            // A delegate-backed recv may block host-side (the embedder owns the real I/O).
            Src::NetDelegate(d) => {
                let mut tmp = vec![0u8; len];
                let n = d.lock().unwrap_or_else(|e| e.into_inner()).recv(&mut tmp);
                if n < 0 {
                    return Ok(vec![n]);
                }
                tmp.truncate(n as usize);
                tmp
            }
            Src::Bad => return Ok(vec![EBADF]),
        };
        mem.write_bytes(buf, &chunk).ok_or(Trap::Malformed)?;
        Ok(vec![chunk.len() as i64])
    }

    /// Borrow the entry at `fd` if it is a valid, open fd (`fd >= 0` and the slot is `Some`).
    fn fd(&self, fd: i64) -> Option<&FdEntry> {
        if fd < 0 {
            return None;
        }
        self.fds.get(fd as usize).and_then(|s| s.as_ref())
    }

    /// `open(path_ptr, path_len, flags) -> fd | -errno`: open (or `O_CREAT`) a memfs file, returning a
    /// fresh fd. `O_TRUNC` clears it, `O_APPEND` seeks to the end; a missing file without `O_CREAT` is
    /// `-ENOENT`, a non-UTF-8 path `-EINVAL`.
    fn open(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let flags = *args.get(2).ok_or(Trap::Malformed)?;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        let exists = self.files.contains_key(&path);
        if !exists && flags & O_CREAT == 0 {
            return Ok(vec![ENOENT]);
        }
        let file = self.files.entry(path.clone()).or_default();
        if flags & O_TRUNC != 0 {
            file.clear();
        }
        let pos = if flags & O_APPEND != 0 { file.len() } else { 0 };
        let acc = flags & O_ACCMODE;
        let writable = acc == O_WRONLY || acc == O_RDWR;
        Ok(vec![self.alloc_fd(FdEntry::File(OpenFile {
            path,
            pos,
            writable,
        }))])
    }

    /// `close(fd) -> 0 | -errno`: release any open fd (a file, a pipe end, or a stdio sentinel — a shell
    /// closes and reuses `0`/`1`/`2` freely). An out-of-range / already-closed fd is `-EBADF`.
    fn close(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&-1);
        if fd >= 0 {
            if let Some(slot) = self.fds.get_mut(fd as usize) {
                if let Some(entry) = slot.take() {
                    // Closing a memnet listener releases its port — but only if the registry still
                    // points at *this* listener's queue (a stale dup must not evict a later binder).
                    if let FdEntry::NetListener(l) = &entry {
                        if self
                            .net_listeners
                            .get(&l.addr.port)
                            .is_some_and(|q| Arc::ptr_eq(q, &l.pending))
                        {
                            self.net_listeners.remove(&l.addr.port);
                        }
                    }
                    return 0;
                }
            }
        }
        EBADF
    }

    /// `lseek(fd, offset, whence) -> new_offset | -errno`: reposition a `File` fd (`SEEK_SET`/`CUR`/`END`).
    /// A negative result or bad whence is `-EINVAL`; a pipe/stdio fd is `-ESPIPE`; an unopened fd `-EBADF`.
    fn lseek(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&-1);
        let offset = *args.get(1).unwrap_or(&0);
        let whence = *args.get(2).unwrap_or(&-1);
        let (path, pos) = match self.fd(fd) {
            Some(FdEntry::File(of)) => (of.path.clone(), of.pos as i64),
            Some(_) => return ESPIPE,
            None => return EBADF,
        };
        let size = self.files.get(&path).map_or(0, |f| f.len()) as i64;
        let newpos = match whence {
            SEEK_SET => offset,
            SEEK_CUR => pos + offset,
            SEEK_END => size + offset,
            _ => return EINVAL,
        };
        if newpos < 0 {
            return EINVAL;
        }
        if let Some(FdEntry::File(of)) = self.fds[fd as usize].as_mut() {
            of.pos = newpos as usize;
        }
        newpos
    }

    /// `pipe(fds_ptr) -> 0 | -errno`: create an in-personality byte FIFO and store the read and write fds
    /// (`[i32; 2]`, little-endian) at `fds_ptr`. Both ends share one [`PipeBuf`]; the write end is the
    /// higher fd, matching Linux (which allocates the read end first). Non-blocking (see [`PipeBuf`]).
    fn pipe(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let buf: PipeBuf = Arc::new(Mutex::new(VecDeque::new()));
        let rfd = self.alloc_fd(FdEntry::PipeRead(Arc::clone(&buf)));
        let wfd = self.alloc_fd(FdEntry::PipeWrite(buf));
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&(rfd as i32).to_le_bytes());
        out.extend_from_slice(&(wfd as i32).to_le_bytes());
        mem.write_bytes(ptr, &out).ok_or(Trap::Malformed)?;
        Ok(vec![0])
    }

    /// `dup2(oldfd, newfd) -> newfd | -errno`: re-point `newfd` at `oldfd`'s object, closing whatever
    /// `newfd` referred to. `dup2(fd, fd)` is a no-op returning `fd` (POSIX). `oldfd` must be open;
    /// `newfd` must be non-negative. This is the redirect primitive (`dup2(pipe_w, 1)` before a spawn).
    fn dup2(&mut self, args: &[i64]) -> i64 {
        let oldfd = *args.first().unwrap_or(&-1);
        let newfd = *args.get(1).unwrap_or(&-1);
        let Some(entry) = self.fd(oldfd) else {
            return EBADF;
        };
        if newfd < 0 {
            return EBADF;
        }
        if oldfd == newfd {
            return newfd;
        }
        let dup = entry.dup_clone();
        let n = newfd as usize;
        if self.fds.len() <= n {
            self.fds.resize_with(n + 1, || None);
        }
        self.fds[n] = Some(dup);
        newfd
    }

    /// `dup(oldfd) -> fd | -errno`: clone `oldfd` onto the lowest free fd. `oldfd` must be open.
    fn dup(&mut self, args: &[i64]) -> i64 {
        let oldfd = *args.first().unwrap_or(&-1);
        match self.fd(oldfd) {
            Some(entry) => {
                let dup = entry.dup_clone();
                self.alloc_fd(dup)
            }
            None => EBADF,
        }
    }

    /// `fcntl(fd, cmd, arg) -> result | -errno`: `F_DUPFD`/`F_DUPFD_CLOEXEC` clone `fd` onto the lowest
    /// free fd `>= arg`; `F_GETFD`/`F_GETFL` return `0`, `F_SETFD`/`F_SETFL` accept and return `0` (no
    /// exec-in-place here, so `FD_CLOEXEC`/status flags have nothing to gate). `fd` must be open.
    fn fcntl(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&-1);
        let cmd = *args.get(1).unwrap_or(&-1);
        let arg = *args.get(2).unwrap_or(&0);
        let Some(entry) = self.fd(fd) else {
            return EBADF;
        };
        match cmd {
            F_DUPFD | F_DUPFD_CLOEXEC => {
                let dup = entry.dup_clone();
                self.alloc_fd_from(dup, arg.max(0) as usize)
            }
            F_GETFD | F_GETFL | F_SETFD | F_SETFL => 0,
            _ => EINVAL,
        }
    }

    /// Drain **all** currently-available bytes from fd `fd`'s binding, advancing it: preloaded stdin (the
    /// `Stdin` sentinel), the rest of a `File`, or the whole of a `PipeRead` buffer. Anything else yields
    /// no bytes. This is how `spawn` hands the child its inherited stdin (fd 0).
    fn drain_fd(&mut self, fd: i64) -> Vec<u8> {
        enum Src {
            Stdin,
            File,
            Pipe(PipeBuf),
            None,
        }
        let src = match self.fd(fd) {
            Some(FdEntry::Stdin) => Src::Stdin,
            Some(FdEntry::File(_)) => Src::File,
            Some(FdEntry::PipeRead(p)) => Src::Pipe(Arc::clone(p)),
            _ => Src::None,
        };
        match src {
            Src::Stdin => {
                let out = self.stdin[self.stdin_pos.min(self.stdin.len())..].to_vec();
                self.stdin_pos = self.stdin.len();
                out
            }
            // A file has a bounded length; read from the offset to EOF in one shot.
            Src::File => {
                let n = match self.fds.get(fd as usize).and_then(|s| s.as_ref()) {
                    Some(FdEntry::File(of)) => self
                        .files
                        .get(&of.path)
                        .map_or(0, |f| f.len())
                        .saturating_sub(of.pos),
                    _ => 0,
                };
                self.file_read(fd as usize, n).unwrap_or_default()
            }
            Src::Pipe(p) => p
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .drain(..)
                .collect(),
            Src::None => Vec::new(),
        }
    }

    /// `spawn(name_ptr, name_len, argv_ptr, argv_len) -> pid | -errno`: run a registered command as a
    /// child via the embedder's [`spawn delegate`](Posix::set_spawn), inheriting the caller's fd 0
    /// (drained as the child's stdin) and fd 1 (its captured stdout is routed there). `argv` is the
    /// `argv_len` bytes at `argv_ptr` split on NUL (empty ⇒ `[name]`). Returns a synthetic pid whose
    /// status `waitpid` reaps. `-ENOSYS` if no delegate is wired; `-EINVAL` on a non-UTF-8 name.
    fn spawn(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let name_ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let name_len = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let argv_ptr = *args.get(2).unwrap_or(&0) as u64;
        let argv_len = (*args.get(3).unwrap_or(&0)).max(0) as u64;
        let name_bytes = mem.read_bytes(name_ptr, name_len).ok_or(Trap::Malformed)?;
        let Ok(name) = String::from_utf8(name_bytes) else {
            return Ok(vec![EINVAL]);
        };
        // argv: the blob split on NUL, trailing empties dropped; empty ⇒ [name] (argv[0] = program name).
        let mut argv: Vec<String> = if argv_len == 0 {
            Vec::new()
        } else {
            let blob = mem.read_bytes(argv_ptr, argv_len).ok_or(Trap::Malformed)?;
            blob.split(|&b| b == 0)
                .map(|s| String::from_utf8_lossy(s).into_owned())
                .collect()
        };
        while argv.last().is_some_and(|s| s.is_empty()) {
            argv.pop();
        }
        if argv.is_empty() {
            argv.push(name.clone());
        }
        // Fail closed *before* any side effect (draining stdin) if no delegate is wired.
        if self.spawn_fn.is_none() {
            return Ok(vec![ENOSYS]);
        }
        // The child inherits fd 0 as stdin — drain it before invoking the delegate.
        let stdin = self.drain_fd(0);
        // Take the delegate out to call it (a `&mut self` method cannot also borrow the boxed closure),
        // then restore it.
        let mut f = self.spawn_fn.take().unwrap();
        let res = f(&name, &argv, &stdin);
        self.spawn_fn = Some(f);
        // Route the child's stdout/stderr to the caller's current fd 1 / fd 2 (inheritance: a prior
        // `dup2(_, 1)` / `dup2(_, 2)` redirect lands each in a file or pipe; otherwise the stdio sink).
        self.sink_write(1, &res.stdout);
        self.sink_write(2, &res.stderr);
        let pid = self.next_pid;
        self.next_pid += 1;
        // Wait-encode the exit status: WEXITSTATUS occupies bits 8–15, low bits 0 (a normal exit).
        self.children.insert(pid, (res.status & 0xff) << 8);
        Ok(vec![pid as i64])
    }

    /// `waitpid(pid, status_ptr, options) -> pid | -errno`: reap `pid` (or any pending child when
    /// `pid == -1`), writing its wait-encoded status to `status_ptr` when non-null. `options` (e.g.
    /// `WNOHANG`) is ignored — a spawned child has already run to completion, so a reap never blocks.
    /// `-ECHILD` when there is no such child.
    fn waitpid(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let pid = *args.first().ok_or(Trap::Malformed)?;
        let status_ptr = *args.get(1).unwrap_or(&0) as u64;
        let reaped = if pid == -1 {
            self.children.keys().min().copied()
        } else if self.children.contains_key(&(pid as i32)) {
            Some(pid as i32)
        } else {
            None
        };
        let Some(p) = reaped else {
            return Ok(vec![ECHILD]);
        };
        let status: i32 = self.children.remove(&p).unwrap_or(0);
        if status_ptr != 0 {
            mem.write_bytes(status_ptr, &status.to_le_bytes())
                .ok_or(Trap::Malformed)?;
        }
        Ok(vec![p as i64])
    }

    /// `signal(signum, handler) -> prev_handler | -errno`: record `signum`'s disposition (`SIG_DFL`/
    /// `SIG_IGN`/a guest handler pointer) and return the previous (`SIG_DFL` if never set). Out-of-range
    /// `signum` is `-EINVAL`.
    fn signal(&mut self, args: &[i64]) -> i64 {
        let signum = *args.first().unwrap_or(&0);
        let handler = *args.get(1).unwrap_or(&SIG_DFL);
        if !(1..=63).contains(&signum) {
            return EINVAL;
        }
        let prev = self
            .sig_handler
            .insert(signum as i32, handler)
            .unwrap_or(SIG_DFL);
        self.arm_signals(); // #796 L2 — a handler installed for an already-pending signal may deliver now
        prev
    }

    /// `kill(pid, sig) -> 0 | -errno`: raise `sig` (set its pending bit). `pid` is advisory in this
    /// single-process doorbell (`raise(s)` is the guest `kill(0, s)`). Out-of-range `sig` is `-EINVAL`;
    /// `sig == 0` (the POSIX existence probe) is a `0` no-op.
    fn kill(&mut self, args: &[i64]) -> i64 {
        let sig = *args.get(1).unwrap_or(&0);
        if sig == 0 {
            return 0; // kill(pid, 0): liveness probe — the (single) process exists
        }
        if !(1..=63).contains(&sig) {
            return EINVAL;
        }
        self.sig_pending |= 1 << sig;
        self.arm_signals();
        0
    }

    /// #796 L2 — nudge the interp's per-op poll to check for delivery: set the shared `armed` flag when
    /// async delivery is *possible* (a signal stack is registered). `SignalSource::take_deliverable` is
    /// authoritative and disarms if nothing is actually deliverable, so an over-eager arm costs only one
    /// no-op poll. Called wherever a signal may become deliverable (raise / unblock / install a handler /
    /// register a stack).
    fn arm_signals(&self) {
        if self.sig_stack_base != 0 {
            self.sig_armed.store(true, Ordering::Relaxed);
        }
    }

    /// #799 L1 — is a signal **deliverable right now**? Non-destructive twin of [`SignalDoor::
    /// take_deliverable`]'s gate: a caught (`> SIG_IGN`), unmasked, pending signal with async delivery on
    /// (a signal stack registered). This is the personality's *policy* the embedder-`^C` path consults
    /// before poking the interp: an ignored or masked signal is **not** deliverable, so it never interrupts
    /// a blocked syscall.
    fn deliverable_now(&self) -> bool {
        if self.sig_stack_base == 0 {
            return false; // async delivery off (poll-only)
        }
        let mut d = self.sig_pending & !self.sig_mask;
        while d != 0 {
            let s = d.trailing_zeros() as i32;
            d &= !(1u64 << s);
            if self.sig_handler.get(&s).copied().unwrap_or(SIG_DFL) > SIG_IGN {
                return true;
            }
        }
        false
    }

    /// `sigaltstack(sp, size) -> 0` (#796 L2): register the guest's dedicated signal-handler stack (the
    /// data-SP an async handler runs on). `sp == 0` turns async delivery off (poll-only). `size` is
    /// advisory. Registering a stack may make already-pending caught signals deliverable, so re-arm.
    fn sigaltstack(&mut self, args: &[i64]) -> i64 {
        let sp = *args.first().unwrap_or(&0);
        self.sig_stack_base = sp.max(0) as u64;
        self.arm_signals();
        0
    }

    /// `sigcheck(_) -> handler | 0`: the L0 doorbell poll. Clear and return the handler pointer of the
    /// lowest-numbered pending **caught and unblocked** signal (`handler > SIG_IGN`); pending **ignored**
    /// (`SIG_IGN`) and **default** (`SIG_DFL`) signals are cleared and skipped (L0 does not deliver default
    /// actions). A pending **blocked** signal (`sig_mask`, #796) is **held** — neither delivered nor cleared
    /// — until `sigprocmask` unblocks it. `0` when nothing is deliverable — so the guest runs
    /// `((void(*)(void))handler)()` at its safe point.
    fn sigcheck(&mut self) -> i64 {
        loop {
            let deliverable = self.sig_pending & !self.sig_mask;
            if deliverable == 0 {
                return 0; // nothing pending, or all pending signals are blocked (held)
            }
            let s = deliverable.trailing_zeros() as i32;
            self.sig_pending &= !(1u64 << s);
            let handler = self.sig_handler.get(&s).copied().unwrap_or(SIG_DFL);
            if handler > SIG_IGN {
                return handler;
            }
            // SIG_DFL / SIG_IGN: dropped in L0, keep scanning for a caught, unblocked one.
        }
    }

    /// `sigprocmask(how, set, oldset) -> 0 | -errno` (#796): examine/change the blocked-signal set. Writes
    /// the current mask to `oldset` first (when non-null), then applies `set` per `how`. `SIGKILL`/`SIGSTOP`
    /// are silently kept unblocked ([`UNMASKABLE`]). A bad `how` (with a non-null `set`) is `-EINVAL`. The
    /// mask is a `u64` `sigset_t` read/written as 8 little-endian bytes.
    fn sigprocmask(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let how = *args.first().unwrap_or(&0);
        let set_ptr = *args.get(1).unwrap_or(&0) as u64;
        let oldset_ptr = *args.get(2).unwrap_or(&0) as u64;
        let mem = mem.ok_or(Trap::Malformed)?;
        if oldset_ptr != 0 {
            mem.write_bytes(oldset_ptr, &self.sig_mask.to_le_bytes())
                .ok_or(Trap::Malformed)?;
        }
        if set_ptr != 0 {
            let bytes = mem.read_bytes(set_ptr, 8).ok_or(Trap::Malformed)?;
            let set = u64::from_le_bytes(bytes.try_into().map_err(|_| Trap::Malformed)?);
            let new = match how {
                SIG_BLOCK => self.sig_mask | set,
                SIG_UNBLOCK => self.sig_mask & !set,
                SIG_SETMASK => set,
                _ => return Ok(vec![EINVAL]),
            };
            self.sig_mask = new & !UNMASKABLE; // SIGKILL/SIGSTOP can never be blocked
            self.arm_signals(); // #796 L2 — unblocking may free a held signal for async delivery
        }
        Ok(vec![0])
    }

    /// `sigaction(signum, act, oldact) -> 0 | -errno` (#796): the richer `signal`. Records `signum`'s
    /// disposition (`sa_handler`) plus its `sa_mask`/`sa_flags`, writing the previous action to `oldact`
    /// first (when non-null). The `struct sigaction` ABI is 24 bytes: `sa_handler` (i64@0), `sa_mask`
    /// (u64@8), `sa_flags` (i64@16). Out-of-range `signum` is `-EINVAL`. (The poll model does not yet
    /// auto-block `sa_mask` while the handler runs, nor honor `SA_RESTART` — those land with L2/L1.)
    fn sigaction(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let signum = *args.first().unwrap_or(&0);
        let act_ptr = *args.get(1).unwrap_or(&0) as u64;
        let oldact_ptr = *args.get(2).unwrap_or(&0) as u64;
        if !(1..=63).contains(&signum) {
            return Ok(vec![EINVAL]);
        }
        let s = signum as i32;
        let mem = mem.ok_or(Trap::Malformed)?;
        if oldact_ptr != 0 {
            let mut buf = [0u8; 24];
            let h = self.sig_handler.get(&s).copied().unwrap_or(SIG_DFL);
            let m = self.sig_action_mask.get(&s).copied().unwrap_or(0);
            let f = self.sig_action_flags.get(&s).copied().unwrap_or(0);
            buf[0..8].copy_from_slice(&h.to_le_bytes());
            buf[8..16].copy_from_slice(&m.to_le_bytes());
            buf[16..24].copy_from_slice(&f.to_le_bytes());
            mem.write_bytes(oldact_ptr, &buf).ok_or(Trap::Malformed)?;
        }
        if act_ptr != 0 {
            let b = mem.read_bytes(act_ptr, 24).ok_or(Trap::Malformed)?;
            let handler = i64::from_le_bytes(b[0..8].try_into().map_err(|_| Trap::Malformed)?);
            let mask = u64::from_le_bytes(b[8..16].try_into().map_err(|_| Trap::Malformed)?);
            let flags = i64::from_le_bytes(b[16..24].try_into().map_err(|_| Trap::Malformed)?);
            self.sig_handler.insert(s, handler);
            self.sig_action_mask.insert(s, mask);
            self.sig_action_flags.insert(s, flags);
            self.arm_signals(); // #796 L2 — a handler for an already-pending signal may deliver now
        }
        Ok(vec![0])
    }

    /// `unlink(path_ptr, path_len) -> 0 | -errno`: remove a memfs file. Already-open fds keep their
    /// (now-detached) contents via the file map only until closed — POSIX unlink-while-open nuance is a
    /// follow-up; here a removed path simply reads as absent to a fresh `open`. Missing file is `-ENOENT`.
    fn unlink(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        Ok(vec![if self.files.remove(&path).is_some() {
            0
        } else {
            ENOENT
        }])
    }

    /// `mkdir(path, plen, mode) -> 0 | -errno`: record an explicit empty directory. `mode` is ignored.
    /// `-EEXIST` if the path is already a file or directory; `-ENOENT` if the parent isn't a directory
    /// (`create_dir`, not `create_dir_all` — the std layer creates parents itself); `-EINVAL` for a
    /// non-UTF-8 path.
    fn mkdir(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        let norm = path.trim_end_matches('/').to_string();
        if norm.is_empty() {
            return Ok(vec![EEXIST]); // the root always exists
        }
        if self.files.contains_key(&norm) || self.is_dir(&norm) {
            return Ok(vec![EEXIST]);
        }
        let parent = match norm.rfind('/') {
            Some(0) | None => "/",
            Some(i) => &norm[..i],
        };
        if !self.is_dir(parent) {
            return Ok(vec![ENOENT]);
        }
        self.explicit_dirs.insert(norm);
        Ok(vec![0])
    }

    /// `rmdir(path, plen) -> 0 | -errno`: remove an empty directory. `-ENOTDIR` if it's a file, `-ENOENT`
    /// if it isn't a directory, `-ENOTEMPTY` if it still has children (an implicit dir always does),
    /// `-EINVAL` for the root or a non-UTF-8 path.
    fn rmdir(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        let norm = path.trim_end_matches('/').to_string();
        if norm.is_empty() {
            return Ok(vec![EINVAL]); // cannot remove the root
        }
        if self.files.contains_key(&norm) {
            return Ok(vec![ENOTDIR]);
        }
        if !self.is_dir(&norm) {
            return Ok(vec![ENOENT]);
        }
        if !self.dir_children(&norm).is_empty() {
            return Ok(vec![ENOTEMPTY]);
        }
        self.explicit_dirs.remove(&norm);
        Ok(vec![0])
    }

    /// `rename(old, olen, new, nlen) -> 0 | -errno`: move a file key (overwriting any existing target
    /// file) or a directory (re-keying every file and explicit subdir under it). `-ENOENT` if `old`
    /// doesn't exist; `-EINVAL` for a non-UTF-8 path.
    fn rename(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let old_ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let old_len = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let new_ptr = *args.get(2).ok_or(Trap::Malformed)? as u64;
        let new_len = (*args.get(3).ok_or(Trap::Malformed)?).max(0) as u64;
        let old_bytes = mem.read_bytes(old_ptr, old_len).ok_or(Trap::Malformed)?;
        let new_bytes = mem.read_bytes(new_ptr, new_len).ok_or(Trap::Malformed)?;
        let (Ok(old), Ok(new)) = (String::from_utf8(old_bytes), String::from_utf8(new_bytes))
        else {
            return Ok(vec![EINVAL]);
        };
        let old_n = old.trim_end_matches('/').to_string();
        let new_n = new.trim_end_matches('/').to_string();
        // File fast path: move the bytes, shadowing any explicit dir marker at the destination.
        if let Some(v) = self.files.remove(&old_n) {
            self.files.insert(new_n.clone(), v);
            self.explicit_dirs.remove(&new_n);
            return Ok(vec![0]);
        }
        // Directory: re-key every file and explicit subdir under `old_n`, plus the marker itself.
        if self.is_dir(&old_n) {
            let op = format!("{old_n}/");
            let np = format!("{new_n}/");
            let moved: Vec<String> = self
                .files
                .keys()
                .filter(|k| k.starts_with(&op))
                .cloned()
                .collect();
            for k in moved {
                let v = self.files.remove(&k).unwrap();
                self.files.insert(format!("{np}{}", &k[op.len()..]), v);
            }
            let dirs: Vec<String> = self
                .explicit_dirs
                .iter()
                .filter(|d| d.as_str() == old_n || d.starts_with(&op))
                .cloned()
                .collect();
            for d in dirs {
                self.explicit_dirs.remove(&d);
                let nd = if d == old_n {
                    new_n.clone()
                } else {
                    format!("{np}{}", &d[op.len()..])
                };
                self.explicit_dirs.insert(nd);
            }
            return Ok(vec![0]);
        }
        Ok(vec![ENOENT])
    }

    // ---- net ops (POSIX.md §5a) — the `net` capability's dispatch targets ----------------------

    /// The next free ephemeral port (49152..), skipping bound listeners; wraps within the range.
    fn net_alloc_ephemeral(&mut self) -> u16 {
        loop {
            let p = self.net_next_port;
            self.net_next_port = if p == u16::MAX { 49152 } else { p + 1 };
            if !self.net_listeners.contains_key(&p) {
                return p;
            }
        }
    }

    /// Write an address blob to the caller's out-buffer when one was supplied (`out != 0`); a buffer
    /// too small for the blob is `-ERANGE` (returned as `Err(errno)` for the caller to surface).
    fn net_write_addr(
        mem: &mut dyn GuestMem,
        out: u64,
        cap: u64,
        addr: &NetAddr,
    ) -> Result<Result<(), i64>, Trap> {
        if out == 0 {
            return Ok(Ok(()));
        }
        let enc = addr.encode();
        if enc.len() as u64 > cap {
            return Ok(Err(ERANGE));
        }
        mem.write_bytes(out, &enc).ok_or(Trap::Malformed)?;
        Ok(Ok(()))
    }

    /// `connect(addr, alen, laddr_out, cap) -> fd | -errno`: loopback → a memnet pair pushed onto the
    /// target listener's pending queue (`-ECONNREFUSED` if no listener holds the port); beyond
    /// loopback → the embedder's [`NetDelegate`] or `-ECONNREFUSED` (fail closed). Writes the
    /// connection's local address into `laddr_out`.
    fn net_connect(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let alen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let out = *args.get(2).unwrap_or(&0) as u64;
        let cap = (*args.get(3).unwrap_or(&0)).max(0) as u64;
        let blob = mem.read_bytes(ptr, alen).ok_or(Trap::Malformed)?;
        let Some(dst) = NetAddr::parse(&blob) else {
            return Ok(vec![EINVAL]);
        };
        if dst.is_loopback() {
            let Some(q) = self.net_listeners.get(&dst.port).map(Arc::clone) else {
                return Ok(vec![ECONNREFUSED]);
            };
            let src = NetAddr::loopback(self.net_alloc_ephemeral());
            let (client, server) = mem_pair(src, dst);
            q.lock()
                .unwrap_or_else(|e| e.into_inner())
                .push_back(server);
            if let Err(e) = Self::net_write_addr(mem, out, cap, &client.local)? {
                return Ok(vec![e]);
            }
            return Ok(vec![self.alloc_fd(FdEntry::NetSock(client))]);
        }
        let Some(d) = self.net_delegate.as_mut() else {
            return Ok(vec![ECONNREFUSED]);
        };
        match d.connect(&dst) {
            Ok(stream) => {
                // The delegate owns the real endpoint; synthesize an unspecified local address.
                let local = NetAddr {
                    v6: dst.v6,
                    port: 0,
                    addr: [0; 16],
                };
                if let Err(e) = Self::net_write_addr(mem, out, cap, &local)? {
                    return Ok(vec![e]);
                }
                Ok(vec![self.alloc_fd(FdEntry::NetStream(Arc::new(
                    Mutex::new(stream),
                )))])
            }
            Err(e) => Ok(vec![if e < 0 { e } else { ECONNREFUSED }]),
        }
    }

    /// `bind(addr, alen, bound_out, cap) -> listener_fd | -errno`: bind+listen folded. Loopback only
    /// in this slice (`-EACCES` beyond — the delegate-granted real-listener path is the noted
    /// follow-up); `:0` assigns an ephemeral port; a held port is `-EADDRINUSE`. Writes the actual
    /// bound address (so the guest learns its ephemeral port).
    fn net_bind(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let alen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let out = *args.get(2).unwrap_or(&0) as u64;
        let cap = (*args.get(3).unwrap_or(&0)).max(0) as u64;
        let blob = mem.read_bytes(ptr, alen).ok_or(Trap::Malformed)?;
        let Some(req) = NetAddr::parse(&blob) else {
            return Ok(vec![EINVAL]);
        };
        if !req.is_loopback() {
            return Ok(vec![EACCES]);
        }
        let port = if req.port == 0 {
            self.net_alloc_ephemeral()
        } else if self.net_listeners.contains_key(&req.port) {
            return Ok(vec![EADDRINUSE]);
        } else {
            req.port
        };
        let bound = NetAddr { port, ..req };
        if let Err(e) = Self::net_write_addr(mem, out, cap, &bound)? {
            return Ok(vec![e]);
        }
        let pending = Arc::new(Mutex::new(VecDeque::new()));
        self.net_listeners.insert(port, Arc::clone(&pending));
        Ok(vec![self.alloc_fd(FdEntry::NetListener(MemListener {
            addr: bound,
            pending,
        }))])
    }

    /// `accept(fd, peer_out, cap) -> fd | -EAGAIN | -errno`: pop the next pending memnet connection
    /// off the listener's queue (`-EAGAIN` when none — a cooperative guest cannot block on itself),
    /// writing the peer's address.
    fn net_accept(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let out = *args.get(1).unwrap_or(&0) as u64;
        let cap = (*args.get(2).unwrap_or(&0)).max(0) as u64;
        let q = match self.fd(fd) {
            Some(FdEntry::NetListener(l)) => Arc::clone(&l.pending),
            Some(_) => return Ok(vec![ENOTSOCK]),
            None => return Ok(vec![EBADF]),
        };
        let Some(sock) = q.lock().unwrap_or_else(|e| e.into_inner()).pop_front() else {
            return Ok(vec![EAGAIN]);
        };
        if let Err(e) = Self::net_write_addr(mem, out, cap, &sock.peer)? {
            return Ok(vec![e]);
        }
        Ok(vec![self.alloc_fd(FdEntry::NetSock(sock))])
    }

    /// `shutdown(fd, how) -> 0 | -errno` (`0` read / `1` write / `2` both, the Linux values): a
    /// write-shutdown flips the peer's empty reads to EOF; a read-shutdown makes our own reads
    /// return `0`. A delegate stream forwards to the embedder.
    fn net_shutdown(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&-1);
        let how = *args.get(1).unwrap_or(&2);
        match self.fd(fd) {
            Some(FdEntry::NetSock(s)) => {
                let s = s.clone();
                if how == 0 || how == 2 {
                    s.read_shut.store(true, Ordering::Release);
                }
                if how == 1 || how == 2 {
                    s.write_token.closed.store(true, Ordering::Release);
                }
                0
            }
            Some(FdEntry::NetStream(d)) => {
                let d = Arc::clone(d);
                let r = d.lock().unwrap_or_else(|e| e.into_inner()).shutdown(how);
                r
            }
            Some(_) => ENOTSOCK,
            None => EBADF,
        }
    }

    /// `resolve(name, nlen, out, cap) -> nbytes | -errno`: `localhost` → the v4 loopback; anything
    /// else → the delegate or `-ENOENT` (fail closed). Writes the address blobs back-to-back when
    /// they fit; the total byte length is returned either way (size-then-fetch, like `getenv_r`).
    fn net_resolve(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let nlen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let out = *args.get(2).unwrap_or(&0) as u64;
        let cap = (*args.get(3).unwrap_or(&0)).max(0) as u64;
        let bytes = mem.read_bytes(ptr, nlen).ok_or(Trap::Malformed)?;
        let Ok(name) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        let addrs: Vec<NetAddr> = if name == "localhost" {
            vec![NetAddr::loopback(0)]
        } else {
            match self.net_delegate.as_mut() {
                Some(d) => match d.resolve(&name) {
                    Ok(v) => v,
                    Err(e) => return Ok(vec![if e < 0 { e } else { ENOENT }]),
                },
                None => return Ok(vec![ENOENT]),
            }
        };
        let blob: Vec<u8> = addrs.iter().flat_map(NetAddr::encode).collect();
        if out != 0 && blob.len() as u64 <= cap {
            mem.write_bytes(out, &blob).ok_or(Trap::Malformed)?;
        }
        Ok(vec![blob.len() as i64])
    }

    /// The immediate child **names** of directory `path` in the flat memfs — the distinct first
    /// component of every file key under `path` (deduped, sorted for determinism). A file key exactly
    /// one level below yields its basename; a key deeper below yields the intervening subdir name
    /// (so a directory appears once even with many files under it). `"/"` lists top-level components.
    fn dir_children(&self, path: &str) -> Vec<String> {
        // Normalize to the prefix every child key starts with: `path` + "/" (just "/" for the root).
        let prefix = if path == "/" {
            "/".to_string()
        } else {
            format!("{}/", path.trim_end_matches('/'))
        };
        // Immediate child names come from two sources: file keys under `prefix`, and explicitly-created
        // empty directories under `prefix` (`mkdir`). Both contribute the first path component after
        // `prefix`.
        let mut names: Vec<String> = self
            .files
            .keys()
            .chain(self.explicit_dirs.iter())
            .filter_map(|k| k.strip_prefix(&prefix))
            .filter(|rest| !rest.is_empty())
            .map(|rest| rest.split('/').next().unwrap_or(rest).to_string())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// True if `path` names a directory in the flat memfs: the root `"/"`, an explicitly-created empty
    /// directory, or any path that is a proper prefix of some file key (i.e. has at least one child).
    /// Not a file key itself.
    fn is_dir(&self, path: &str) -> bool {
        let norm = path.trim_end_matches('/');
        norm.is_empty() || self.explicit_dirs.contains(norm) || !self.dir_children(path).is_empty()
    }

    /// `stat(path_ptr, path_len, statbuf_ptr) -> 0 | -errno`: fill the caller's `struct stat`
    /// (`{ i64 st_mode; i64 st_size; }`, 16 bytes) for a memfs path. A file key is `S_IFREG` with its
    /// byte length; a directory (a prefix of some key, or `"/"`) is `S_IFDIR` size 0; anything else is
    /// `-ENOENT`. A non-UTF-8 path is `-EINVAL`.
    fn stat(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let buf = *args.get(2).ok_or(Trap::Malformed)? as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        let (mode, size) = if let Some(f) = self.files.get(&path) {
            (S_IFREG | 0o644, f.len() as i64)
        } else if self.is_dir(&path) {
            (S_IFDIR | 0o755, 0)
        } else {
            return Ok(vec![ENOENT]);
        };
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&mode.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        mem.write_bytes(buf, &out).ok_or(Trap::Malformed)?;
        Ok(vec![0])
    }

    /// `opendir(path_ptr, path_len) -> dir | -errno`: snapshot a directory's immediate children and
    /// return a `DIR*`-analog handle for `readdir`/`closedir`. A regular file is `-ENOTDIR`; a path
    /// with no children that isn't the root is `-ENOENT`; a non-UTF-8 path is `-EINVAL`.
    fn opendir(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        if self.files.contains_key(&path) {
            return Ok(vec![ENOTDIR]);
        }
        if !self.is_dir(&path) {
            return Ok(vec![ENOENT]);
        }
        let entries = self.dir_children(&path);
        let stream = DirStream { entries, pos: 0 };
        let idx = match self.dirs.iter().position(Option::is_none) {
            Some(i) => {
                self.dirs[i] = Some(stream);
                i
            }
            None => {
                self.dirs.push(Some(stream));
                self.dirs.len() - 1
            }
        };
        Ok(vec![idx as i64])
    }

    /// `readdir(dir, name_ptr, name_cap) -> namelen | 0 | -errno`: write the next entry's name
    /// (NUL-terminated, C's `dirent.d_name` convention) into the caller's buffer and advance. Returns
    /// the name length (excluding the NUL) on success, `0` at end of stream, `-EBADF` for a stale
    /// handle, `-ERANGE` if the name + NUL won't fit `name_cap`.
    fn readdir(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let dir = *args.first().ok_or(Trap::Malformed)?;
        let name_ptr = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let cap = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as u64;
        let Some(stream) = usize::try_from(dir)
            .ok()
            .and_then(|i| self.dirs.get_mut(i)?.as_mut())
        else {
            return Ok(vec![EBADF]);
        };
        let Some(name) = stream.entries.get(stream.pos) else {
            return Ok(vec![0]); // end of stream
        };
        let mut bytes = name.clone().into_bytes();
        let namelen = bytes.len() as i64;
        bytes.push(0); // NUL
        if bytes.len() as u64 > cap {
            return Ok(vec![ERANGE]);
        }
        stream.pos += 1;
        mem.write_bytes(name_ptr, &bytes).ok_or(Trap::Malformed)?;
        Ok(vec![namelen])
    }

    /// `closedir(dir) -> 0 | -errno`: release a directory stream. A stale handle is `-EBADF`.
    fn closedir(&mut self, args: &[i64]) -> i64 {
        let dir = *args.first().unwrap_or(&-1);
        if let Some(slot @ Some(_)) = usize::try_from(dir).ok().and_then(|i| self.dirs.get_mut(i)) {
            *slot = None;
            0
        } else {
            EBADF
        }
    }

    /// `argv(i, buf, cap) -> len | -errno`: write argument `i` (NUL-terminated) into the caller's
    /// buffer and return its length (excluding the NUL). An out-of-range index is `-EINVAL`; a name
    /// that won't fit `cap` is `-ERANGE`. (`argc` is a fieldless op: `self.args.len()`.)
    fn argv(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let i = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let cap = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as u64;
        let Some(arg) = usize::try_from(i).ok().and_then(|i| self.args.get(i)) else {
            return Ok(vec![EINVAL]);
        };
        let mut bytes = arg.clone().into_bytes();
        let len = bytes.len() as i64;
        bytes.push(0);
        if bytes.len() as u64 > cap {
            return Ok(vec![ERANGE]);
        }
        mem.write_bytes(buf, &bytes).ok_or(Trap::Malformed)?;
        Ok(vec![len])
    }

    /// `exec_lookup(name_ptr, name_len) -> module_handle | -1`: resolve a command name against the PATH
    /// registry (STAGE1.md §5). Returns the granted `Module` handle (a small non-negative i32) or `-1`
    /// when the name is absent — the shell's "command not found". A non-UTF-8 name is likewise `-1` (an
    /// unfindable command, not a trap).
    fn exec_lookup(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, len).ok_or(Trap::Malformed)?;
        let Ok(name) = std::str::from_utf8(&bytes) else {
            return Ok(vec![-1]);
        };
        let h = self
            .commands
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, h, _)| *h as i64)
            .unwrap_or(-1);
        Ok(vec![h])
    }

    /// `exec_win(module_handle) -> size_log2 | -1`: the declared window of the registered command with
    /// this handle, so the shell carves its spawn to match (§14: carve == declared memory). `-1` when
    /// the handle is not a registered command.
    fn exec_win(&mut self, args: &[i64]) -> Result<Vec<i64>, Trap> {
        let handle = *args.first().ok_or(Trap::Malformed)? as i32;
        let wl = self
            .commands
            .iter()
            .find(|(_, h, _)| *h == handle)
            .map(|(_, _, w)| *w as i64)
            .unwrap_or(-1);
        Ok(vec![wl])
    }

    /// `exec_stdin(ptr, len) -> stream_handle`: push the guest bytes `[ptr, len)` — the input the shell
    /// drained for a filter command — into the read-only pipe FIFO and return its read-end handle
    /// ([`Posix::set_exec_stdin`]), which the shell re-grants to the child as `"stdin"`. Returns `-1`
    /// when no input pipe was wired (a shell built without filter support). The FIFO is drained by the
    /// child's `read`s (empty ⇒ EOF); it is empty on entry because the previous, synchronous spawn ran
    /// its child to completion.
    fn exec_stdin(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let Some(fifo) = self.exec_stdin_fifo.clone() else {
            return Ok(vec![-1]);
        };
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, len).ok_or(Trap::Malformed)?;
        let mut q = fifo.lock().unwrap_or_else(|e| e.into_inner());
        q.clear(); // defensively drop any residue from an aborted prior filter
        q.extend(bytes);
        Ok(vec![self.exec_stdin_handle as i64])
    }

    /// Allocate the lowest free fd for `entry`, extending the table if needed.
    fn alloc_fd(&mut self, entry: FdEntry) -> i64 {
        self.alloc_fd_from(entry, 0)
    }

    /// Allocate the lowest free fd `>= min` for `entry` (the `F_DUPFD`/`dup2` "at or above" contract),
    /// extending the table if needed.
    fn alloc_fd_from(&mut self, entry: FdEntry, min: usize) -> i64 {
        while self.fds.len() < min {
            self.fds.push(None);
        }
        match (min..self.fds.len()).find(|&i| self.fds[i].is_none()) {
            Some(i) => {
                self.fds[i] = Some(entry);
                i as i64
            }
            None => {
                self.fds.push(Some(entry));
                (self.fds.len() - 1) as i64
            }
        }
    }

    /// Write `data` into fd `fd`'s memfs file at its offset (extending with zeros if the offset is
    /// past the end), advancing the offset. Returns the count, or `-EBADF` for a non-file / read-only fd.
    fn file_write(&mut self, fd: usize, data: &[u8]) -> i64 {
        let (path, pos) = match self.fds.get(fd).and_then(|s| s.as_ref()) {
            Some(FdEntry::File(of)) if of.writable => (of.path.clone(), of.pos),
            _ => return EBADF,
        };
        let file = self.files.entry(path).or_default();
        let end = pos + data.len();
        if file.len() < end {
            file.resize(end, 0);
        }
        file[pos..end].copy_from_slice(data);
        if let Some(FdEntry::File(of)) = self.fds[fd].as_mut() {
            of.pos = end;
        }
        data.len() as i64
    }

    /// Read up to `len` bytes from fd `fd`'s memfs file at its offset, advancing it. `Err(-EBADF)` for
    /// a non-file fd.
    fn file_read(&mut self, fd: usize, len: usize) -> Result<Vec<u8>, i64> {
        let (path, pos) = match self.fds.get(fd).and_then(|s| s.as_ref()) {
            Some(FdEntry::File(of)) => (of.path.clone(), of.pos),
            _ => return Err(EBADF),
        };
        let file = self.files.get(&path).map(|v| v.as_slice()).unwrap_or(&[]);
        let n = len.min(file.len().saturating_sub(pos));
        let chunk = file[pos..pos + n].to_vec();
        if let Some(FdEntry::File(of)) = self.fds[fd].as_mut() {
            of.pos = pos + n;
        }
        Ok(chunk)
    }

    /// `malloc(size) -> ptr | 0`: an `ALIGN`-aligned window offset from the heap arena. First-fit
    /// **reuse** of a freed block (split if larger), else **bump** from the high-water mark. `0` (the
    /// C `NULL`) when neither can satisfy the request within `heap_end` — the anti-bomb bound.
    fn malloc(&mut self, args: &[i64]) -> i64 {
        // Round the request up to `ALIGN`; a zero-size request still yields a unique non-null cell.
        let want = ((*args.first().unwrap_or(&0)).max(0) as u64)
            .max(1)
            .div_ceil(ALIGN)
            * ALIGN;
        // First-fit over the free list: reuse the first block that fits, splitting off any remainder.
        if let Some(i) = self.free_list.iter().position(|&(_, sz)| sz >= want) {
            let (off, sz) = self.free_list.swap_remove(i);
            if sz > want {
                self.free_list.push((off + want, sz - want));
            }
            self.allocated.insert(off, want);
            return off as i64;
        }
        // Bump a fresh block from the high-water mark and record it as a live allocation.
        match self.arena_bump(want) {
            Some(ptr) => {
                self.allocated.insert(ptr, want);
                ptr as i64
            }
            None => 0, // out of heap → NULL
        }
    }

    /// Bump `n` (already `ALIGN`-aligned) bytes off the heap high-water mark, returning the aligned
    /// start offset, or `None` if it would pass `heap_end`. The low-level arena primitive `malloc` and
    /// the `getenv` string cache both grow from — it advances `heap_next` but does **not** record an
    /// `allocated` entry (the caller decides whether the block is `free`-able).
    fn arena_bump(&mut self, n: u64) -> Option<u64> {
        let ptr = (self.heap_next + (ALIGN - 1)) & !(ALIGN - 1);
        match ptr.checked_add(n) {
            Some(end) if end <= self.heap_end => {
                self.heap_next = end;
                Some(ptr)
            }
            _ => None,
        }
    }

    /// `getcwd(buf, size) -> buf | 0`: copy the current directory (NUL-terminated, C `getcwd`'s
    /// contract) into the caller's window buffer; return `buf` on success, `-ERANGE` if the path plus
    /// its NUL won't fit `size`. `size == 0` with any path is `-EINVAL` (POSIX).
    fn getcwd(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let buf = *args.first().ok_or(Trap::Malformed)? as u64;
        let size = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        if size == 0 {
            return Ok(vec![EINVAL]);
        }
        let mut bytes = self.cwd.clone().into_bytes();
        bytes.push(0); // NUL terminator
        if bytes.len() as u64 > size {
            return Ok(vec![ERANGE]);
        }
        mem.write_bytes(buf, &bytes).ok_or(Trap::Malformed)?;
        Ok(vec![buf as i64])
    }

    /// `chdir(path, len) -> 0 | -errno`: set the working directory. The memfs is flat, so any UTF-8
    /// path is accepted as-is (no existence check — a follow-up, POSIX.md §6); a non-UTF-8 path is
    /// `-EINVAL`.
    fn chdir(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        self.cwd = path;
        Ok(vec![0])
    }

    /// `getenv(name, len) -> ptr | 0`: look up an environment variable and return a **stable** window
    /// pointer to a NUL-terminated copy of its value (C `getenv`'s `char*` into libc storage), or `0`
    /// (C `NULL`) if unset. The copy is materialized in the arena once and cached (`env_ptrs`), so a
    /// repeated lookup returns the same pointer; `0` (out of heap) if the arena can't hold it. A
    /// non-UTF-8 name is treated as unset (`0`).
    fn getenv(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let nlen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, nlen).ok_or(Trap::Malformed)?;
        let Ok(name) = String::from_utf8(bytes) else {
            return Ok(vec![0]); // a name we can't represent can't be set
        };
        if let Some(&cached) = self.env_ptrs.get(&name) {
            return Ok(vec![cached as i64]);
        }
        let Some(value) = self.env.get(&name).cloned() else {
            return Ok(vec![0]); // unset → NULL
        };
        let mut vb = value.into_bytes();
        vb.push(0); // NUL terminator
        let Some(dst) = self.arena_bump(vb.len() as u64) else {
            return Ok(vec![0]); // no room → behave as if unset (best effort)
        };
        mem.write_bytes(dst, &vb).ok_or(Trap::Malformed)?;
        self.env_ptrs.insert(name, dst);
        Ok(vec![dst as i64])
    }

    /// `setenv(name, nlen, value, vlen, overwrite) -> 0 | -errno`: set (or, when `overwrite == 0` and
    /// the name already exists, leave) an environment variable. Invalidates any cached `getenv` pointer
    /// for the name so the next `getenv` materializes the new value. A non-UTF-8 name/value is `-EINVAL`.
    fn setenv(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let nptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let nlen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let vptr = *args.get(2).ok_or(Trap::Malformed)? as u64;
        let vlen = (*args.get(3).ok_or(Trap::Malformed)?).max(0) as u64;
        // The 5-arg C ABI passes `overwrite` explicitly; a 4-arg `__vm_host_call` (the svm `std` PAL's
        // `set_var`, which always overwrites) omits it, so default to overwrite when absent.
        let overwrite = *args.get(4).unwrap_or(&1);
        let nb = mem.read_bytes(nptr, nlen).ok_or(Trap::Malformed)?;
        let vb = mem.read_bytes(vptr, vlen).ok_or(Trap::Malformed)?;
        let (Ok(name), Ok(value)) = (String::from_utf8(nb), String::from_utf8(vb)) else {
            return Ok(vec![EINVAL]);
        };
        if overwrite == 0 && self.env.contains_key(&name) {
            return Ok(vec![0]); // keep the existing value
        }
        self.env_ptrs.remove(&name); // stale cached pointer no longer reflects the value
        self.env.insert(name, value);
        Ok(vec![0])
    }

    /// `getenv_r(name, nlen, buf, cap) -> nbytes | -1`: the value's byte length, written into
    /// `[buf, cap)` when it fits (no NUL — the caller owns the copy); `-1` if unset. A `cap` too small
    /// (or `buf == 0`) still returns the length, so the caller can size a buffer and retry — no arena,
    /// so it never contends with the guest's own heap. A non-UTF-8 name reads as unset.
    fn getenv_r(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let nptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let nlen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let buf = *args.get(2).ok_or(Trap::Malformed)? as u64;
        let cap = (*args.get(3).ok_or(Trap::Malformed)?).max(0) as u64;
        let nb = mem.read_bytes(nptr, nlen).ok_or(Trap::Malformed)?;
        let Ok(name) = String::from_utf8(nb) else {
            return Ok(vec![-1]);
        };
        let Some(value) = self.env.get(&name) else {
            return Ok(vec![-1]); // unset
        };
        let vb = value.as_bytes();
        if buf != 0 && (vb.len() as u64) <= cap {
            mem.write_bytes(buf, vb).ok_or(Trap::Malformed)?;
        }
        Ok(vec![vb.len() as i64])
    }

    /// `unsetenv(name, nlen) -> 0 | -EINVAL`: remove a variable (absent = success no-op).
    fn unsetenv(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let nptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let nlen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let nb = mem.read_bytes(nptr, nlen).ok_or(Trap::Malformed)?;
        let Ok(name) = String::from_utf8(nb) else {
            return Ok(vec![EINVAL]);
        };
        self.env_ptrs.remove(&name);
        self.env.remove(&name);
        Ok(vec![0])
    }

    /// `environ(index, buf, cap) -> len | -1`: the `index`-th `KEY=VALUE` (keys **sorted** for a
    /// deterministic order), written into `[buf, cap)` when it fits (size-then-fetch like `getenv_r`);
    /// `-1` once `index` is past the last variable.
    fn environ(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let index = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let cap = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as u64;
        if index < 0 {
            return Ok(vec![-1]);
        }
        let mut keys: Vec<&String> = self.env.keys().collect();
        keys.sort();
        let Some(key) = keys.get(index as usize) else {
            return Ok(vec![-1]); // past the end
        };
        let entry = format!("{key}={}", self.env[*key]).into_bytes();
        if buf != 0 && (entry.len() as u64) <= cap {
            mem.write_bytes(buf, &entry).ok_or(Trap::Malformed)?;
        }
        Ok(vec![entry.len() as i64])
    }

    /// `clock(clock_id) -> nanos`: `clock_id == 1` → monotonic (nanos since this personality started),
    /// else realtime (nanos since the Unix epoch). Returns a pinned value when [`Posix::set_clock`] set
    /// one, so a differential run is reproducible; otherwise reads the real host clock.
    fn clock(&self, args: &[i64]) -> i64 {
        if let Some(fixed) = self.clock_fixed {
            return fixed;
        }
        if *args.first().unwrap_or(&0) == 1 {
            self.clock_base.elapsed().as_nanos() as i64
        } else {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0)
        }
    }

    /// `free(ptr)`: return `ptr`'s block to the free list for reuse. `free(NULL)` and a double / bogus
    /// free are no-ops (a bogus free never corrupts the arena — the size table is host-side). No
    /// coalescing yet (POSIX.md §6).
    fn free(&mut self, args: &[i64]) {
        let ptr = *args.first().unwrap_or(&0) as u64;
        if ptr == 0 {
            return;
        }
        if let Some(size) = self.allocated.remove(&ptr) {
            self.free_list.push((ptr, size));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use svm_interp::{run_capture_reserved_with_host, Host, Value};
    use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
    use svm_text::parse_module;
    use svm_verify::verify_module;

    const HEAP_BASE: u64 = 4096;
    const HEAP_END: u64 = 64 << 10;
    const WIN: usize = 128 << 10;

    /// func 0 `(host_proc_handle) -> i64`: `malloc(2)`, store `"hi"` into the returned buffer,
    /// `write(1, ptr, 2)`, then encode `write_result * 1_000_000 + ptr`. `malloc` hands out the aligned
    /// heap base (`4096`), `write` returns `2`, so the result is `2_004096` — and stdout is `"hi"`.
    const MALLOC_WRITE: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vsz = i64.const 2\n\
  vptr = cap.call 13 2 (i64) -> (i64) vph (vsz)\n\
  vh = i32.const 104\n\
  i32.store8 vptr vh\n\
  vone = i64.const 1\n\
  vp1 = i64.add vptr vone\n\
  vi = i32.const 105\n\
  i32.store8 vp1 vi\n\
  vfd = i64.const 1\n\
  vn = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd, vptr, vsz)\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vn vk\n\
  vr = i64.add vt vptr\n\
  return vr\n\
  }\n\
}\n";

    fn run_interp(src: &str, stdin: &[u8]) -> (Result<Vec<Value>, svm_interp::Trap>, Vec<u8>) {
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
        let mut host = Host::new();
        let (h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, stdin.to_vec());
        let mut fuel = 5_000_000u64;
        let r = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut host,
        )
        .0;
        (r, posix.stdout())
    }

    fn run_jit(src: &str, stdin: &[u8]) -> (JitOutcome, Vec<u8>) {
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
        let mut host = Host::new();
        let (h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, stdin.to_vec());
        let jo = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[h as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut host as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;
        (jo, posix.stdout())
    }

    /// func 0 `(handle) -> i64`: `read(0, buf, 8)` into a `malloc`'d buffer, then `write(1, buf, n)` —
    /// a cat-style echo. Returns `n` (bytes read); stdout is whatever stdin held.
    const READ_ECHO: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  veight = i64.const 8\n\
  vbuf = cap.call 13 2 (i64) -> (i64) vph (veight)\n\
  vfd0 = i64.const 0\n\
  vn = cap.call 13 1 (i64, i64, i64) -> (i64) vph (vfd0, vbuf, veight)\n\
  vfd1 = i64.const 1\n\
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vbuf, vn)\n\
  return vn\n\
  }\n\
}\n";

    /// func 0 `(handle) -> i64`: `exit(42)` — never returns.
    const EXIT_42: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vc = i64.const 42\n\
  vx = cap.call 13 4 (i64) -> (i64) vph (vc)\n\
  vz = i64.const 0\n\
  return vz\n\
  }\n\
}\n";

    #[test]
    fn read_echo_matches_across_backends() {
        let (ir, iout) = run_interp(READ_ECHO, b"cat\n");
        let (jo, jout) = run_jit(READ_ECHO, b"cat\n");
        assert_eq!(ir, Ok(vec![Value::I64(4)]), "interp: read 4 bytes of stdin");
        assert_eq!(iout, b"cat\n", "interp: echoed stdin to stdout");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[4]),
            "jit: read count must match interp, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: echoed output must match interp");
    }

    /// func 0 `(handle) -> i64`: `a = malloc(32)`, `b = malloc(32)`, `free(a)`, `c = malloc(32)`, then
    /// return `(c - a) * 1_000_000 + (b - a)`. A working free list reuses `a`'s exact block for `c`
    /// (`c - a == 0`), and `b` sits one 32-byte block above `a` (`b - a == 32`) → `32`. (Without reuse
    /// `c` would bump fresh to `a + 64`, giving `64_000032` — so the value is non-vacuous.)
    const MALLOC_FREE_REUSE: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vsz = i64.const 32\n\
  va = cap.call 13 2 (i64) -> (i64) vph (vsz)\n\
  vb = cap.call 13 2 (i64) -> (i64) vph (vsz)\n\
  vf = cap.call 13 3 (i64) -> (i64) vph (va)\n\
  vc = cap.call 13 2 (i64) -> (i64) vph (vsz)\n\
  vcva = i64.sub vc va\n\
  vbva = i64.sub vb va\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vcva vk\n\
  vr = i64.add vt vbva\n\
  return vr\n\
  }\n\
}\n";

    #[test]
    fn free_list_reuses_a_freed_block_on_both() {
        let (ir, _iout) = run_interp(MALLOC_FREE_REUSE, b"");
        let (jo, _jout) = run_jit(MALLOC_FREE_REUSE, b"");
        // c reused a (diff 0); b is 32 bytes above a → 0*1_000_000 + 32 = 32.
        assert_eq!(
            ir,
            Ok(vec![Value::I64(32)]),
            "interp: free then malloc reuses the block"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[32]),
            "jit: allocator must match interp, got {jo:?}"
        );
    }

    #[test]
    fn exit_terminates_on_both_backends() {
        let (ir, _iout) = run_interp(EXIT_42, b"");
        let (jo, _jout) = run_jit(EXIT_42, b"");
        assert_eq!(ir, Err(svm_interp::Trap::Exit(42)), "interp: exit(42)");
        assert!(
            matches!(jo, JitOutcome::Exited(42)),
            "jit: exit(42) must terminate the domain, got {jo:?}"
        );
    }

    #[test]
    fn malloc_store_write_matches_across_backends() {
        let (ir, iout) = run_interp(MALLOC_WRITE, b"");
        let (jo, jout) = run_jit(MALLOC_WRITE, b"");
        // Interpreter reference: malloc → 4096, write → 2, so 2*1_000_000 + 4096 = 2_004096; "hi" out.
        assert_eq!(
            ir,
            Ok(vec![Value::I64(2_004_096)]),
            "interp: malloc+write result"
        );
        assert_eq!(
            iout, b"hi",
            "interp: bytes written to stdout via the personality"
        );
        // JIT parity: the HostProc dispatches through the same Host path, so identical result + output.
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2_004_096]),
            "jit: must match interp, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: stdout must match interp");
    }

    /// func 0 `(handle) -> i64`: `open("f", O_CREAT|O_RDWR)`, `write` "Hi!", `lseek` to 0, `read` it
    /// back, echo the bytes to stdout. Returns `fd * 1_000_000 + read_count`. The first file fd is `3`
    /// and 3 bytes round-trip → `3_000003`; stdout and the memfs file `"f"` are both `"Hi!"`.
    const FILE_ROUNDTRIP: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vpath = i64.const 0\n\
  vfch = i32.const 102\n\
  i32.store8 vpath vfch\n\
  vplen = i64.const 1\n\
  vflags = i64.const 66\n\
  vfd = cap.call 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
  a16 = i64.const 16\n\
  cH = i32.const 72\n\
  i32.store8 a16 cH\n\
  a17 = i64.const 17\n\
  ci = i32.const 105\n\
  i32.store8 a17 ci\n\
  a18 = i64.const 18\n\
  cbang = i32.const 33\n\
  i32.store8 a18 cbang\n\
  vwlen = i64.const 3\n\
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd, a16, vwlen)\n\
  vzero = i64.const 0\n\
  vsk = cap.call 13 7 (i64, i64, i64) -> (i64) vph (vfd, vzero, vzero)\n\
  a32 = i64.const 32\n\
  veight = i64.const 8\n\
  vr = cap.call 13 1 (i64, i64, i64) -> (i64) vph (vfd, a32, veight)\n\
  vfd1 = i64.const 1\n\
  vso = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd1, a32, vr)\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vfd vk\n\
  vres = i64.add vt vr\n\
  return vres\n\
  }\n\
}\n";

    #[test]
    fn file_open_write_seek_read_matches_across_backends() {
        // Interpreter.
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        let m = parse_module(FILE_ROUNDTRIP).expect("parse");
        verify_module(&m).expect("verify");
        let mut fuel = 5_000_000u64;
        let ir = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut ih,
        )
        .0;
        // JIT.
        let mut jh = Host::new();
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        let jo = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[jhh as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;

        // fd 3, 3 bytes read → 3_000003; the file and the echoed stdout both hold "Hi!".
        assert_eq!(
            ir,
            Ok(vec![Value::I64(3_000_003)]),
            "interp: file roundtrip"
        );
        assert_eq!(iposix.stdout(), b"Hi!", "interp: echoed the file's bytes");
        assert_eq!(
            iposix.read_file("f").as_deref(),
            Some(&b"Hi!"[..]),
            "interp: memfs file written"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[3_000_003]),
            "jit: file roundtrip must match interp, got {jo:?}"
        );
        assert_eq!(
            jposix.stdout(),
            b"Hi!",
            "jit: echoed bytes must match interp"
        );
        assert_eq!(
            jposix.read_file("f").as_deref(),
            Some(&b"Hi!"[..]),
            "jit: memfs file written"
        );
    }

    /// func 0 `(handle) -> i64`: `unlink("g")` (a preloaded file → `0`), then `open("g", O_RDONLY)` (now
    /// gone → `-ENOENT`). Returns `unlink_result * 1000 + (-open_result)` = `0*1000 + 2` = `2`.
    const UNLINK_THEN_OPEN: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vpath = i64.const 0\n\
  vg = i32.const 103\n\
  i32.store8 vpath vg\n\
  vplen = i64.const 1\n\
  vu = cap.call 13 8 (i64, i64) -> (i64) vph (vpath, vplen)\n\
  vflags = i64.const 0\n\
  vo = cap.call 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
  vzero = i64.const 0\n\
  vneg = i64.sub vzero vo\n\
  vk = i64.const 1000\n\
  vt = i64.mul vu vk\n\
  vr = i64.add vt vneg\n\
  return vr\n\
  }\n\
}\n";

    #[test]
    fn unlink_removes_then_open_is_enoent_on_both() {
        let m = parse_module(UNLINK_THEN_OPEN).expect("parse");
        verify_module(&m).expect("verify");
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        iposix.write_file("g", b"x");
        let mut fuel = 5_000_000u64;
        let ir = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut ih,
        )
        .0;
        let mut jh = Host::new();
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        jposix.write_file("g", b"x");
        let jo = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[jhh as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;
        assert_eq!(
            ir,
            Ok(vec![Value::I64(2)]),
            "interp: unlink 0, then open -ENOENT"
        );
        assert_eq!(iposix.read_file("g"), None, "interp: file is gone");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2]),
            "jit: must match interp, got {jo:?}"
        );
        assert_eq!(jposix.read_file("g"), None, "jit: file is gone");
    }

    /// func 0 `(handle) -> i64`: `open("f", O_WRONLY|O_CREAT|O_TRUNC=577)` → fd, `dup2(fd, 1)` (redirect
    /// stdout onto the file), then `write(1, "Yo", 2)`. Because fd 1 now names the file, the bytes land in
    /// the memfs, **not** in captured stdout — the shell-redirect shape (`cmd > f`). Returns the write
    /// count (2). The `577` = `O_WRONLY(1) | O_CREAT(0o100) | O_TRUNC(0o1000)`.
    const DUP2_REDIRECT: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vp = i64.const 100\n\
  vf = i32.const 102\n\
  i32.store8 vp vf\n\
  vlen = i64.const 1\n\
  vflags = i64.const 577\n\
  vfd = cap.call 13 5 (i64, i64, i64) -> (i64) vph (vp, vlen, vflags)\n\
  vone = i64.const 1\n\
  vd = cap.call 13 24 (i64, i64) -> (i64) vph (vfd, vone)\n\
  vsz = i64.const 2\n\
  vbuf = cap.call 13 2 (i64) -> (i64) vph (vsz)\n\
  vY = i32.const 89\n\
  i32.store8 vbuf vY\n\
  vbuf1 = i64.add vbuf vone\n\
  voo = i32.const 111\n\
  i32.store8 vbuf1 voo\n\
  vn = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vone, vbuf, vsz)\n\
  return vn\n\
  }\n\
}\n";

    #[test]
    fn dup2_redirects_stdout_to_a_file_on_both_backends() {
        let (ir, iout) = run_interp(DUP2_REDIRECT, b"");
        let (jo, jout) = run_jit(DUP2_REDIRECT, b"");
        assert_eq!(ir, Ok(vec![Value::I64(2)]), "interp: write count 2");
        assert_eq!(
            iout, b"",
            "interp: nothing reached stdout — fd 1 was redirected to the file"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2]),
            "jit: must match interp, got {jo:?}"
        );
        assert_eq!(jout, b"", "jit: nothing reached stdout either");
        // Both backends wrote "Yo" into the memfs file the redirected fd 1 named.
        let (mut ih, mut jh) = (Host::new(), Host::new());
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        let m = parse_module(DUP2_REDIRECT).expect("parse");
        let mut fuel = 5_000_000u64;
        let _ = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut ih,
        );
        compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[jhh as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit");
        assert_eq!(
            iposix.read_file("f").as_deref(),
            Some(&b"Yo"[..]),
            "interp: file holds redirected bytes"
        );
        assert_eq!(
            jposix.read_file("f").as_deref(),
            Some(&b"Yo"[..]),
            "jit: file holds redirected bytes"
        );
    }

    #[test]
    fn pipe_dup_fcntl_over_the_fd_table() {
        // A host-level unit for the POSIX process/fd surface (slice 1). Exercises the whole fd model:
        // pipe round-trip, dup2 redirect + shared buffer, dup lowest-free, F_DUPFD ≥ arg, ESPIPE on a
        // pipe lseek, EBADF fail-closed, and stdio fds as ordinary (closable, reusable) table entries.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut st = posix.inner.lock().unwrap();
        let mut win = vec![0u8; WIN];
        win[16..21].copy_from_slice(b"hello");
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);

        // pipe(fds@0) → 0, and stores [rfd=3, wfd=4] (the first free fds after stdio 0/1/2).
        assert_eq!(
            st.pipe(&[0], Some(&mut mem)).unwrap(),
            vec![0],
            "pipe returns 0"
        );
        let got = mem.read_bytes(0, 8).unwrap();
        let rfd = i32::from_le_bytes(got[0..4].try_into().unwrap()) as i64;
        let wfd = i32::from_le_bytes(got[4..8].try_into().unwrap()) as i64;
        assert_eq!((rfd, wfd), (3, 4), "read end below write end (Linux order)");

        // write "hello" to the write end, then drain it from the read end.
        assert_eq!(
            st.write(&[wfd, 16, 5], Some(&mut mem)).unwrap(),
            vec![5],
            "pipe write"
        );
        assert_eq!(
            st.read(&[rfd, 64, 5], Some(&mut mem)).unwrap(),
            vec![5],
            "pipe read count"
        );
        assert_eq!(
            mem.read_bytes(64, 5).unwrap(),
            b"hello",
            "pipe delivered the bytes in order"
        );
        assert_eq!(
            st.read(&[rfd, 64, 5], Some(&mut mem)).unwrap(),
            vec![0],
            "empty pipe reads 0 (EOF)"
        );
        // Reading a write end / writing a read end is -EBADF (wrong direction).
        assert_eq!(
            st.read(&[wfd, 64, 5], Some(&mut mem)).unwrap(),
            vec![EBADF],
            "read a write end is EBADF"
        );
        assert_eq!(
            st.write(&[rfd, 16, 5], Some(&mut mem)).unwrap(),
            vec![EBADF],
            "write a read end is EBADF"
        );
        // A pipe is not seekable.
        assert_eq!(
            st.lseek(&[rfd, 0, SEEK_SET]),
            ESPIPE,
            "lseek on a pipe is ESPIPE"
        );

        // dup2(wfd, 8): fd 8 becomes a second write end sharing the same buffer.
        assert_eq!(st.dup2(&[wfd, 8]), 8, "dup2 returns newfd");
        assert_eq!(
            st.write(&[8, 16, 5], Some(&mut mem)).unwrap(),
            vec![5],
            "write via the dup'd end"
        );
        assert_eq!(
            st.read(&[rfd, 64, 5], Some(&mut mem)).unwrap(),
            vec![5],
            "the original read end sees it"
        );
        assert_eq!(
            mem.read_bytes(64, 5).unwrap(),
            b"hello",
            "shared buffer, same bytes"
        );
        // dup2(fd, fd) is a no-op; dup2 of an unopened old fd is EBADF.
        assert_eq!(st.dup2(&[wfd, wfd]), wfd, "dup2(fd, fd) is a no-op");
        assert_eq!(st.dup2(&[99, 9]), EBADF, "dup2 of an unopened fd is EBADF");

        // dup(rfd) → lowest free fd (5, since 3/4 and 8 are taken).
        assert_eq!(st.dup(&[rfd]), 5, "dup takes the lowest free fd");
        // F_DUPFD ≥ 10 → 10; a bad fd is EBADF; an unknown cmd is EINVAL.
        assert_eq!(
            st.fcntl(&[rfd, F_DUPFD, 10]),
            10,
            "F_DUPFD honours the floor"
        );
        assert_eq!(
            st.fcntl(&[rfd, F_SETFD, 1]),
            0,
            "F_SETFD is an accepted no-op"
        );
        assert_eq!(
            st.fcntl(&[99, F_DUPFD, 0]),
            EBADF,
            "fcntl on a bad fd is EBADF"
        );
        assert_eq!(
            st.fcntl(&[rfd, 999, 0]),
            EINVAL,
            "unknown fcntl cmd is EINVAL"
        );

        // stdio fds are ordinary table entries: close(1) then write(1,…) is EBADF, and the next open
        // reuses fd 1 (lowest free). Restoring via dup2 makes fd 1 a stdout sink again.
        assert_eq!(st.close(&[1]), 0, "close(1) succeeds");
        assert_eq!(
            st.write(&[1, 16, 5], Some(&mut mem)).unwrap(),
            vec![EBADF],
            "write to a closed fd 1 is EBADF"
        );
        assert_eq!(st.close(&[1]), EBADF, "double close is EBADF");
    }

    #[test]
    fn spawn_waitpid_over_the_delegate() {
        // Host-level unit for the spawn/wait surface (slice 2): fail-closed without a delegate, then a
        // wired delegate sees the command/argv/inherited-stdin, its stdout is routed to fd 1, and
        // waitpid/wait reap the encoded status.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, b"data".to_vec());

        // No delegate ⇒ spawn is ENOSYS and there are no children to reap.
        {
            let mut st = posix.inner.lock().unwrap();
            let mut win = vec![0u8; WIN];
            win[0..2].copy_from_slice(b"up");
            let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
            assert_eq!(
                st.spawn(&[0, 2, 0, 0], Some(&mut mem)).unwrap(),
                vec![ENOSYS],
                "spawn with no delegate fails closed"
            );
            assert_eq!(
                st.waitpid(&[-1, 0, 0], Some(&mut mem)).unwrap(),
                vec![ECHILD],
                "no children ⇒ ECHILD"
            );
        }

        // Wire a delegate that records what it saw and uppercases the inherited stdin, exiting 7.
        let seen = Arc::new(Mutex::new(Vec::<(String, Vec<String>, Vec<u8>)>::new()));
        let rec = Arc::clone(&seen);
        posix.set_spawn(move |name, argv, stdin| {
            rec.lock()
                .unwrap()
                .push((name.to_string(), argv.to_vec(), stdin.to_vec()));
            SpawnResult {
                stdout: stdin.to_ascii_uppercase(),
                status: 7,
                ..Default::default()
            }
        });

        let (pid, status_word, stdin_after) = {
            let mut st = posix.inner.lock().unwrap();
            let mut win = vec![0u8; WIN];
            win[0..2].copy_from_slice(b"up"); // name
            win[8..13].copy_from_slice(b"up\0-n"); // argv blob: ["up", "-n"]
            let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
            // spawn("up", argv "up\0-n"): drains preloaded stdin "data", delegate → "DATA" to fd 1.
            let pid = st.spawn(&[0, 2, 8, 5], Some(&mut mem)).unwrap()[0];
            // stdin is now consumed (the child inherited and drained it).
            let after = st.read(&[0, 32, 4], Some(&mut mem)).unwrap()[0];
            // waitpid(pid, status@64, 0) reaps it.
            let r = st.waitpid(&[pid, 64, 0], Some(&mut mem)).unwrap()[0];
            assert_eq!(r, pid, "waitpid returns the reaped pid");
            let status = i32::from_le_bytes(mem.read_bytes(64, 4).unwrap().try_into().unwrap());
            // A second reap of the same pid is ECHILD (already reaped).
            assert_eq!(
                st.waitpid(&[pid, 64, 0], Some(&mut mem)).unwrap(),
                vec![ECHILD],
                "double waitpid is ECHILD"
            );
            (pid, status, after)
        };

        assert_eq!(pid, 1000, "first synthetic pid");
        assert_eq!(
            stdin_after, 0,
            "the child drained the inherited stdin (fd 0 now EOF)"
        );
        assert_eq!(
            status_word >> 8 & 0xff,
            7,
            "WEXITSTATUS = the delegate's exit code"
        );
        assert_eq!(
            posix.stdout(),
            b"DATA",
            "child stdout routed to fd 1 (no redirect ⇒ the sink)"
        );
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "the delegate ran exactly once");
        assert_eq!(seen[0].0, "up", "delegate saw the command name");
        assert_eq!(
            seen[0].1,
            vec!["up".to_string(), "-n".to_string()],
            "delegate saw argv"
        );
        assert_eq!(seen[0].2, b"data", "delegate saw the inherited stdin");
    }

    /// func 0 `(handle) -> i64`: `open("out", 577)` → fd, `dup2(fd, 1)`, `spawn("up", argv=[])`, then
    /// `waitpid(pid, &status, 0)`. With a delegate that uppercases the inherited stdin ("hi"), the child's
    /// "HI" follows the `dup2` into the file `out` (fd inheritance) rather than to stdout. Returns the
    /// reaped pid (1000). Ops: open=5, dup2=24, spawn=27, waitpid=28.
    const SPAWN_REDIRECT: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vp0 = i64.const 110\n\
  voo = i32.const 111\n\
  i32.store8 vp0 voo\n\
  vp1 = i64.const 111\n\
  vu2 = i32.const 117\n\
  i32.store8 vp1 vu2\n\
  vp2 = i64.const 112\n\
  vtt = i32.const 116\n\
  i32.store8 vp2 vtt\n\
  vn0 = i64.const 100\n\
  vuu = i32.const 117\n\
  i32.store8 vn0 vuu\n\
  vn1 = i64.const 101\n\
  vpp = i32.const 112\n\
  i32.store8 vn1 vpp\n\
  vpath = i64.const 110\n\
  vplen = i64.const 3\n\
  vflags = i64.const 577\n\
  vfd = cap.call 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
  vone = i64.const 1\n\
  vd = cap.call 13 24 (i64, i64) -> (i64) vph (vfd, vone)\n\
  vnm = i64.const 100\n\
  vnl = i64.const 2\n\
  vz = i64.const 0\n\
  vpid = cap.call 13 27 (i64, i64, i64, i64) -> (i64) vph (vnm, vnl, vz, vz)\n\
  vsb = i64.const 120\n\
  vr = cap.call 13 28 (i64, i64, i64) -> (i64) vph (vpid, vsb, vz)\n\
  return vr\n\
  }\n\
}\n";

    #[test]
    fn spawn_child_inherits_redirected_stdout_on_both_backends() {
        let m = parse_module(SPAWN_REDIRECT).expect("parse");
        verify_module(&m).expect("verify");
        let up = |_n: &str, _a: &[String], stdin: &[u8]| SpawnResult {
            stdout: stdin.to_ascii_uppercase(),
            status: 0,
            ..Default::default()
        };

        // Interp.
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, b"hi".to_vec());
        iposix.set_spawn(up);
        let mut fuel = 5_000_000u64;
        let ir = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut ih,
        )
        .0;

        // JIT.
        let mut jh = Host::new();
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, b"hi".to_vec());
        jposix.set_spawn(up);
        let jo = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[jhh as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;

        assert_eq!(
            ir,
            Ok(vec![Value::I64(1000)]),
            "interp: waitpid returns the spawned pid"
        );
        assert_eq!(
            iposix.read_file("out").as_deref(),
            Some(&b"HI"[..]),
            "interp: child stdout followed the dup2 into the file"
        );
        assert_eq!(
            iposix.stdout(),
            b"",
            "interp: nothing leaked to real stdout"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[1000]),
            "jit: must match interp, got {jo:?}"
        );
        assert_eq!(
            jposix.read_file("out").as_deref(),
            Some(&b"HI"[..]),
            "jit: child stdout followed the dup2 into the file"
        );
        assert_eq!(jposix.stdout(), b"", "jit: nothing leaked to real stdout");
    }

    #[test]
    fn signal_kill_sigcheck_l0_doorbell() {
        // Host-level unit for the L0 signal doorbell (slice 3): install dispositions, raise (guest `kill`
        // and embedder `raise_signal`), and poll — caught signals deliver their handler once, ignored and
        // default ones are dropped, lowest number first.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut st = posix.inner.lock().unwrap();

        assert_eq!(st.sigcheck(), 0, "nothing pending");
        // Install then re-install a caught handler for SIGINT(2); `signal` returns the previous.
        assert_eq!(
            st.signal(&[2, 0xABCD]),
            SIG_DFL,
            "first signal returns SIG_DFL"
        );
        assert_eq!(
            st.signal(&[2, 0x1234]),
            0xABCD,
            "signal returns the prior handler"
        );
        // Raise (guest kill), poll delivers the handler once, then it's cleared.
        assert_eq!(st.kill(&[0, 2]), 0, "kill raises");
        assert_eq!(
            st.sigcheck(),
            0x1234,
            "sigcheck delivers the caught handler"
        );
        assert_eq!(st.sigcheck(), 0, "delivered once, then cleared");

        // An ignored signal is raised but dropped, never delivered.
        assert_eq!(st.signal(&[15, SIG_IGN]), SIG_DFL, "SIGTERM set to SIG_IGN");
        st.kill(&[0, 15]);
        assert_eq!(st.sigcheck(), 0, "ignored signal is dropped");
        // A default-disposition signal (never `signal`'d) is likewise dropped in L0.
        st.kill(&[0, 3]);
        assert_eq!(st.sigcheck(), 0, "default-disposition signal dropped in L0");

        // Lowest-numbered caught signal delivers first.
        st.signal(&[10, 0xAA]);
        st.signal(&[7, 0xBB]);
        st.kill(&[0, 10]);
        st.kill(&[0, 7]);
        assert_eq!(st.sigcheck(), 0xBB, "signal 7 before signal 10");
        assert_eq!(st.sigcheck(), 0xAA, "then signal 10");
        assert_eq!(st.sigcheck(), 0, "drained");

        // Range / probe edges.
        assert_eq!(st.signal(&[0, 5]), EINVAL, "signum 0 is EINVAL");
        assert_eq!(st.signal(&[64, 5]), EINVAL, "signum 64 out of range");
        assert_eq!(st.kill(&[0, 0]), 0, "kill(pid, 0) is a liveness no-op");
        assert_eq!(
            st.kill(&[0, 99]),
            EINVAL,
            "kill with a bad signal is EINVAL"
        );
        assert_eq!(st.sigcheck(), 0, "the probe/edge calls raised nothing");

        // The embedder's door (a terminal ^C) reaches the still-installed SIGINT handler.
        drop(st);
        posix.raise_signal(2);
        let mut st = posix.inner.lock().unwrap();
        assert_eq!(
            st.sigcheck(),
            0x1234,
            "raise_signal delivers to the caught handler"
        );

        // #796: the signal mask **holds** a pending blocked signal — `sigcheck` skips it and leaves it
        // pending, until it is unblocked. (`sigprocmask` itself needs guest memory to read the sigset;
        // the compiled-C `c_posix` tests cover that path on both backends. Here we drive the mask field
        // directly to unit-test the `sigcheck` hold-vs-deliver decision.)
        st.signal(&[4, 0xCAFE]); // catch signal 4
        st.sig_mask = 1 << 4; // block it
        st.kill(&[0, 4]); // raise -> pending but blocked
        assert_eq!(
            st.sigcheck(),
            0,
            "a blocked pending signal is held, not delivered"
        );
        assert_eq!(
            st.sig_pending & (1 << 4),
            1 << 4,
            "the held signal stays pending across the poll"
        );
        st.sig_mask = 0; // unblock
        assert_eq!(st.sigcheck(), 0xCAFE, "delivered once unblocked");
        assert_eq!(st.sigcheck(), 0, "delivered exactly once");
    }

    // func 0 `(handle) -> i64`: `signal(SIGINT=2, 999)` (a caught handler), `kill(0, 2)` (raise), then
    // `sigcheck(_)` — which must return the installed handler `999`. Ops: signal=30, kill=31, sigcheck=32.
    const SIG_DOORBELL: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vsig = i64.const 2\n\
  vh = i64.const 999\n\
  vprev = cap.call 13 30 (i64, i64) -> (i64) vph (vsig, vh)\n\
  vz = i64.const 0\n\
  vk = cap.call 13 31 (i64, i64) -> (i64) vph (vz, vsig)\n\
  vc = cap.call 13 32 (i64) -> (i64) vph (vz)\n\
  return vc\n\
  }\n\
}\n";

    #[test]
    fn signal_doorbell_round_trips_on_both_backends() {
        let (ir, _) = run_interp(SIG_DOORBELL, b"");
        let (jo, _) = run_jit(SIG_DOORBELL, b"");
        assert_eq!(
            ir,
            Ok(vec![Value::I64(999)]),
            "interp: signal→kill→sigcheck delivers the installed handler"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[999]),
            "jit: doorbell must match interp, got {jo:?}"
        );
    }

    /// func 0 `(handle) -> i64`: `getenv("PATH")` (name bytes staged at offset 0 by the harness), then
    /// `write(1, ptr, 4)` echoing the first 4 bytes of the value to stdout, and return the returned
    /// pointer. With `PATH=/bin` staged host-side, `getenv` materializes `"/bin\0"` in the arena at the
    /// heap base (`4096`) and returns it; stdout is `"/bin"`.
    const GETENV_ECHO: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vp = i64.const 0\n\
  vP = i32.const 80\n\
  i32.store8 vp vP\n\
  vp1 = i64.const 1\n\
  vA = i32.const 65\n\
  i32.store8 vp1 vA\n\
  vp2 = i64.const 2\n\
  vT = i32.const 84\n\
  i32.store8 vp2 vT\n\
  vp3 = i64.const 3\n\
  vH = i32.const 72\n\
  i32.store8 vp3 vH\n\
  vnlen = i64.const 4\n\
  vptr = cap.call 13 11 (i64, i64) -> (i64) vph (vp, vnlen)\n\
  vfd1 = i64.const 1\n\
  vfour = i64.const 4\n\
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vptr, vfour)\n\
  return vptr\n\
  }\n\
}\n";

    #[test]
    fn getenv_returns_stable_ptr_and_value_on_both() {
        let m = parse_module(GETENV_ECHO).expect("parse");
        verify_module(&m).expect("verify");
        // Interpreter.
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        iposix.set_env("PATH", "/bin");
        let mut fuel = 5_000_000u64;
        let ir = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut ih,
        )
        .0;
        // JIT.
        let mut jh = Host::new();
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        jposix.set_env("PATH", "/bin");
        let jo = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[jhh as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;
        // getenv materializes "/bin\0" at the aligned heap base (4096) and returns it.
        assert_eq!(
            ir,
            Ok(vec![Value::I64(HEAP_BASE as i64)]),
            "interp: getenv returns the arena pointer"
        );
        assert_eq!(iposix.stdout(), b"/bin", "interp: echoed the env value");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[HEAP_BASE as i64]),
            "jit: getenv pointer must match interp, got {jo:?}"
        );
        assert_eq!(
            jposix.stdout(),
            b"/bin",
            "jit: echoed value must match interp"
        );
    }

    /// FORK.md PR 5 — `svm_posix::grant` is **forkable**: the fork factory (`cap`'s `make`) mints libc
    /// handlers over the *same* shared `Inner` (memfs + fd table + cwd/env), so a `fork()` twin's libc
    /// inherits the parent's open-file state — POSIX fork-shares-open-file-descriptions. This pins the
    /// factory half at the personality level: a handler minted by the factory sees a file the shared
    /// `Posix` handle wrote host-side. (The interp half — `fork_powerbox` carrying the factory across a
    /// twin — is pinned by `svm-interp`'s `fork_carries_a_forkable_host_proc_via_its_factory`.)
    ///
    /// func 0 `(handle) -> i64`: stage `"greet"` at offset 0, `open(., 5, O_RDONLY)` → fd, `read(fd,
    /// buf=32, 3)`, `write(1, buf, 3)` echoing to stdout, return the byte count.
    const READ_GREET: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vp0 = i64.const 0\n\
  vg = i32.const 103\n\
  i32.store8 vp0 vg\n\
  vp1 = i64.const 1\n\
  vr = i32.const 114\n\
  i32.store8 vp1 vr\n\
  vp2 = i64.const 2\n\
  ve = i32.const 101\n\
  i32.store8 vp2 ve\n\
  vp3 = i64.const 3\n\
  ve2 = i32.const 101\n\
  i32.store8 vp3 ve2\n\
  vp4 = i64.const 4\n\
  vt = i32.const 116\n\
  i32.store8 vp4 vt\n\
  vpath = i64.const 0\n\
  vplen = i64.const 5\n\
  vflags = i64.const 0\n\
  vfd = cap.call 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
  vbuf = i64.const 32\n\
  vcap = i64.const 3\n\
  vn = cap.call 13 1 (i64, i64, i64) -> (i64) vph (vfd, vbuf, vcap)\n\
  vfd1 = i64.const 1\n\
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vbuf, vn)\n\
  return vn\n\
  }\n\
}\n";

    #[test]
    fn grant_is_forkable_and_the_factory_shares_the_memfs() {
        let m = parse_module(READ_GREET).expect("parse");
        verify_module(&m).expect("verify");
        // `cap()` exposes the fork factory `make` — the exact factory `grant` now registers as forkable.
        // A handler minted from it (a `fork()` twin's libc) must observe the shared memfs.
        let (posix, make) = cap(HEAP_BASE, HEAP_END, Vec::new());
        posix.write_file("greet", b"hi!"); // host writes into the shared memfs
        let mut host = Host::new();
        let h = host.grant_host_proc(make()); // a factory-minted libc handler (the twin's libc)
        let mut fuel = 5_000_000u64;
        let ir = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut host,
        )
        .0;
        assert_eq!(
            ir,
            Ok(vec![Value::I64(3)]),
            "read 3 bytes through the factory-minted libc handler"
        );
        assert_eq!(
            posix.stdout(),
            b"hi!",
            "the factory-minted handler shared the parent's memfs — fork-shares-fds"
        );
    }

    /// func 0 `(handle) -> i64`: `chdir("/tmp")` (path bytes staged at offset 0), then `getcwd(buf, 8)`
    /// into a scratch window buffer, echo the result (minus its NUL) to stdout, and return
    /// `chdir_result * 1_000_000 + getcwd_ptr`. A working roundtrip: `chdir` → `0`, `getcwd` writes
    /// `"/tmp\0"` and returns the buffer offset (32) → `0*1_000_000 + 32 = 32`; stdout is `"/tmp"`.
    const CHDIR_GETCWD: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vp = i64.const 0\n\
  vsl = i32.const 47\n\
  i32.store8 vp vsl\n\
  vp1 = i64.const 1\n\
  vt = i32.const 116\n\
  i32.store8 vp1 vt\n\
  vp2 = i64.const 2\n\
  vm = i32.const 109\n\
  i32.store8 vp2 vm\n\
  vp3 = i64.const 3\n\
  vpc = i32.const 112\n\
  i32.store8 vp3 vpc\n\
  vplen = i64.const 4\n\
  vcd = cap.call 13 10 (i64, i64) -> (i64) vph (vp, vplen)\n\
  vbuf = i64.const 32\n\
  veight = i64.const 8\n\
  vgc = cap.call 13 9 (i64, i64) -> (i64) vph (vbuf, veight)\n\
  vfd1 = i64.const 1\n\
  vfour = i64.const 4\n\
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vbuf, vfour)\n\
  vk = i64.const 1000000\n\
  vtt = i64.mul vcd vk\n\
  vr = i64.add vtt vgc\n\
  return vr\n\
  }\n\
}\n";

    #[test]
    fn chdir_then_getcwd_roundtrips_on_both() {
        let (ir, iout) = run_interp(CHDIR_GETCWD, b"");
        let (jo, jout) = run_jit(CHDIR_GETCWD, b"");
        // chdir 0, getcwd returns buf (32) → 0*1_000_000 + 32 = 32; stdout "/tmp".
        assert_eq!(
            ir,
            Ok(vec![Value::I64(32)]),
            "interp: chdir then getcwd roundtrip"
        );
        assert_eq!(iout, b"/tmp", "interp: getcwd wrote the new cwd");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[32]),
            "jit: roundtrip must match interp, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: getcwd output must match interp");
    }

    #[test]
    fn setenv_updates_and_getenv_repoints_at_the_new_value() {
        // A host-level unit for the setenv/getenv cache-invalidation contract (no guest module needed):
        // getenv caches a pointer; setenv must invalidate it so the next getenv reflects the new value.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut st = posix.inner.lock().unwrap();
        // Stage the name "K" at offset 0 and value "v2" at offset 8 in a scratch window.
        let mut win = vec![0u8; WIN];
        win[0] = b'K';
        win[8] = b'v';
        win[9] = b'2';
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        // setenv("K", "v1", overwrite=1): name@0 len 1, value staged separately — reuse offset 8 with "v1".
        st.env.insert("K".to_string(), "v1".to_string());
        // getenv("K") materializes "v1\0" and caches the pointer.
        let p1 = st.getenv(&[0, 1], Some(&mut mem)).unwrap()[0];
        assert!(p1 > 0, "getenv returns a non-null arena pointer");
        // setenv("K", "v2", overwrite=1): name@0 len1, value@8 len2.
        let r = st.setenv(&[0, 1, 8, 2, 1], Some(&mut mem)).unwrap()[0];
        assert_eq!(r, 0, "setenv succeeds");
        // getenv("K") now re-materializes at a *fresh* pointer holding "v2\0".
        let p2 = st.getenv(&[0, 1], Some(&mut mem)).unwrap()[0];
        assert_ne!(p2, p1, "setenv invalidated the cached getenv pointer");
        let got = mem.read_bytes(p2 as u64, 3).unwrap();
        assert_eq!(got, b"v2\0", "getenv reflects the setenv'd value");
        // overwrite=0 on an existing name is a no-op (keeps "v2").
        let r0 = st.setenv(&[0, 1, 8, 2, 0], Some(&mut mem)).unwrap()[0];
        assert_eq!(r0, 0, "setenv(overwrite=0) on existing name returns 0");
        assert_eq!(
            st.env.get("K").map(String::as_str),
            Some("v2"),
            "overwrite=0 kept the existing value"
        );
    }

    #[test]
    fn stat_and_readdir_over_the_memfs() {
        // A host-level unit for the fs-metadata surface (stat + the opendir/readdir/closedir stream):
        // a file stats as a regular file with its size; a path with children stats as a directory and
        // enumerates its immediate children (files *and* the subdir once), sorted, ending at `0`.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        posix.write_file("/tmp/a", b"hello");
        posix.write_file("/tmp/b", b"hi");
        posix.write_file("/tmp/sub/c", b"x");

        let mut win = vec![0u8; WIN];
        win[..6].copy_from_slice(b"/tmp/a"); // path at offset 0
        win[100..104].copy_from_slice(b"/tmp"); // dir path at offset 100
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        let mut st = posix.inner.lock().unwrap();
        let rd = |mem: &svm_interp::WindowMem, off: u64| {
            i64::from_le_bytes(mem.read_bytes(off, 8).unwrap().try_into().unwrap())
        };

        // stat("/tmp/a", statbuf@200) → regular file, size 5.
        assert_eq!(st.stat(&[0, 6, 200], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(rd(&mem, 200), S_IFREG | 0o644, "st_mode: regular file");
        assert_eq!(rd(&mem, 208), 5, "st_size: the file's byte length");

        // stat("/tmp") → directory.
        assert_eq!(st.stat(&[100, 4, 200], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(rd(&mem, 200), S_IFDIR | 0o755, "st_mode: directory");

        // stat of an absent path → -ENOENT.
        win_write(&mut mem, 400, b"/nope");
        assert_eq!(st.stat(&[400, 5, 200], Some(&mut mem)).unwrap()[0], ENOENT);

        // opendir("/tmp") → children {a, b, sub} (the subdir listed once), sorted, then `0` at end.
        let dir = st.opendir(&[100, 4], Some(&mut mem)).unwrap()[0];
        assert!(dir >= 0, "opendir returns a stream handle");
        let mut got = Vec::new();
        loop {
            let n = st.readdir(&[dir, 300, 64], Some(&mut mem)).unwrap()[0];
            if n == 0 {
                break;
            }
            got.push(String::from_utf8(mem.read_bytes(300, n as u64).unwrap()).unwrap());
        }
        assert_eq!(got, vec!["a", "b", "sub"], "immediate children, sorted");
        assert_eq!(st.closedir(&[dir]), 0);
        assert_eq!(st.closedir(&[dir]), EBADF, "double closedir is -EBADF");

        // opendir of a regular file → -ENOTDIR.
        assert_eq!(st.opendir(&[0, 6], Some(&mut mem)).unwrap()[0], ENOTDIR);
    }

    #[test]
    fn mkdir_rename_rmdir_over_the_memfs() {
        // The directory-mutation surface: `mkdir` records an explicit empty dir (visible to stat and
        // readdir), `rename` moves a file or a whole subtree, `rmdir` removes only an empty dir — each
        // with the POSIX errnos `std::fs` maps to `ErrorKind`s.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        posix.write_file("/data/seed", b"x");
        posix.write_file("/data/d/inner", b"yy");

        let mut win = vec![0u8; WIN];
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        win_write(&mut mem, 0, b"/data/sub");
        win_write(&mut mem, 16, b"/data/seed");
        win_write(&mut mem, 32, b"/gone/x");
        win_write(&mut mem, 48, b"/data");
        win_write(&mut mem, 64, b"/data/renamed");
        win_write(&mut mem, 80, b"/data/d");
        win_write(&mut mem, 96, b"/data/e");
        win_write(&mut mem, 112, b"/data/e/inner");
        win_write(&mut mem, 128, b"/data/d/inner");
        let mut st = posix.inner.lock().unwrap();
        let rd = |mem: &svm_interp::WindowMem, off: u64| {
            i64::from_le_bytes(mem.read_bytes(off, 8).unwrap().try_into().unwrap())
        };

        // mkdir("/data/sub") → 0; again → -EEXIST; over a file → -EEXIST; missing parent → -ENOENT.
        assert_eq!(st.mkdir(&[0, 9, 0o777], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(st.mkdir(&[0, 9, 0o777], Some(&mut mem)).unwrap()[0], EEXIST);
        assert_eq!(
            st.mkdir(&[16, 10, 0o777], Some(&mut mem)).unwrap()[0],
            EEXIST
        );
        assert_eq!(
            st.mkdir(&[32, 7, 0o777], Some(&mut mem)).unwrap()[0],
            ENOENT
        );

        // The new empty dir stats as a directory and joins its parent's listing.
        assert_eq!(st.stat(&[0, 9, 512], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(
            rd(&mem, 512),
            S_IFDIR | 0o755,
            "mkdir'd path stats as a directory"
        );
        let dir = st.opendir(&[48, 5], Some(&mut mem)).unwrap()[0];
        let mut got = Vec::new();
        loop {
            let n = st.readdir(&[dir, 600, 64], Some(&mut mem)).unwrap()[0];
            if n == 0 {
                break;
            }
            got.push(String::from_utf8(mem.read_bytes(600, n as u64).unwrap()).unwrap());
        }
        st.closedir(&[dir]);
        assert_eq!(
            got,
            vec!["d", "seed", "sub"],
            "explicit dir joins file-derived children"
        );

        // rmdir: -ENOTEMPTY on a populated dir, -ENOTDIR on a file, success on the empty explicit dir.
        assert_eq!(st.rmdir(&[48, 5], Some(&mut mem)).unwrap()[0], ENOTEMPTY);
        assert_eq!(st.rmdir(&[16, 10], Some(&mut mem)).unwrap()[0], ENOTDIR);
        assert_eq!(st.rmdir(&[0, 9], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(st.stat(&[0, 9, 512], Some(&mut mem)).unwrap()[0], ENOENT);

        // rename a file: /data/seed → /data/renamed (contents follow, old name gone).
        assert_eq!(st.rename(&[16, 10, 64, 13], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(st.stat(&[16, 10, 512], Some(&mut mem)).unwrap()[0], ENOENT);
        assert_eq!(st.stat(&[64, 13, 512], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(rd(&mem, 520), 1, "renamed file keeps its 1-byte contents");

        // rename a directory subtree: /data/d → /data/e (its file moves with it).
        assert_eq!(st.rename(&[80, 7, 96, 7], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(
            st.stat(&[112, 13, 512], Some(&mut mem)).unwrap()[0],
            0,
            "/data/e/inner exists"
        );
        assert_eq!(
            st.stat(&[128, 13, 512], Some(&mut mem)).unwrap()[0],
            ENOENT,
            "old subtree gone"
        );

        // rename of a missing source → -ENOENT.
        assert_eq!(
            st.rename(&[32, 7, 96, 7], Some(&mut mem)).unwrap()[0],
            ENOENT
        );
    }

    #[test]
    fn memnet_bind_connect_accept_round_trip() {
        // The loopback memnet (POSIX.md §5a): bind :0 (ephemeral) → connect → accept → bytes flow
        // both ways through the libc read/write ops (the data plane needs no net surface) → EAGAIN
        // on empty → close flips the peer's reads to EOF. Plus the refusal edges: connect with no
        // listener, bind beyond loopback, bind on a held port.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut win = vec![0u8; WIN];
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        // bind blob: v4 loopback :0 at offset 0 → [4, 0,0, 127,0,0,1]
        win_write(&mut mem, 0, &[4u8, 0, 0, 127, 0, 0, 1]);
        let mut st = posix.inner.lock().unwrap();

        // bind :0 → listener fd + the actual bound addr (ephemeral port) written at 100.
        let lfd = st.net_bind(&[0, 7, 100, 32], Some(&mut mem)).unwrap()[0];
        assert!(lfd >= 0, "bind returns a listener fd");
        let bound = mem.read_bytes(100, 7).unwrap();
        assert_eq!(bound[0], 4, "bound family");
        let port = u16::from_le_bytes([bound[1], bound[2]]);
        assert!(port >= 49152, "ephemeral port assigned: {port}");

        // accept before any connect → -EAGAIN.
        assert_eq!(
            st.net_accept(&[lfd, 0, 0], Some(&mut mem)).unwrap()[0],
            EAGAIN
        );

        // connect to the bound port (blob at 200) → client fd, local addr written at 300.
        win_write(&mut mem, 200, &[4u8, bound[1], bound[2], 127, 0, 0, 1]);
        let cfd = st.net_connect(&[200, 7, 300, 32], Some(&mut mem)).unwrap()[0];
        assert!(cfd >= 0, "connect returns a socket fd");

        // accept → server fd; the written peer addr matches the client's local addr.
        let sfd = st.net_accept(&[lfd, 400, 32], Some(&mut mem)).unwrap()[0];
        assert!(sfd >= 0, "accept returns a socket fd");
        assert_eq!(
            mem.read_bytes(400, 7).unwrap(),
            mem.read_bytes(300, 7).unwrap(),
            "accept's peer addr == connect's local addr"
        );

        // client → server bytes through the ordinary write/read ops.
        win_write(&mut mem, 500, b"ping");
        assert_eq!(st.write(&[cfd, 500, 4], Some(&mut mem)).unwrap()[0], 4);
        assert_eq!(st.read(&[sfd, 600, 16], Some(&mut mem)).unwrap()[0], 4);
        assert_eq!(mem.read_bytes(600, 4).unwrap(), b"ping");
        // server → client.
        win_write(&mut mem, 500, b"pong!");
        assert_eq!(st.write(&[sfd, 500, 5], Some(&mut mem)).unwrap()[0], 5);
        assert_eq!(st.read(&[cfd, 600, 16], Some(&mut mem)).unwrap()[0], 5);
        assert_eq!(mem.read_bytes(600, 5).unwrap(), b"pong!");

        // Empty with a live peer → -EAGAIN; after the client closes → 0 (EOF).
        assert_eq!(st.read(&[sfd, 600, 16], Some(&mut mem)).unwrap()[0], EAGAIN);
        assert_eq!(st.close(&[cfd]), 0);
        assert_eq!(
            st.read(&[sfd, 600, 16], Some(&mut mem)).unwrap()[0],
            0,
            "EOF after close"
        );

        // Refusals: no listener on a random port; bind beyond loopback; bind on the held port.
        win_write(&mut mem, 700, &[4u8, 0x39, 0x30, 127, 0, 0, 1]); // port 12345
        assert_eq!(
            st.net_connect(&[700, 7, 0, 0], Some(&mut mem)).unwrap()[0],
            ECONNREFUSED
        );
        win_write(&mut mem, 700, &[4u8, 0x50, 0x00, 8, 8, 8, 8]); // 8.8.8.8:80
        assert_eq!(
            st.net_bind(&[700, 7, 0, 0], Some(&mut mem)).unwrap()[0],
            EACCES
        );
        assert_eq!(
            st.net_connect(&[700, 7, 0, 0], Some(&mut mem)).unwrap()[0],
            ECONNREFUSED,
            "non-loopback connect with no delegate fails closed"
        );
        win_write(&mut mem, 700, &[4u8, bound[1], bound[2], 127, 0, 0, 1]);
        assert_eq!(
            st.net_bind(&[700, 7, 0, 0], Some(&mut mem)).unwrap()[0],
            EADDRINUSE
        );

        // Shutdown-write on the server end → its peer... (client closed) writing now is -EPIPE.
        assert_eq!(st.net_shutdown(&[sfd, 1]), 0);
        win_write(&mut mem, 500, b"x");
        assert_eq!(st.write(&[sfd, 500, 1], Some(&mut mem)).unwrap()[0], EPIPE);

        // Closing the listener releases the port: a rebind of the same port now succeeds.
        assert_eq!(st.close(&[lfd]), 0);
        assert!(st.net_bind(&[700, 7, 0, 0], Some(&mut mem)).unwrap()[0] >= 0);
    }

    #[test]
    fn net_delegate_serves_egress_and_resolve() {
        // Beyond loopback: the embedder's NetDelegate is the authority — a scripted delegate serves
        // a canned request/response stream and a name lookup; without it (previous test) everything
        // fails closed. `localhost` resolves in-personality without any delegate.
        struct Canned;
        impl NetStream for Canned {
            fn send(&mut self, buf: &[u8]) -> i64 {
                assert_eq!(buf, b"GET /");
                buf.len() as i64
            }
            fn recv(&mut self, buf: &mut [u8]) -> i64 {
                let msg = b"HTTP/1.1 200 OK";
                buf[..msg.len()].copy_from_slice(msg);
                msg.len() as i64
            }
        }
        struct Scripted;
        impl NetDelegate for Scripted {
            fn connect(&mut self, addr: &NetAddr) -> Result<Box<dyn NetStream>, i64> {
                assert_eq!(addr.port, 80);
                assert_eq!(&addr.addr[..4], &[93, 184, 216, 34]);
                Ok(Box::new(Canned))
            }
            fn resolve(&mut self, host: &str) -> Result<Vec<NetAddr>, i64> {
                assert_eq!(host, "example.com");
                let mut addr = [0u8; 16];
                addr[..4].copy_from_slice(&[93, 184, 216, 34]);
                Ok(vec![NetAddr {
                    v6: false,
                    port: 0,
                    addr,
                }])
            }
        }
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        posix.set_net(Scripted);
        let mut win = vec![0u8; WIN];
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        let mut st = posix.inner.lock().unwrap();

        // resolve("localhost") → the loopback blob, no delegate involved.
        win_write(&mut mem, 0, b"localhost");
        let n = st.net_resolve(&[0, 9, 100, 32], Some(&mut mem)).unwrap()[0];
        assert_eq!(n, 7);
        assert_eq!(mem.read_bytes(100, 7).unwrap(), &[4u8, 0, 0, 127, 0, 0, 1]);

        // resolve("example.com") → the delegate's answer.
        win_write(&mut mem, 0, b"example.com");
        let n = st.net_resolve(&[0, 11, 100, 32], Some(&mut mem)).unwrap()[0];
        assert_eq!(n, 7);
        assert_eq!(
            mem.read_bytes(100, 7).unwrap(),
            &[4u8, 0, 0, 93, 184, 216, 34]
        );

        // connect(93.184.216.34:80) → a delegate-backed stream; send/recv round-trip the script.
        win_write(&mut mem, 200, &[4u8, 80, 0, 93, 184, 216, 34]);
        let fd = st.net_connect(&[200, 7, 0, 0], Some(&mut mem)).unwrap()[0];
        assert!(fd >= 0, "delegate connect returns a socket fd");
        win_write(&mut mem, 300, b"GET /");
        assert_eq!(st.write(&[fd, 300, 5], Some(&mut mem)).unwrap()[0], 5);
        let n = st.read(&[fd, 400, 64], Some(&mut mem)).unwrap()[0];
        assert_eq!(mem.read_bytes(400, n as u64).unwrap(), b"HTTP/1.1 200 OK");
    }

    #[test]
    fn spawn_routes_stdout_and_stderr_to_fd1_and_fd2() {
        // A `SpawnResult` carries both streams: the personality routes `stdout` to the caller's fd 1
        // and `stderr` to fd 2 (here the default stdio sinks, since the guest wired no redirect), and
        // `waitpid` reaps the wait-encoded status. This is what lets `Command::output` capture stderr.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        posix.set_spawn(|_n, _a, _stdin| SpawnResult {
            stdout: b"to-out".to_vec(),
            stderr: b"to-err".to_vec(),
            status: 3,
        });
        let mut win = vec![0u8; WIN];
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        win_write(&mut mem, 0, b"prog");
        let mut st = posix.inner.lock().unwrap();

        let pid = st.spawn(&[0, 4, 0, 0], Some(&mut mem)).unwrap()[0];
        assert!(pid >= 0, "spawn returns a pid");
        assert_eq!(st.stdout, b"to-out", "stdout routed to fd 1's sink");
        assert_eq!(st.stderr, b"to-err", "stderr routed to fd 2's sink");

        assert_eq!(st.waitpid(&[pid, 200, 0], Some(&mut mem)).unwrap()[0], pid);
        let status = i32::from_le_bytes(mem.read_bytes(200, 4).unwrap().try_into().unwrap());
        assert_eq!(
            (status >> 8) & 0xff,
            3,
            "WEXITSTATUS is the delegate's exit code"
        );
    }

    /// Write `bytes` into `mem` at `off` (test helper — `WindowMem` has no direct slice setter).
    fn win_write(mem: &mut svm_interp::WindowMem, off: u64, bytes: &[u8]) {
        mem.write_bytes(off, bytes).unwrap();
    }

    #[test]
    fn argc_argv_deliver_the_argument_vector() {
        // The host-side argument vector (the `sh -c "…"` path): `argc` reports the count, `argv(i, …)`
        // writes arg `i` NUL-terminated; an out-of-range index is -EINVAL.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        posix.set_args(&["sh", "-c", "echo hi"]);
        let mut win = vec![0u8; WIN];
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        let mut st = posix.inner.lock().unwrap();

        assert_eq!(st.args.len() as i64, 3, "argc");
        assert_eq!(st.argv(&[1, 0, 64], Some(&mut mem)).unwrap()[0], 2); // "-c" len 2
        assert_eq!(mem.read_bytes(0, 3).unwrap(), b"-c\0");
        assert_eq!(st.argv(&[2, 100, 64], Some(&mut mem)).unwrap()[0], 7); // "echo hi"
        assert_eq!(mem.read_bytes(100, 8).unwrap(), b"echo hi\0");
        assert_eq!(st.argv(&[9, 0, 64], Some(&mut mem)).unwrap()[0], EINVAL);
    }

    #[test]
    fn resolve_binds_libc_names() {
        // The §7 name → (HOST_PROC, op) map a linker uses to bind a shell's libc imports.
        assert_eq!(resolve("malloc").map(|c| c.op), Some(OP_MALLOC));
        assert_eq!(resolve("posix.write").map(|c| c.op), Some(OP_WRITE));
        assert_eq!(resolve("_exit").map(|c| c.op), Some(OP_EXIT));
        assert_eq!(resolve("open").map(|c| c.op), Some(OP_OPEN));
        assert_eq!(resolve("lseek").map(|c| c.op), Some(OP_LSEEK));
        assert!(
            resolve("dlopen").is_none(),
            "unknown libc name fails closed"
        );
    }

    /// The manifest form a chibicc/`svm-llvm` frontend emits for unresolved libc symbols: the
    /// module *imports* the libc names `malloc`/`write` (never hand-writes a `cap.call`), each
    /// call site carries a dummy `i32.const 0` handle operand (vestigial in static dispatch —
    /// IMPORTS.md §2.5), and the entry takes **no capability parameters at all** — the granted
    /// handle arrives through the slot binding ([`bind`]), never through an entry argument
    /// (→ `2_004096`, `"hi"`).
    const IMPORT_BOUND_MALLOC_WRITE: &str = "memory 17\n\
func () -> (i64) {\n\
block 0 () {\n\
  vph = i32.const 0\n\
  vsz = i64.const 2\n\
  vptr = call.sym \"malloc\" (i64) -> (i64) vph (vsz)\n\
  vh = i32.const 104\n\
  i32.store8 vptr vh\n\
  vone = i64.const 1\n\
  vp1 = i64.add vptr vone\n\
  vi = i32.const 105\n\
  i32.store8 vp1 vi\n\
  vph2 = i32.const 0\n\
  vfd = i64.const 1\n\
  vn = call.sym \"write\" (i64, i64, i64) -> (i64) vph2 (vfd, vptr, vsz)\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vn vk\n\
  vr = i64.add vt vptr\n\
  return vr\n\
  }\n\
}\n";

    #[test]
    fn bound_imports_supply_the_handle_at_resolve() {
        let m = parse_module(IMPORT_BOUND_MALLOC_WRITE).expect("parse");
        assert_eq!(
            m.imports
                .iter()
                .map(|i| i.name.as_str())
                .collect::<Vec<_>>(),
            ["malloc", "write"],
            "the module declares the libc names it imports"
        );

        // Grant FIRST (resolution needs the handle), on two identical hosts; deterministic grant
        // order gives both backends the same handle value, so one resolved module serves both.
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        let mut jh = Host::new();
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        assert_eq!(h, jhh, "identical grant order → identical handle");

        // Phase 3: no rewrite — the manifest stays and each slot binds to the personality.
        assert!(bind(&m, &mut ih, h), "posix names bind");
        assert!(bind(&m, &mut jh, jhh), "posix names bind");
        let resolved = m;
        verify_module(&resolved).expect("verify the manifest module");

        // No entry args: the program holds no capability parameters — authority came in at resolve.
        let mut fuel = 5_000_000u64;
        let ir =
            run_capture_reserved_with_host(&resolved, 0, &[], &mut fuel, &[0u8; WIN], 0, &mut ih).0;
        let jo = compile_and_run_capture_reserved_with_host(
            &resolved,
            0,
            &[],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;

        assert_eq!(
            ir,
            Ok(vec![Value::I64(2_004_096)]),
            "interp: bound-handle malloc+write"
        );
        assert_eq!(
            iposix.stdout(),
            b"hi",
            "interp: the write reached the personality"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2_004_096]),
            "jit: must match interp, got {jo:?}"
        );
        assert_eq!(jposix.stdout(), b"hi", "jit: stdout must match interp");
    }
}
