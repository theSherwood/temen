//! A **POSIX personality** as an embedder host capability (POSIX.md, §7).
//!
//! The "host provides libc as a capability; the §7 named-import mechanism carries it" story — the
//! same shape as the WASI named-import shim (the `wasi_named_imports` test), generalized from a WASI
//! subset to the POSIX/libc surface a
//! fork-less shell (BusyBox `ash` → Bash) links against. [`resolve`] binds libc symbol names to a
//! single [`temen_interp::cap_id::HOST_PROC`] capability; [`handler`] implements the ops over the guest
//! window. All libc *semantics* live **here** — outside the interp escape-TCB — reached only through a
//! granted, masked, type-checked handle (DESIGN.md §7).
//!
//! **State model (POSIX.md §3):** the bytes a libc call touches — a `malloc`'d buffer, a `write`
//! source — live in the **guest window** (native-speed access; `malloc` returns a window offset). The
//! *bookkeeping* — the allocator's cursor, captured stdout/stderr, the stdin cursor — lives host-side
//! in [`World`]/[`Proc`], never in the guest's address space, so the guest cannot corrupt it.
//!
//! Scope: `write` / `read` / `malloc` / `free` / `exit`, plus `open` / `close` / `lseek` / `unlink`
//! over an in-memory filesystem (a `path → bytes` memfs) with a host-side fd table, and
//! `getcwd` / `chdir` / `getenv` / `setenv` over a host-side cwd + environment. `malloc` is a
//! first-fit free list over a configured window-heap region. A minimal **`exec` surface** (STAGE1.md
//! §5) lets a shell on this personality launch an external command: `exec_lookup` resolves a name
//! against a `name → Module` PATH registry and `exec_stdout` hands back the `Stream` to forward — the
//! spawn itself is the shell's own `Instantiator` `call.cap` (op 13), not a personality op. Still to
//! come (POSIX.md §6): signals and `fork`/`clone`. Pure computation (`strlen`, `snprintf`, `math`, …)
//! is **guest code**, not a cap — it needs no authority (POSIX.md §1).
#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use temen_interp::{
    cap_id, ForkedProc, GuestMem, Host, HostProc, HostProcFork, SignalSource, Trap,
};
use temen_ir::ResolvedCap;

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
/// against the [`World::commands`] PATH registry (a `name → Module` handle map the embedder seeds); the
/// shell then drives `Instantiator.instantiate_module_named` (op 13) + `join` on the returned handle.
/// `exec_stdout() -> stream_handle` returns the `Stream` the shell should re-grant to the child under
/// the name `"stdout"` so the command's `write(1, …)` reaches the shell's sink. `exec_stdin(ptr, len)
/// -> stream_handle` is the input counterpart (a **filter** command): the personality pushes the
/// `[ptr, len)` bytes — the shell's current input, e.g. a `< file` redirect it drained — into a
/// read-only pipe FIFO and returns the read end for the shell to re-grant as `"stdin"`, so the
/// command's `read(0, …)` drains them (then `0` = EOF). Neither op *is* the spawn — op 13 is a guest
/// `call.cap` (the compiled shell holds the `Instantiator`); the personality only supplies the registry
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

/// **Parallel-safe spawn + capture** (#848). Identical to [`OP_SPAWN`] except the child's stdio is
/// bound **per-child, atomically inside the one op** rather than by a `dup2(pipe,1)` → spawn →
/// `dup2(saved,1)` bracket around it. The #825 audit found that bracket races on the **parallel**
/// driver: the personality lock is released between the three ops, so two vCPUs running
/// `Command::output()` concurrently corrupt the shared fd-1/fd-2 binding. Carrying the redirect *in*
/// the spawn op keeps it atomic (the lock is held for the whole spawn) and per-child (the shared
/// fd-0/1/2 table is never mutated), so concurrent captures cannot collide.
///
/// `spawn2(req_ptr) -> pid | -errno`: `req_ptr` points at a 44-byte little-endian request struct
/// carrying the command target and three fd-actions (the guest FFI has only four payload slots, which
/// the `spawn` target already fills, so `spawn2`'s extra arguments travel by struct — the
/// `posix_spawn(…, file_actions, …)` shape):
///
/// ```text
///   +0  name_ptr : u64      +24 argv_len : u64
///   +8  name_len : u64      +32 stdin_fd : i32
///   +16 argv_ptr : u64      +36 stdout_fd: i32
///                           +40 stderr_fd: i32
/// ```
///
/// `stdin_fd` is drained as the child's input; the child's captured stdout is routed to `stdout_fd`
/// and its stderr to `stderr_fd`. A `-1` fd inherits the caller's current fd 0 / 1 / 2 binding (the
/// [`OP_SPAWN`] default), so a request with all three fds `-1` is exactly `spawn`.
pub const OP_SPAWN2: u32 = 43;

/// `getpid() -> pid` (#863 slice 2): this process's own pid — `1` for the root, the scheduler
/// `TaskId` (= the parent's `fork()` return) for a fork twin, a personality-allocated pid for a
/// re-grant clone (slice 3). One pid space with `kill(pid)`/`waitpid(pid)`: the value is
/// meaningful to both.
pub const OP_GETPID: u32 = 44;

/// **Job control — the process-group surface** (#798 slice 1). `setpgid(pid, pgid) -> 0 | -errno`:
/// move process `pid` (`0` = self) into group `pgid` (`0` = `pid`'s own id — become a leader), by
/// the process table; the target must be the caller or a live table entry. POSIX's exec/session
/// restrictions are not modeled (one world ≈ one session, #798).
pub const OP_SETPGID: u32 = 45;
/// `getpgid(pid) -> pgid | -errno`: process `pid`'s (`0` = self) group id, from the table.
pub const OP_GETPGID: u32 = 46;
/// `tcgetpgrp(fd) -> pgid | -errno`: the **foreground process group** of the terminal `fd` refers
/// to. Until the TTY layer (#797) the personality's captured stdio is the one proto-terminal: any
/// stdio-sentinel fd names it, anything else is `-ENOTTY`. Foreground starts as the root's group
/// (`1`), so nothing changes for a guest that never calls `tcsetpgrp`.
pub const OP_TCGETPGRP: u32 = 47;
/// `tcsetpgrp(fd, pgid) -> 0 | -errno`: make `pgid` the terminal's foreground process group
/// (`-EINVAL` non-positive, `-EPERM` if no live process is in it, `-ENOTTY` off-terminal). A
/// background process's terminal I/O then rings its `SIGTTOU`/`SIGTTIN` doorbell (see
/// [`OP_WRITE`]/[`OP_READ`]) — the L0 approximation of stop-on-background-I/O until #798 slice 2
/// brings real stop/continue.
pub const OP_TCSETPGRP: u32 = 48;

/// `isatty(fd) -> 1 | 0` (#800): is `fd` the terminal? Until the TTY layer (#797) the captured
/// stdio is the one proto-terminal (the same convention as `tcgetpgrp`/`tcsetpgrp`): the stdio
/// fds 0/1/2 answer `1`, everything else `0`. Bash probes `isatty(0)`/`isatty(2)` to decide
/// interactive mode — this is the op that decision rides on.
pub const OP_ISATTY: u32 = 49;
/// `getppid() -> pid` (#800): the parent's pid — the pid of the process whose `fork` (or whose
/// re-grant, for a spawned clone) minted this one; `0` only for the root (init-like, no parent).
/// Bash exports it as `$PPID`.
pub const OP_GETPPID: u32 = 50;
/// `fork() -> pid | 0 | -errno` (#799): **return-twice fork through the personality** — no offer
/// topology, no manager. The op fires the core's caller-request door ([`ParkEvent::ForkSelf`])
/// and returns a `-ENOSYS` placeholder: a parkable route discards it, runs the core's fork engine
/// (private window copy, duplicated powerbox — which runs this personality's own fork factory, so
/// the twin is table-registered with its pid and full doors at birth), and re-admits the parent
/// with the twin's pid and the twin with `0`. A refused fork re-admits `-EAGAIN` (retryable — the
/// `while ((pid = fork()) < 0)` idiom); a route or tier without the door keeps `-ENOSYS` (fork
/// unavailable — an error a shell surfaces, never an infinite retry).
pub const OP_FORK: u32 = 51;

/// #972 slice 1 — **adopt two core pipe-end handles into the fd table** (`pipe_adopt(read_h,
/// write_h, fds_ptr) -> 0 | -errno`). The unification's mint path: the guest mints its own
/// counted pipe with the `CAP_SELF_PIPE` self-op (`__vm_pipe` — synchronous, its own powerbox,
/// no host authority), then this op records the two handles as [`FdEntry::CorePipe`] entries and
/// writes `[read_fd, write_fd]` at `fds_ptr`. The personality gains **no authority**: it stores
/// handle *numbers* as bookkeeping — it cannot exercise them (INVARIANTS.md #3; the handles never
/// leave the guest's powerbox), and a garbage handle simply fails the guest's own later cap-call.
pub const OP_PIPE_ADOPT: u32 = 52;

/// #801 slice A — **`exec_resolve(path_ptr, path_len) -> module_handle | -errno`**: the
/// execve-of-a-file resolver. A path registered through [`Posix::register_executable`] returns its
/// pre-granted `Module` handle (the exec_lookup discipline: minting happened at registration on
/// the embedder's side, so this dispatch is pure bookkeeping); a memfs file that is NOT a
/// registered executable is `-EACCES` (a real file without the exec bit); anything else is
/// `-ENOENT`. The guest libc's `execve` follows a hit with the `CAP_SELF_EXEC` self-op
/// (`__vm_exec_module`) — the core image-replace — so the personality adds no exec mechanism,
/// only the path→module policy (INVARIANTS #4).
pub const OP_EXEC_RESOLVE: u32 = 53;

/// #797 — the **termios control surface** as named ops (the libc `ioctl()`/`tcgetattr` shims
/// multiplex onto these, keeping the vtable's signature checking meaningful). The personality's
/// `struct termios` is a deliberately minimal 32 bytes: `{ i64 c_lflag; u8 c_cc[8] (packed i64);
/// i64 c_vmin; i64 c_vtime }` — the lflag bits (Linux values: `ISIG 0o1`, `ICANON 0o2`,
/// `ECHO 0o10`) and cc slots (`[VINTR, VQUIT, VERASE, VKILL, VEOF, VSUSP, 0, 0]`) the discipline
/// honors; unknown bits round-trip uninterpreted (POSIX.md: extended by consumer demand).
pub const OP_TCGETATTR: u32 = 54;
/// #797 — `tcsetattr(fd, attr_ptr)` (the `when` is a shim concern; the personality applies
/// immediately — TCSANOW; a shell's drain semantics are a follow-up).
pub const OP_TCSETATTR: u32 = 55;
/// #797 — `tcgetwinsize(fd, ws_ptr)`: `{ i32 row; i32 col }` (8 bytes), the `TIOCGWINSZ` shape.
pub const OP_TCGETWINSIZE: u32 = 56;

/// #972 slice 1 — the **handle-carrying tag** returned by `read`/`write`/`close` on a
/// [`FdEntry::CorePipe`] fd: `PX_TAG_BASE - handle`. A **personality ↔ shim private convention**,
/// never interpreted by the core (INVARIANTS.md #5/#11 discipline): every tag is `<= PX_TAG_BASE`,
/// every real errno is `> -4096`, so the ranges can never alias; no top-byte semantics — plain
/// negative i64 values on the existing sign-tested `count | -errno` returns. The shim decodes
/// `handle = PX_TAG_BASE - tag` and follows with the core cap-call (`__vm_read`/`__vm_write`/
/// `__vm_close`) — the blocking/EINTR/EOF/`-EPIPE` path. A caller that is not our shim sees a
/// large negative "error" and fails closed: no bytes moved, no fd state changed.
pub const PX_TAG_BASE: i64 = -(1 << 20);

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
/// `clock(clock_id) -> nanos` (POSIX.md — the `Clock` surface; `std::time` reaches it via the temen
/// `std` PAL). `clock_id == 1` is monotonic (nanos since this personality started), anything else is
/// realtime (nanos since the Unix epoch). An embedder/test can pin the value with [`Posix::set_clock`]
/// for determinism (the differential harness wants a reproducible clock).
pub const OP_CLOCK: u32 = 33;
/// `getenv_r(name, nlen, buf, cap) -> nbytes | -1` — a **buffer-writing** `getenv` for callers that
/// own the destination (Rust's `std::env::var` copies into an `OsString`), unlike op 11 which
/// materializes a stable `char*` in the personality arena. Returns the value's byte length (writing it
/// into `[buf, cap)` when it fits — the two-call size-then-fetch shape of `getcwd`/`readdir`), or `-1`
/// if unset. Because it writes into guest-owned memory it needs **no arena**, so it never contends
/// with the guest's own heap — the reason the temen `std` PAL uses it.
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

/// Negative errnos this personality returns — the one shared table ([`temen_ir::errno`], Linux
/// values, so a guest's `<errno.h>` agrees).
use temen_ir::errno::*;

/// #798 — the job-control signal numbers (Linux values, like every signum here).
const SIGCHLD: i32 = 17; // a child stopped, continued, or exited (#802 rung 3 — now generated)
const SIGCONT: i32 = 18; // continue a stopped process (also deliverable if caught)
const SIGSTOP: i32 = 19; // unconditional stop (uncatchable)
const SIGTSTP: i32 = 20; // the terminal ^Z — stop unless caught/ignored
const SIGTTIN: i32 = 21; // background read from the terminal
const SIGTTOU: i32 = 22; // background write to the terminal

/// #796 default actions — `SIGKILL` (uncatchable, unmaskable terminate).
const SIGKILL: i32 = 9;
/// #796 — `sigaction` `sa_flags` bit: restart an interrupted blocking call instead of `-EINTR`
/// (Linux value, so a guest's `<signal.h>` agrees).
const SA_RESTART: i64 = 0x10000000;
/// #796 default actions — the signals whose `SIG_DFL` action is **ignore** (Linux: CHLD/URG/WINCH).
/// Everything else outside the job-control set above defaults to **terminate**.
/// #802 — map the `/dev/fd/N` and `/dev/std{in,out,err}` pseudo-paths to the fd they alias, or
/// `None` for an ordinary path. `open` treats a hit as a `dup` of that fd (bash's `HAVE_DEV_FD`
/// process-substitution path). Only these exact forms — a `/dev/fd/` with trailing junk is not a
/// valid fd and falls through to the memfs (→ `ENOENT`), matching Linux.
fn dev_fd_target(path: &str) -> Option<i64> {
    match path {
        "/dev/stdin" => Some(0),
        "/dev/stdout" => Some(1),
        "/dev/stderr" => Some(2),
        _ => path
            .strip_prefix("/dev/fd/")
            .and_then(|n| n.parse::<i64>().ok()),
    }
}

fn default_ignored(sig: i32) -> bool {
    matches!(
        sig,
        17 /* SIGCHLD */ | 23 /* SIGURG */ | 28 /* SIGWINCH */
    )
}

/// #798 slice 2 — `waitpid` option bits (Linux values).
const WNOHANG: i64 = 1; // never block (#799 — the poll everyone implicitly had before)
const WUNTRACED: i64 = 2; // also report a freshly-stopped child
const WCONTINUED: i64 = 8; // also report a freshly-continued child

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
const EACCES: i64 = -13;
const ENOTTY: i64 = -25; // #797: not a terminal // #801: a memfs file without the executable registration
/// #802 rung-3 tail — the **restart sentinel** (Linux's kernel-internal `ERESTART`): a terminal
/// read that just STOPPED its caller (SIGTTIN) returns this instead of minting the input tag, and
/// the guest read wrappers re-issue the op in a loop — the stop lands at the re-issued dispatch's
/// safepoint poll, so the reader is benched BEFORE it can touch the input pipe. Mirrors the
/// kernel's stop-then-transparently-restart contract; without it a `bg`-continued background
/// reader raced its own stop and STOLE the next typed line from the foreground shell.
const ERESTART: i64 = -85;
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
/// `pipe` adds a `PipeRead`/`PipeWrite` pair sharing one [`PipeBuf`]. Every entry is a reference to a
/// **shared open-file description** (#863): `dup`/`dup2` clone the reference, so dups share the offset
/// (POSIX), and [`Proc::fork`]'s entry-wise table copy shares descriptions across processes the same
/// way. Descriptions lock innermost (World → Proc → description; see [`World`]).
enum FdEntry {
    Stdin,
    Stdout,
    Stderr,
    File(Arc<Mutex<OpenFile>>),
    PipeRead(PipeBuf),
    PipeWrite(PipeBuf),
    /// A connected memnet socket end (loopback; POSIX.md §5a).
    NetSock(MemSock),
    /// A delegate-backed connected stream (beyond loopback; the embedder owns the I/O).
    NetStream(Arc<Mutex<Box<dyn NetStream>>>),
    /// A memnet listener (`bind`+listen folded); `accept` pops its pending queue.
    NetListener(MemListener),
    /// #972 slice 1 — a **core pipe-end handle** adopted into the table ([`OP_PIPE_ADOPT`]): the fd
    /// is a view over a counted core pipe end living in the guest's own powerbox. `read`/`write`/
    /// `close` on it return a [`PX_TAG_BASE`] tag redirecting the shim to the core cap-call path
    /// (blocking, EINTR, true EOF, `-EPIPE`). The `Arc` token counts **intra-process dups only**:
    /// `dup`/`dup2`/`F_DUPFD` share it, and the last `close` (strong count 1) tags "release the
    /// handle" so the shim's `__vm_close` fires the powerbox count decrement (EOF/EPIPE wakes).
    /// [`Proc::fork`] re-splits tokens per process — the twin's powerbox is a *duplicate* with its
    /// own end-counts, so its release decisions must be its own (sharing the token across the fork
    /// would let the parent's last close swallow the twin's release, or vice versa).
    CorePipe(Arc<CorePipeToken>),
}

/// The shared token behind every intra-process dup of one adopted core pipe-end fd
/// ([`FdEntry::CorePipe`]): the handle number, plus the `Arc` itself as the dup count. Atomic
/// (#972): the exec-remap hook re-points it when the image-replace renumbers the carried end.
struct CorePipeToken {
    handle: std::sync::atomic::AtomicI32,
}

impl CorePipeToken {
    fn new(h: i32) -> Self {
        CorePipeToken {
            handle: std::sync::atomic::AtomicI32::new(h),
        }
    }
    fn get(&self) -> i32 {
        self.handle.load(Ordering::Relaxed)
    }
}

impl FdEntry {
    /// Clone this entry for `dup`/`dup2` (and [`Proc::fork`]'s table copy): every non-sentinel arm
    /// shares its description via `Arc` clone — a dup'd or fork-inherited file fd shares the offset,
    /// pipe ends share the buffer, per POSIX.
    fn dup_clone(&self) -> FdEntry {
        match self {
            FdEntry::Stdin => FdEntry::Stdin,
            FdEntry::Stdout => FdEntry::Stdout,
            FdEntry::Stderr => FdEntry::Stderr,
            FdEntry::File(of) => FdEntry::File(Arc::clone(of)),
            FdEntry::PipeRead(p) => FdEntry::PipeRead(Arc::clone(p)),
            FdEntry::PipeWrite(p) => FdEntry::PipeWrite(Arc::clone(p)),
            // Socket ends and listeners share their connection state (`Arc` clones throughout) —
            // a dup is another reference to the same socket, and the write-liveness token's
            // last-drop EOF accounts for every dup.
            FdEntry::NetSock(s) => FdEntry::NetSock(s.clone()),
            FdEntry::NetStream(d) => FdEntry::NetStream(Arc::clone(d)),
            FdEntry::NetListener(l) => FdEntry::NetListener(l.clone()),
            // Intra-process dup: share the token, so only the last close releases the handle.
            // Fork does NOT use this arm for CorePipe — see [`Proc::fork`]'s re-split.
            FdEntry::CorePipe(t) => FdEntry::CorePipe(Arc::clone(t)),
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

/// #797 — the controlling terminal's state (see [`World::terminal`]). Termios subset: the
/// lflag bits and cc slots the discipline honors; everything else round-trips.
struct Terminal {
    /// `c_lflag` (Linux bit values; default `ISIG | ICANON | ECHO`).
    lflag: i64,
    /// Packed cc: `[VINTR, VQUIT, VERASE, VKILL, VEOF, VSUSP, 0, 0]`.
    cc: [u8; 8],
    vmin: i64,
    vtime: i64,
    /// Winsize (`{row, col}`), reported by op 56; updated by [`Posix::set_winsize`].
    rows: i32,
    cols: i32,
    /// The input pipe: the guest-facing read handle (`read(0)`'s tag target), the backing the
    /// feed deposits into, the held writer count (`^D` on an empty line drops it to 0 — true
    /// EOF, one-shot, the rung-1 decision), and the pipe id for the wake door.
    input_handle: i32,
    input_backing: Arc<Mutex<VecDeque<u8>>>,
    input_writers: Arc<std::sync::atomic::AtomicUsize>,
    input_pipe: u32,
    /// The canonical-mode line under construction (flushed to the backing on `\n`/`VEOF`).
    canon_buf: Vec<u8>,
}

impl Terminal {
    fn default_termios() -> (i64, [u8; 8]) {
        // ISIG | ICANON | ECHO; VINTR=^C VQUIT=^\ VERASE=DEL VKILL=^U VEOF=^D VSUSP=^Z.
        (0o1 | 0o2 | 0o10, [3, 28, 127, 21, 4, 26, 0, 0])
    }
}

/// #863 — state **all processes of one personality share**: the "kernel side" of the split. One
/// per world, behind one `Arc<Mutex<World>>` every process's handler holds. POSIX draws the line:
/// the filesystem, the network, the registered commands, the embedder delegates, and the captured
/// stdio *descriptions* are world-shared; everything a `fork()` copies lives in [`Proc`].
///
/// **Lock order: `World` before `Proc`, always.** Every op that needs both takes the world lock
/// first ([`handler`] takes both at dispatch top); a proc-only path ([`SignalDoor`],
/// [`Posix::raise_signal`]) may lock its `Proc` alone but must never then take the world lock.
/// Open-file descriptions ([`OpenFile`]) nest innermost: World → Proc → description.
struct World {
    stdout: Vec<u8>,
    /// When set, fd-1 writes go **here** instead of [`World::stdout`], and [`Posix::stdout`] reads it
    /// back. This unifies the shell's own output with a spawned child's: the embedder points it at the
    /// `Host`'s shared stdout sink (`Host::shared_stdout`), the same buffer a re-granted `Stream` writes
    /// to, so the shell's `write(1, …)` and the command's `write(1, …)` interleave in one stream
    /// (STAGE1.md §5). `None` keeps the self-contained captured-`Vec` behaviour (unchanged for every
    /// existing embedder).
    stdout_sink: Option<Arc<Mutex<Vec<u8>>>>,
    stderr: Vec<u8>,
    /// Preloaded standard input; `read(0, …)` drains it from `stdin_pos`. World-shared: the stdin
    /// **description** (bytes + offset) is one open file description all processes' fd-0 sentinels
    /// point at — POSIX fork shares the offset.
    stdin: Vec<u8>,
    stdin_pos: usize,
    /// The **in-memory filesystem**: path → contents. A memfs keeps the personality self-contained and
    /// deterministic (the playground has no disk); a native embedder routing to a real `fs` cap is a
    /// follow-up. Shared file bytes; per-fd offsets live in each fd's [`OpenFile`] description.
    files: HashMap<String, Vec<u8>>,
    /// Explicitly-created **empty** directories (`mkdir`). The memfs otherwise infers directories as
    /// prefixes of file keys, which can't represent a dir with no files under it; this set carries those.
    /// A path is a directory if it is the root, appears here, or is a proper prefix of some file key.
    explicit_dirs: HashSet<String>,
    /// Loopback memnet listeners: bound port → the pending-connection queue its `accept` pops and a
    /// loopback `connect` pushes into. (The queue is shared with the listener's `FdEntry`; this index
    /// exists so `connect` can find it by port and `bind` can detect `-EADDRINUSE`.)
    net_listeners: HashMap<u16, Arc<Mutex<VecDeque<MemSock>>>>,
    /// The next ephemeral port a `bind :0` (or a connect's synthesized source) is assigned from.
    net_next_port: u16,
    /// The embedder's network delegate ([`Posix::set_net`]) — authority beyond loopback. `None` ⇒
    /// non-loopback fails closed.
    net_delegate: Option<Box<dyn NetDelegate>>,
    /// The monotonic-clock base — `clock(1)` reports nanos elapsed since this. Captured at creation.
    /// wasm32 (the browser playground) has **no std time source** — `Instant::now`/`SystemTime::now`
    /// PANIC on wasm32-unknown-unknown, and this field's initializer took the whole personality
    /// `grant` down before the guest ran one op (found by the #1080 bash card, the first wasm test
    /// to actually exercise the post-#800 grant). There the personality serves a deterministic
    /// strictly-increasing tick instead — the core `Binding::Clock` shape.
    #[cfg(not(target_arch = "wasm32"))]
    clock_base: std::time::Instant,
    /// The wasm32 twin of `clock_base`: the next nanos `clock()` serves, bumped per read so time
    /// always advances (a guest's `$SECONDS`/timestamps need monotonicity, not wall truth).
    #[cfg(target_arch = "wasm32")]
    clock_tick: std::sync::atomic::AtomicI64,
    /// A pinned clock value (`Some(nanos)`) for determinism: when set, `clock(_)` returns it verbatim
    /// so a differential run is reproducible. `None` reads the real host clock.
    clock_fixed: Option<i64>,
    /// The **PATH registry** (STAGE1.md §5): command name → `(granted Module handle, declared window
    /// size_log2)`. `exec_lookup` returns the handle; `exec_win` the size_log2 (so the shell carves each
    /// spawn to the command's own window). The embedder seeds it with [`Posix::register_command`] after
    /// granting each command `Module`. A plain `Vec` (a shell's PATH is short); first match wins.
    commands: Vec<(String, i32, u8)>,
    /// #801 — paths registered as **executables** ([`Posix::register_executable`]): the exec-bit
    /// set `stat` consults (mode `0o755` vs a plain file's `0o644`) and the `exec_resolve` gate
    /// (a memfs file outside this set is `-EACCES`).
    executables: HashSet<String>,
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
    /// #863 slice 2 — the **process table**: `pid → entry`, ONE pid space for every process this
    /// world knows. A **fork twin**'s pid is its scheduler `TaskId` (the value the parent's `fork()`
    /// returned), registered at mint by [`fork_factory`]; a **spawn-delegate child** (already run to
    /// completion) sits as a [`ProcEntry::Zombie`] holding its wait-encoded status until `waitpid`
    /// reaps it. `kill(pid, sig)` and `waitpid(pid)` are lookups here; the root is pid `1`.
    procs: HashMap<i32, ProcEntry>,
    /// The next pid `spawn` hands out for a delegate child. Starts at `1000` and skips occupied
    /// pids (fork twins occupy their `TaskId`s in the same table — one space, no collisions).
    next_pid: i32,
    /// #798 — the proto-terminal's **foreground process group** (`tcsetpgrp`/`tcgetpgrp`; the
    /// captured stdio stands in for the pty until #797). Init `1` — the root's group — so every
    /// pre-job-control guest is foreground and nothing rings. A background process's terminal
    /// I/O rings its `SIGTTOU`/`SIGTTIN` doorbell.
    fg_pgid: i32,
    /// #797 — the **controlling terminal**, wired by [`Posix::enable_terminal`] (`None` = the
    /// preloaded-stdin world, unchanged). Input is a held-writer core pipe the embedder feeds
    /// ([`Posix::feed_terminal`]) with the **line discipline running at feed time**; `read(0)`
    /// tag-redirects to `input_handle` (the #972 park/EINTR/EOF path). Output echo goes to the
    /// stdout sink. The discipline is pure personality policy (INVARIANTS 4/12).
    terminal: Option<Terminal>,
}

/// #863 slice 2 — a [`World::procs`] process-table entry.
enum ProcEntry {
    /// A live process: its [`Proc`], so `kill(pid, sig)` can set **its** pending bit (and wake
    /// **its** run). The root (pid `1`) and every fork twin (pid = scheduler `TaskId`) live here.
    Live(Arc<Mutex<Proc>>),
    /// A spawn-delegate child that already ran to completion: its wait-encoded exit status, held
    /// until `waitpid`/`wait` reaps it — a zombie.
    Zombie {
        /// The wait-encoded status `waitpid` serves.
        status: i32,
        /// #799 — the pgid at exit, retained so `waitpid(-pgid)` can group-reap a zombie whose
        /// `Proc` (the live pgid's home) is already gone.
        pgid: i32,
        /// #1080 pipeline rung — the ppid at exit, retained so `waitpid(-1)`/`waitpid(-pgid)` reap
        /// only the CALLER's OWN children (POSIX), never a sibling's zombie. Without it a bash
        /// pipeline stage's `waitpid(-1)` steals the reap the shell is blocked waiting for, wedging
        /// `echo | cat`. The root's pid is `1`, so a root child carries `ppid == 1`.
        ppid: i32,
    },
}

/// #863 — **per-process** state: what POSIX `fork()` copies (and what `exec` will one day reset).
/// One per process domain; a `fork()` twin gets [`Proc::fork`]'s clone — fd *table* copied
/// entry-wise over shared descriptions, cwd/env/args copied, allocator copied (paired with the
/// twin's private window copy), signal dispositions/mask copied with **pending cleared**. See
/// [`World`] for the shared side and the lock order.
struct Proc {
    /// This process's own pid (#863 slice 2): `1` for the root (init-like), the scheduler `TaskId`
    /// for a fork twin (stamped by [`fork_factory`] at mint — the same value the parent's `fork()`
    /// returned), or a personality-allocated pid for a re-grant clone (slice 3 — a spawned child is
    /// wired before any `TaskId` exists, so it draws from the spawn allocator instead; every
    /// process is table-addressable). `getpid` reports it, and `kill(own pid)` short-circuits to
    /// the self path on it.
    pid: i32,
    /// #800 `getppid` — the minting process's pid, recorded in [`Proc::fork`] (twins and
    /// re-grant clones alike); `0` only for the root.
    ppid: i32,
    /// #798 — this process's **process group** (`setpgid`/`getpgid`): the unit `kill(-pgid)`
    /// sweeps and `tcsetpgrp` foregrounds. The root leads group `1`; a `fork` twin **inherits**
    /// its parent's (POSIX — unlike the core's `Twin.pgid`, which defaults to own-id and feeds
    /// only the core reap's `-pgid` filter; the personality's is authoritative for personality
    /// ops, and the two unify when blocking `waitpid` lands, #799).
    pgid: i32,
    /// #801 — the heap **re-base** a committed exec applies (POSIX: `brk` is per-image state,
    /// re-based at exec — the caller's heap placement was chosen for the caller's image, and the
    /// command's data/stack may extend right across it). Stashed by a successful `exec_resolve`
    /// as the command's own convention — heap in the top quarter of ITS registered window, the
    /// same split embedders use for roots — and consumed by the exec-remap hook, which the core
    /// fires only on a COMMITTED image-replace, so a refused exec leaves the live heap untouched.
    pending_exec_heap: Option<(u64, u64)>,
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
    /// The host-side fd **table** (indexed by fd) — per-process, POSIX: a fork copies the table, a
    /// child's `close`/`dup2` never disturbs its parent's numbering. Entries point at **shared**
    /// descriptions (an [`OpenFile`] `Arc`, a pipe buffer, a socket), so offsets and liveness are
    /// shared where POSIX shares them. Seeded with the three stdio sentinels at `0`/`1`/`2`
    /// (`FdEntry::Stdin`/`Stdout`/`Stderr`); `open`/`pipe`/`dup` allocate the lowest free slot.
    fds: Vec<Option<FdEntry>>,
    /// Open directory streams (`opendir`/`readdir`/`closedir`), indexed by the `DIR*`-analog handle
    /// `opendir` returns. Each holds the immediate child names snapshotted at `opendir` time and a
    /// read cursor. Separate from [`Proc::fds`] (a directory stream is not a file fd here).
    dirs: Vec<Option<DirStream>>,
    /// The program's argument vector (`args[0]` is the program name), delivered **host-side** — the
    /// symmetric analogue of the environment: `argc`/`argv` read it, the embedder sets it. This is how
    /// a personality program gets `sh -c "…"` without the window args buffer (POSIX.md §5); a guest
    /// crt that wants a standard `main(int, char**)` builds `argv[]` from these ops.
    args: Vec<String>,
    /// The current working directory `getcwd` reports and `chdir` updates — per-process (a subshell's
    /// `cd` must not move its parent). A plain string — the memfs is flat (paths are used as-given),
    /// so `cwd` is not validated against it; path normalization/resolution is a follow-up (POSIX.md §6).
    cwd: String,
    /// The environment: `name → value` — per-process (POSIX fork copies it; a child's `setenv` is
    /// invisible to the parent). `getenv`/`setenv` read and update it; host-side, out of the guest's
    /// reach, like the rest of the bookkeeping (POSIX.md §3).
    env: HashMap<String, String>,
    /// Cache of `getenv` results already materialized into the window: `name → ptr`. C's `getenv`
    /// returns a stable `char*` into libc-owned storage, so a repeated `getenv("X")` must return the
    /// **same** pointer; we allocate a NUL-terminated copy in the arena once and reuse it. `setenv`
    /// invalidates the entry so the next `getenv` re-materializes the new value. Per-process (the
    /// pointers land in this process's own window arena).
    env_ptrs: HashMap<String, u64>,
    /// **Pending signals** — the L0 doorbell. Bit `s` set ⇒ signal `s` has been raised (`kill`/
    /// [`Posix::raise_signal`]) and not yet polled. Signals are `1..=63` (one `u64`). Per-process, and
    /// **cleared** in a fork twin (POSIX: the child starts with no pending signals).
    sig_pending: u64,
    /// **Signal dispositions**: `signum → handler` (`SIG_DFL`/`SIG_IGN`/a guest handler pointer), set by
    /// `signal`/`sigaction`. Absent ⇒ `SIG_DFL`. `sigcheck` consults this to deliver caught signals and
    /// drop the rest. Copied into a fork twin (POSIX: dispositions are inherited).
    sig_handler: HashMap<i32, i64>,
    /// **Signal mask** — the blocked set (`sigprocmask`, #796). Bit `s` set ⇒ signal `s` is blocked: a
    /// pending blocked signal is held (not delivered by `sigcheck`) until unblocked. `SIGKILL`/`SIGSTOP`
    /// are never blockable, so those bits stay clear. Copied into a fork twin.
    sig_mask: u64,
    /// **Per-signal `sigaction` extras** (`sa_mask` / `sa_flags`), kept for round-trip fidelity (`oldact`).
    /// The poll model does not yet auto-block `sa_mask` while a handler runs, nor act on `SA_RESTART` —
    /// those land with the L2 async-delivery / L1 EINTR slices of #796.
    sig_action_mask: HashMap<i32, u64>,
    sig_action_flags: HashMap<i32, i64>,
    /// #796 L2 — the guest's registered **signal-handler stack** base (`sigaltstack`), the data-SP an async
    /// handler runs on. `0` ⇒ no stack ⇒ async delivery is off (poll-only). Distinct from the guest's normal
    /// stack because the interp can't compute where the interrupted frame's stack ends. Copied into a
    /// fork twin (POSIX: the signal stack settings are inherited; the twin's window is a private copy,
    /// so the base points at the twin's own copy of the stack).
    sig_stack_base: u64,
    /// #796 L2 — the **armed** flag shared with the interp ([`Host::sig_armed`]): set when a caught,
    /// unmasked signal may be deliverable, so the interp's per-op poll knows to ask [`SignalSource`]. The
    /// same `Arc` is handed to the `Host` at grant time. A fork twin gets a **fresh** flag (its door is
    /// its own).
    sig_armed: Arc<AtomicBool>,
    /// #799 L1 (embedder `^C`) — the interp's **scheduler-wake** closure, installed at run start via
    /// [`SignalSource::set_wake`] and cleared to a no-op at teardown. [`Posix::raise_signal`] invokes it
    /// after raising a **deliverable** signal, so an embedder `^C` interrupts a parked blocking syscall
    /// even when every fiber is parked (no per-op safepoint to notice the arm). `None` until a run installs
    /// it (and between runs); a fork twin / spawned child gets its own installed at mint/admission
    /// (#863 slice 3 — the core hands every door a domain-scoped weak wake, so reachability is
    /// independent of nesting depth and fork history; weak ⇒ a post-run fire is a no-op).
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
    /// #797 — the run's **pipe-reader wake** door ([`SignalSource::set_pipe_wake`]): how a
    /// host-side `feed_terminal` wakes a reader parked on the terminal's input pipe.
    pipe_wake: Option<Arc<dyn Fn(u32) + Send + Sync>>,
    /// #798 slice 2 — the core's **stop/continue closure** for this process's domain
    /// ([`SignalSource::set_stop`], installed beside the wake): `f(true)` parks the domain at its
    /// next per-op poll, `f(false)` resumes it. `None` (a driver without the mechanism — the
    /// bytecode driver, a bare unit test) degrades to bookkeeping-only stops: the table records
    /// the stop, `WUNTRACED` reports it, but the process keeps running (the L0 posture).
    stop: Option<Arc<dyn Fn(bool) + Send + Sync>>,
    /// #798 slice 2 — the signal this process is currently **stopped** by (`None` = running).
    /// Set by the delivery gate on a default-action stop signal; cleared by `SIGCONT`.
    stopped_sig: Option<i32>,
    /// #798 slice 2 — a stop not yet reported through `waitpid(WUNTRACED)` (report-once).
    stop_fresh: bool,
    /// #798 slice 2 — a continue not yet reported through `waitpid(WCONTINUED)` (report-once).
    cont_fresh: bool,
    /// #1171 — a **one-shot** "a child of mine transitioned (stop/continue)" edge, set on this
    /// (parent) process when a child stops or continues (beside the run-wake). The cooperative
    /// driver's all-parked sweep reads-and-clears it via [`SignalSource::reap_pending`] to re-admit
    /// this process's blocked `waitpid(WUNTRACED/WCONTINUED)` even when it has no async SIGCHLD
    /// delivery armed (bash: no sigaltstack) — the parked reap park is not woken by a poll-only
    /// SIGCHLD, and a foreground job stopped on a parked read never enters the core stop-park that
    /// would drain the reap waiters. One-shot so the re-admitted `waitpid` (report-once via
    /// `stop_fresh`/`cont_fresh`) cannot re-trigger itself into a spin.
    reap_wake: bool,
    /// #796 default actions — the core's **terminate closure** for this process's domain
    /// ([`SignalSource::set_kill`], installed beside the wake/stop pair): firing it kills the
    /// domain at its next per-op poll. `None` (a driver without the mechanism) degrades to
    /// bookkeeping-only termination: `term_sig` is recorded (so `waitpid` reports it if the
    /// process exits by other means) but the process keeps running — the L0 posture, same
    /// degradation as `stop`.
    kill: Option<Arc<dyn Fn() + Send + Sync>>,
    /// #796 default actions — the signal that terminated this process (`SIG_DFL` terminate
    /// action). Read by the exit hook at retire time: set ⇒ the zombie's wait status is the
    /// `WIFSIGNALED` shape (`sig` in the low 7 bits) instead of the exit-code encode.
    term_sig: Option<i32>,
    /// #796 block-during-handler — the pre-delivery masks of the **live async handler frames**,
    /// innermost last: [`SignalDoor::take_deliverable`] pushes the current mask and blocks the
    /// delivered signal + its `sa_mask` for the handler's duration (POSIX);
    /// [`SignalDoor::handler_returned`] pops and restores. LIFO per process — with `thread.spawn`
    /// siblings sharing one process, interleaved cross-fiber returns restore in push order (a
    /// documented approximation: POSIX masks are per-thread, ours is per-process).
    handler_mask_stack: Vec<u64>,
    /// #796 `SA_RESTART` — whether the most recent caught delivery's action carried `SA_RESTART`,
    /// answered to the core through [`SignalDoor::syscall_restart`] when an interrupt reaches a
    /// parked blocking call. Last delivery wins (the documented approximation of POSIX's
    /// per-interrupting-handler rule for near-simultaneous mixed-flag deliveries).
    restart_ok: bool,
    /// #799 — the core's **park-request closure** ([`SignalSource::set_park_request`], installed
    /// beside the wake/stop/kill doors): `waitpid` fires it with `ParkEvent::TaskExit(child)` to
    /// ask that the calling vCPU be benched until the child exits, then returns the `-ECHILD`
    /// placeholder (which doubles as the degraded poll answer on any route that cannot park).
    /// `None` (a driver without the door — the bytecode engine, the JIT, a bare unit `Ctx`)
    /// keeps today's poll everywhere — same degradation family as `stop`/`kill`.
    park_req: Option<Arc<dyn Fn(temen_interp::ParkEvent) + Send + Sync>>,
    /// #799 — this process's pid **is a core scheduler `TaskId`** (a fork twin registered by the
    /// factory with the core-minted pid) — exactly the processes whose exit the core's
    /// twin-completion wake covers, so exactly the ones a blocking `waitpid` may bench on.
    /// `false` for the root and for spawn/re-grant clones (personality-allocated pids).
    core_task: bool,
    /// #797 interactive rung 2 — this process's OWN handle on the terminal input pipe (the
    /// world's [`Terminal::input_handle`] is the ROOT namespace's; handle values are per-powerbox,
    /// so every process needs its own). Seeded at [`Posix::enable_terminal`] (root), cloned by
    /// [`Proc::fork`] (the twin's cloned table keeps the value valid), and re-pointed by
    /// [`exec_remap_hook`] when an exec carries the end into a fresh powerbox — exactly the
    /// [`CorePipeToken`] discipline the fd table's adopted pipe ends follow. `None` = fall back
    /// to the world handle (a pre-terminal or spawn-delegate process).
    term_in: Option<CorePipeToken>,
}

/// One dispatch's view over the two personality lock domains — the shared [`World`] and the calling
/// process's [`Proc`], both locked at dispatch top ([`handler`]) in the canonical order (world,
/// then proc; see [`World`]). The op bodies are methods here; pure-`Proc` signal helpers
/// ([`Proc::arm_signals`], [`Proc::deliverable_now`]) live on [`Proc`] so proc-only doors reach
/// them without the world lock.
struct Ctx<'a> {
    w: &'a mut World,
    p: &'a mut Proc,
    /// #863 slice 2 — scheduler wakes to fire **after** this dispatch's locks are released.
    /// `kill` pushes one per target the signal is deliverable to (a `-pgid` group kill may have
    /// several, #798): the wake closures take the scheduler lock, and the fork factory takes the
    /// world lock *under* the scheduler lock, so firing while still holding the world lock would
    /// deadlock (scheduler → world vs world → scheduler). [`handler`] fires them once the guards
    /// are dropped.
    wake_after: Vec<Arc<dyn Fn() + Send + Sync>>,
}

/// A handle to a granted POSIX personality's state — read the captured output after a run, stage
/// the memfs/env, raise embedder signals. Cheap to clone (shares the `Arc`s). #863: holds the
/// shared [`World`] plus the **root process's** [`Proc`] — the proc-side setters (`set_env`,
/// `set_args`, `cwd`, `raise_signal`) address the root process, matching the pre-split embedder
/// semantics (per-child addressing is #863 slice 2's process table).
#[derive(Clone)]
pub struct Posix {
    world: Arc<Mutex<World>>,
    root: Arc<Mutex<Proc>>,
}

impl Posix {
    /// Bytes the guest `write`-to-fd-1'd — from the shared sink when one is set ([`Posix::set_stdout_sink`]),
    /// else the personality's own captured buffer.
    pub fn stdout(&self) -> Vec<u8> {
        let st = self.world.lock().unwrap_or_else(|e| e.into_inner());
        match &st.stdout_sink {
            Some(sink) => sink.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            None => st.stdout.clone(),
        }
    }
    /// Bytes the guest `write`-to-fd-2'd.
    pub fn stderr(&self) -> Vec<u8> {
        self.world
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stderr
            .clone()
    }

    /// Seed (or overwrite) a memfs file — how an embedder/test stages the filesystem a guest `open`s.
    pub fn write_file(&self, path: &str, bytes: &[u8]) {
        self.world
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .files
            .insert(path.to_string(), bytes.to_vec());
    }

    /// Read a memfs file back — how an embedder/test inspects what the guest wrote.
    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        self.world
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .files
            .get(path)
            .cloned()
    }

    /// Seed (or overwrite) an environment variable — how an embedder/test stages the environment a
    /// guest `getenv`s. Invalidates any cached `getenv` pointer for the name.
    pub fn set_env(&self, name: &str, value: &str) {
        let mut st = self.root.lock().unwrap_or_else(|e| e.into_inner());
        st.env_ptrs.remove(name);
        st.env.insert(name.to_string(), value.to_string());
    }

    /// Pin the clock to a fixed `nanos` value (all `clock(_)` calls return it) — how a test makes
    /// `std::time` deterministic. Passing a value makes a differential run reproducible.
    pub fn set_clock(&self, nanos: i64) {
        self.world
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clock_fixed = Some(nanos);
    }

    /// The current working directory — how an embedder/test observes a guest `chdir`.
    pub fn cwd(&self) -> String {
        self.root
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cwd
            .clone()
    }

    /// Set the program's argument vector (`args[0]` is conventionally the program name) — how an
    /// embedder hands a personality program its `argv` (e.g. `["sh", "-c", "echo hi"]`), read back by
    /// the guest through the `argc`/`argv` ops.
    pub fn set_args(&self, args: &[&str]) {
        self.root.lock().unwrap_or_else(|e| e.into_inner()).args =
            args.iter().map(|s| s.to_string()).collect();
    }

    /// Route fd-1 (`stdout`) writes to a **shared sink** instead of the personality's own buffer, so a
    /// spawned child's re-granted `Stream` output and the shell's own output land in one stream
    /// (STAGE1.md §5). Pass the `Host`'s shared stdout (`Host::shared_stdout()`). [`Posix::stdout`] then
    /// reads this sink back.
    /// #797 — wire the **controlling terminal** (opt-in; without it the preloaded-stdin world is
    /// unchanged): mints the held-writer input pipe on `host` and installs the terminal state
    /// with default termios (`ISIG | ICANON | ECHO`) and an 80×24 winsize. `read(0)` then
    /// tag-redirects to the input pipe (park on empty, the #972 path); the embedder delivers
    /// keystrokes with [`Self::feed_terminal`].
    pub fn enable_terminal(&self, host: &mut Host) {
        let (h, backing, writers, pipe) = host.grant_terminal_input();
        let (lflag, cc) = Terminal::default_termios();
        let mut w = self.world.lock().unwrap_or_else(|e| e.into_inner());
        w.terminal = Some(Terminal {
            lflag,
            cc,
            vmin: 1,
            vtime: 0,
            rows: 24,
            cols: 80,
            input_handle: h,
            input_backing: backing,
            input_writers: writers,
            input_pipe: pipe,
            canon_buf: Vec::new(),
        });
        // #797 interactive rung 2 — the ROOT's own terminal token (`h` was minted in its
        // powerbox); forks clone it, execs re-point it (see [`Proc::term_in`]).
        drop(w);
        self.root.lock().unwrap_or_else(|e| e.into_inner()).term_in = Some(CorePipeToken::new(h));
    }

    /// #797 — deliver keystrokes to the terminal: the **line discipline runs here**, at feed
    /// time, host-side. Raw mode (`!ICANON`) deposits bytes immediately; canonical mode buffers
    /// with `VERASE`/`VKILL` editing and flushes completed lines (newline, or a mid-line `VEOF`);
    /// `VEOF` on an empty line drops the held writer count — parked readers wake to true EOF
    /// (one-shot, the rung-1 decision). `ECHO` mirrors input to the stdout sink (`VERASE` echoes
    /// backspace-space-backspace). `ISIG` chars never enter the stream: they fire the #798 group
    /// machinery at the foreground pgid (`VINTR`→SIGINT, `VQUIT`→SIGQUIT, `VSUSP`→SIGTSTP) via
    /// [`Self::kill_pid`] per member, after every lock is released. A parked reader is woken
    /// through the pipe-wake door — data arrival, never a signal interrupt.
    pub fn feed_terminal(&self, bytes: &[u8]) {
        let mut kills: Vec<(i32, i32)> = Vec::new();
        type PipeWake = Arc<dyn Fn(u32) + Send + Sync>;
        let mut wake: Option<(PipeWake, u32)> = None;
        {
            let mut w = self.world.lock().unwrap_or_else(|e| e.into_inner());
            let fg = w.fg_pgid;
            let Some(mut term) = w.terminal.take() else {
                return;
            };
            let isig = term.lflag & 0o1 != 0;
            let icanon = term.lflag & 0o2 != 0;
            let echo = term.lflag & 0o10 != 0;
            let mut out: Vec<u8> = Vec::new(); // echo bytes
            let mut deposit: Vec<u8> = Vec::new(); // bytes for the backing
            let mut eof = false;
            for &b in bytes {
                if isig && (b == term.cc[0] || b == term.cc[1] || b == term.cc[5]) {
                    let sig = if b == term.cc[0] {
                        2 // SIGINT
                    } else if b == term.cc[1] {
                        3 // SIGQUIT
                    } else {
                        20 // SIGTSTP
                    };
                    // Sweep the foreground group after the locks drop.
                    for (pid, e) in w.procs.iter() {
                        if let ProcEntry::Live(p) = e {
                            if p.lock().unwrap_or_else(|er| er.into_inner()).pgid == fg {
                                kills.push((*pid, sig));
                            }
                        }
                    }
                    continue;
                }
                if !icanon {
                    deposit.push(b);
                    if echo {
                        out.push(b);
                    }
                    continue;
                }
                if b == term.cc[2] {
                    // VERASE
                    if term.canon_buf.pop().is_some() && echo {
                        out.extend_from_slice(b" ");
                    }
                } else if b == term.cc[3] {
                    // VKILL
                    for _ in 0..term.canon_buf.len() {
                        if echo {
                            out.extend_from_slice(b" ");
                        }
                    }
                    term.canon_buf.clear();
                } else if b == term.cc[4] {
                    // VEOF: empty line = one-shot EOF; mid-line = flush without newline.
                    if term.canon_buf.is_empty() {
                        eof = true;
                    } else {
                        deposit.append(&mut term.canon_buf);
                    }
                } else if b == b'\n' {
                    term.canon_buf.push(b);
                    deposit.append(&mut term.canon_buf);
                    if echo {
                        out.push(b'\n');
                    }
                } else {
                    term.canon_buf.push(b);
                    if echo {
                        out.push(b);
                    }
                }
            }
            let deposit_was_empty = deposit.is_empty();
            if !deposit.is_empty() {
                term.input_backing
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .extend(deposit);
            }
            if eof {
                term.input_writers
                    .store(0, std::sync::atomic::Ordering::SeqCst);
            }
            if !out.is_empty() {
                match &w.stdout_sink {
                    Some(sk) => sk
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .extend_from_slice(&out),
                    None => w.stdout.extend_from_slice(&out),
                }
            }
            let pipe = term.input_pipe;
            // The data wake fires only when something changed for a reader — a deposit or the
            // EOF close. A pure-signal feed (`^C` alone) must NOT wake the park: the reader
            // would re-admit, find nothing, and re-park — racing the kill's EINTR sweep into a
            // window where nothing is parked. Signal delivery does its own waking.
            let should_wake = !deposit_was_empty || eof;
            w.terminal = Some(term);
            drop(w);
            if should_wake {
                let p = self.root.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(pw) = p.pipe_wake.as_ref() {
                    wake = Some((Arc::clone(pw), pipe));
                }
            }
        }
        if let Some((pw, pipe)) = wake {
            pw(pipe); // data arrival (or the EOF close): re-admit parked readers
        }
        for (pid, sig) in kills {
            self.kill_pid(pid, sig);
        }
    }

    /// #797 — update the terminal's winsize and fire **SIGWINCH** (28) at the foreground group.
    pub fn set_winsize(&self, rows: i32, cols: i32) {
        let mut kills: Vec<i32> = Vec::new();
        {
            let mut w = self.world.lock().unwrap_or_else(|e| e.into_inner());
            let fg = w.fg_pgid;
            if let Some(t) = w.terminal.as_mut() {
                t.rows = rows;
                t.cols = cols;
                for (pid, e) in w.procs.iter() {
                    if let ProcEntry::Live(p) = e {
                        if p.lock().unwrap_or_else(|er| er.into_inner()).pgid == fg {
                            kills.push(*pid);
                        }
                    }
                }
            }
        }
        for pid in kills {
            self.kill_pid(pid, 28);
        }
    }

    pub fn set_stdout_sink(&self, sink: Arc<Mutex<Vec<u8>>>) {
        self.world
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stdout_sink = Some(sink);
    }

    /// Register a command in the PATH registry (STAGE1.md §5): `name → module_handle`, where
    /// `module_handle` is the handle a granted command `Module` has in the shell's cap table. The
    /// shell's `exec_lookup(name)` returns it (or `-1` when absent). A later registration of the same
    /// name shadows the earlier (last wins), matching a `PATH` re-export.
    pub fn register_command(&self, name: &str, module_handle: i32, win_log2: u8) {
        let mut st = self.world.lock().unwrap_or_else(|e| e.into_inner());
        st.commands.retain(|(n, _, _)| n != name);
        st.commands
            .push((name.to_string(), module_handle, win_log2));
    }

    /// #801 — register a **filesystem executable**: `path → module_handle` in the same registry
    /// [`Self::register_command`] feeds (a path is just a name with `/`), plus the userland
    /// presentation — a memfs marker file at `path` (so `stat`/`glob`/`open` see a real file) and
    /// the exec bit (`stat` reports mode `0o755`; `exec_resolve` refuses unregistered files with
    /// `-EACCES`). The `Module` was granted on the shell's `Host` by the embedder — authority
    /// arrived down the grant graph (invariant 3); this records the path view over it.
    pub fn register_executable(&self, path: &str, module_handle: i32, win_log2: u8) {
        self.register_command(path, module_handle, win_log2);
        let mut st = self.world.lock().unwrap_or_else(|e| e.into_inner());
        st.executables.insert(path.to_string());
        st.files
            .entry(path.to_string())
            .or_insert_with(|| b"\x7fSVM".to_vec());
    }

    /// Set the `Stream` handle the shell re-grants to a spawned child as its `"stdout"` — what
    /// `exec_stdout()` returns. Grant the `Stream` on the same `Host`, routed to the shared sink
    /// (`set_stdout_sink`), so the child's output joins the shell's (STAGE1.md §5).
    pub fn set_exec_stdout(&self, handle: i32) {
        self.world
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .exec_stdout_handle = handle;
    }

    /// Set the read-only-pipe `Stream` + FIFO the shell re-grants to a **filter** command as `"stdin"`
    /// — the input twin of [`Self::set_exec_stdout`]. `handle` and `fifo` come from one
    /// [`Host::grant_input_pipe`] call on the same `Host`; `exec_stdin(ptr, len)` pushes the guest bytes
    /// into `fifo` and returns `handle`, so the child's `read(0, …)` drains them (then EOF).
    pub fn set_exec_stdin(&self, handle: i32, fifo: Arc<Mutex<VecDeque<u8>>>) {
        let mut st = self.world.lock().unwrap_or_else(|e| e.into_inner());
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
        self.world
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .spawn_fn = Some(Box::new(f));
    }

    /// Wire the **network delegate** — the authority for anything beyond the loopback memnet
    /// (POSIX.md §5a; the [`Self::set_spawn`] analog). Until one is set, a non-loopback `connect`
    /// is `-ECONNREFUSED` and a non-`localhost` `resolve` is `-ENOENT` — fail closed. The delegate
    /// is where policy lives: a real socket, a scripted table, an allowlisting proxy.
    pub fn set_net(&self, delegate: impl NetDelegate + 'static) {
        self.world
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
            let mut st = self.root.lock().unwrap_or_else(|e| e.into_inner());
            // The delivery gate (#798 slice 2): an embedder ^Z (SIGTSTP) stops the root, an
            // embedder SIGCONT resumes it — same policy as every in-world raise.
            st.deliver_signal(signum)
        };
        if let Some(w) = wake {
            w();
        }
    }

    /// #863 slice 3 — the embedder's **pid-targeted** signal: [`Posix::raise_signal`]'s semantics
    /// (pending + arm + wake-if-deliverable, disposition-gated by the TARGET's own state) routed
    /// through the process table instead of pinned to the root. This is how a terminal `^C` reaches
    /// the *right* process when the world has several — e.g. a spawned shell (a slice-3 re-grant
    /// clone, pid from the personality allocator) or a fork twin (pid = its `TaskId`). The wake it
    /// fires is the target's own domain-scoped run-wake, so only that process's blocked syscalls
    /// return `-EINTR` (INVARIANTS.md #12). Returns `0`, `-ESRCH` for an unknown pid, `-EINVAL`
    /// for a bad signal. Locks world → target proc (the canonical order), releases both, then wakes.
    pub fn kill_pid(&self, pid: i32, signum: i32) -> i64 {
        if !(1..=63).contains(&signum) {
            return EINVAL;
        }
        let (wake, transitioned) = {
            let w = self.world.lock().unwrap_or_else(|e| e.into_inner());
            match w.procs.get(&pid) {
                Some(ProcEntry::Live(t)) => {
                    let mut tp = t.lock().unwrap_or_else(|e| e.into_inner());
                    // The delivery gate (#798 slice 2): pending, stop, or continue by the
                    // target's dispositions; fired below, after the locks drop.
                    let was_stopped = tp.stopped_sig.is_some();
                    let f = tp.deliver_signal(signum);
                    // #802 rung 3 — stopped or continued: SIGCHLD its parent (below, unlocked).
                    let trans = (was_stopped != tp.stopped_sig.is_some()).then_some(tp.ppid);
                    (f, trans)
                }
                Some(ProcEntry::Zombie { .. }) => (None, None), // unreaped; takes no signal
                None => return ESRCH,
            }
        };
        if let Some(f) = wake {
            f();
        }
        if let Some(ppid) = transitioned {
            self.chld_to(ppid);
        }
        0
    }

    /// #802 rung 3 — deliver `SIGCHLD` to table process `ppid` (a child of theirs transitioned),
    /// firing the returned wake after the locks drop. The embedder-path twin of
    /// [`Ctx::notify_parent_chld`]; an absent, dead, or handler-less parent is a no-op.
    fn chld_to(&self, ppid: i32) {
        // #1171 — grab BOTH the (maybe-`None`) async SIGCHLD delivery wake AND the parent's domain
        // run-wake. A child's stop/continue transition must wake a parent blocked in
        // `waitpid(WUNTRACED/WCONTINUED)` even when the parent has no async SIGCHLD delivery armed
        // (bash: no sigaltstack), so its blocked `waitpid` re-runs and reports the fresh stop.
        let (wake, run_wake) = {
            let w = self.world.lock().unwrap_or_else(|e| e.into_inner());
            match w.procs.get(&ppid) {
                Some(ProcEntry::Live(t)) => {
                    let mut tp = t.lock().unwrap_or_else(|e| e.into_inner());
                    tp.reap_wake = true; // #1171 — one-shot reap re-check edge for the coop sweep
                    (tp.deliver_signal(SIGCHLD), tp.wake.clone())
                }
                _ => (None, None),
            }
        };
        if let Some(f) = wake {
            f();
        }
        if let Some(w) = run_wake {
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
        "pipe_adopt" => OP_PIPE_ADOPT,
        "exec_resolve" => OP_EXEC_RESOLVE,
        "tcgetattr" => OP_TCGETATTR,
        "tcsetattr" => OP_TCSETATTR,
        "tcgetwinsize" => OP_TCGETWINSIZE,
        "dup2" => OP_DUP2,
        "dup" => OP_DUP,
        "fcntl" => OP_FCNTL,
        "spawn" | "posix_spawn" | "posix_spawnp" => OP_SPAWN,
        "spawn2" => OP_SPAWN2,
        "getpid" => OP_GETPID,
        "setpgid" => OP_SETPGID,
        "getpgid" => OP_GETPGID,
        "tcgetpgrp" => OP_TCGETPGRP,
        "tcsetpgrp" => OP_TCSETPGRP,
        "isatty" => OP_ISATTY,
        "getppid" => OP_GETPPID,
        "fork" => OP_FORK,
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
pub fn bind(m: &temen_ir::Module, host: &mut Host, handle: i32) -> bool {
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
    m: &temen_ir::Module,
    host: &mut Host,
    handle: i32,
    fork: Option<(u32, i32)>,
) -> bool {
    let mut binds = Vec::with_capacity(m.imports.len());
    for i in &m.imports {
        let bare = i.name.strip_prefix("posix.").unwrap_or(&i.name);
        if bare == "fork" {
            let Some((tid, fh)) = fork else { return false };
            binds.push(temen_interp::BoundImport::required(tid, 0, fh));
        } else {
            let Some(c) = resolve(&i.name) else {
                return false;
            };
            binds.push(temen_interp::BoundImport::required(c.type_id, c.op, handle));
        }
    }
    host.set_import_bindings(binds);
    true
}

/// Grant a POSIX personality on `host`, returning the `HOST_PROC` handle and a [`Posix`] handle to its
/// captured state. `heap_base`/`heap_end` bound the window region `malloc` hands out (both window
/// offsets, `heap_base <= heap_end`, within the guest window and clear of the guest's static
/// data/stack). `stdin` preloads standard input for `read(0, …)`. Every libc import in a linked module
/// shares this **one** handle (temen-wasm/chibicc thread a single capability handle); the op number
/// distinguishes the call, so pass the handle as the entry's leading argument.
/// #796 L2 — the interp-facing async-signal door: the [`SignalSource`] the `Host` holds. It locks
/// **its own process's** [`Proc`] on demand (the interp holds no personality lock at a safepoint; a
/// proc-only lock, never the world — see [`World`]'s lock order) and returns the next deliverable
/// handler, keeping POSIX signal *policy* (dispositions, mask, pending) inside this personality
/// while the interp only performs the *mechanism* (the safepoint redirect, invariant 4). #863: one
/// door per process — a fork twin's door locks the twin's `Proc`, so its signals are its own.
struct SignalDoor(Arc<Mutex<Proc>>);

impl SignalSource for SignalDoor {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        let mut st = self.0.lock().unwrap_or_else(|e| e.into_inner());
        // #798 slice 2 — a STOPPED process handles its signals at continue, not while stopped
        // (POSIX): hold everything pending. Also closes the stop-fire race — the bookkeeping stop
        // is synchronous under this lock, the domain flag lands a beat later, and in between the
        // process must not consume a signal it should sleep on.
        if st.stopped_sig.is_some() {
            st.sig_armed.store(false, Ordering::Relaxed);
            return None;
        }
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
                // A caught handler: deliver it. #796 block-during-handler — push the current mask
                // and block the delivered signal + its `sa_mask` for the handler's duration
                // (POSIX; `handler_returned` restores). This is also what makes NESTED delivery
                // safe: a further signal may interrupt the handler, but never this same one.
                st.note_delivery_flags(s); // #796 SA_RESTART (the safepoint dispatch also sweeps parks)
                let saved = st.sig_mask;
                st.handler_mask_stack.push(saved);
                let sa = st.sig_action_mask.get(&s).copied().unwrap_or(0);
                st.sig_mask = (saved | (1u64 << s) | sa) & !UNMASKABLE;
                // Re-arm iff another signal is deliverable UNDER THE HANDLER MASK, so the interp
                // returns for it (a nested delivery) or picks it up at the handler's return.
                let more = (st.sig_pending & !st.sig_mask) != 0;
                st.sig_armed.store(more, Ordering::Relaxed);
                return Some((handler as i32, s, sp));
            }
            // SIG_DFL / SIG_IGN: nothing to run async — ignored signals are discarded at generation
            // and default-terminate fires through the kill door (#796), so anything still here is a
            // stale pending bit; drop it and keep scanning for a caught one.
        }
    }

    /// #799 L1 — store the interp's scheduler-wake closure so [`Posix::raise_signal`] can interrupt a
    /// parked blocking syscall on an embedder `^C`. Installed at run start, cleared to a no-op at teardown.
    fn set_pipe_wake(&self, wake: Arc<dyn Fn(u32) + Send + Sync>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).pipe_wake = Some(wake);
    }

    fn set_wake(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).wake = Some(wake);
    }

    /// #798 slice 2 — store the core's stop/continue closure for this process's domain.
    fn set_stop(&self, stop: Arc<dyn Fn(bool) + Send + Sync>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).stop = Some(stop);
    }

    /// #796 default actions — store the core's terminate closure for this process's domain.
    fn set_kill(&self, kill: Arc<dyn Fn() + Send + Sync>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).kill = Some(kill);
    }

    /// #799 — store the core's park-request closure for this process's domain.
    fn set_park_request(&self, req: Arc<dyn Fn(temen_interp::ParkEvent) + Send + Sync>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).park_req = Some(req);
    }

    /// #796 `SA_RESTART` — answer the park sites: does the delivery behind the just-consumed
    /// interrupt want the blocking call restarted?
    fn syscall_restart(&self) -> bool {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).restart_ok
    }

    /// #796 slice D — the pre-park pending check: a deliverable (caught, unmasked, async-on,
    /// not-stopped) signal is pending right now, so an about-to-insert interruptible park should
    /// complete `-EINTR` instead of blocking through it.
    fn interrupt_pending(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .deliverable_now()
    }

    /// #1171 — read-and-clear the one-shot child-transition edge (set by [`Ctx::notify_parent_chld`]
    /// / [`Posix::chld_to`] on this process when a child of it stopped or continued).
    fn reap_pending(&self) -> bool {
        let mut p = self.0.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut p.reap_wake)
    }

    /// #1171 — is this process stopped (`stopped_sig` set, before `SIGCONT`)? Gates the read re-admit.
    fn stopped(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stopped_sig
            .is_some()
    }

    /// #796 block-during-handler — an injected handler frame returned: restore the pre-delivery
    /// mask pushed by [`SignalDoor::take_deliverable`], then act on what the unblocking exposes —
    /// a now-deliverable caught signal re-arms (the running vCPU picks it up at its next per-op
    /// poll, no wake needed), and a held **fatal** signal runs its default action (the kill fire,
    /// invoked here directly: the interp holds no locks at this call, and the fire's scheduler
    /// work is safe from a running vCPU).
    fn handler_returned(&self) {
        let kill_fire = {
            let mut st = self.0.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(saved) = st.handler_mask_stack.pop() {
                st.sig_mask = saved & !UNMASKABLE;
            }
            st.arm_signals();
            st.dispatch_default_actions()
        };
        if let Some(f) = kill_fire {
            f();
        }
    }
}

/// #801 — the personality's published **op vtable**: `(names, sigs)` indexed by op number, the
/// machine-checked form of this file's op table (POSIX.md §4's "pin the ABI", enforced at bind).
/// Names are the guest-manifest names verbatim (`__px_` + the [`resolve`] name); signatures are
/// the **manifest** shapes chibicc emits: the `int cap` dummy becomes the call's HANDLE OPERAND
/// (IMPORTS.md §2.5) and is NOT part of the import signature, and every remaining arg widens to
/// `i64` — so op `k` with `n` real args is `(i64 × n) -> (i64)` (`exit` alone is `(i64) -> ()`).
/// A command
/// whose declared shim signature drifts from this table is refused at bind — a clean `-EINVAL`
/// from `execve`, never a runtime misdispatch.
fn px_vtable() -> (Vec<String>, Vec<temen_ir::FuncType>) {
    use temen_ir::ValType::I64;
    // (bare resolve-name, i64 params after the dummy). Order IS the op number.
    const OPS: &[(&str, usize)] = &[
        ("write", 3),        // 0
        ("read", 3),         // 1
        ("malloc", 1),       // 2
        ("free", 1),         // 3
        ("exit", 0),         // 4 (special-cased below: (i32, i32) -> ())
        ("open", 3),         // 5
        ("close", 1),        // 6
        ("lseek", 3),        // 7
        ("unlink", 2),       // 8
        ("getcwd", 2),       // 9
        ("chdir", 2),        // 10
        ("getenv", 2),       // 11
        ("setenv", 5),       // 12
        ("stat", 3),         // 13
        ("opendir", 2),      // 14
        ("readdir", 3),      // 15
        ("closedir", 1),     // 16
        ("argc", 0),         // 17
        ("argv", 3),         // 18
        ("exec_lookup", 2),  // 19
        ("exec_stdout", 0),  // 20
        ("exec_stdin", 2),   // 21
        ("exec_win", 1),     // 22
        ("pipe", 1),         // 23
        ("dup2", 2),         // 24
        ("dup", 1),          // 25
        ("fcntl", 3),        // 26
        ("spawn", 4),        // 27
        ("waitpid", 3),      // 28
        ("wait", 1),         // 29
        ("signal", 2),       // 30
        ("kill", 2),         // 31
        ("sigcheck", 1),     // 32
        ("clock", 1),        // 33
        ("getenv_r", 4),     // 34
        ("unsetenv", 2),     // 35
        ("environ", 3),      // 36
        ("mkdir", 3),        // 37
        ("rename", 4),       // 38
        ("rmdir", 2),        // 39
        ("sigprocmask", 3),  // 40
        ("sigaction", 3),    // 41
        ("sigaltstack", 2),  // 42
        ("spawn2", 1),       // 43
        ("getpid", 0),       // 44
        ("setpgid", 2),      // 45
        ("getpgid", 1),      // 46
        ("tcgetpgrp", 1),    // 47
        ("tcsetpgrp", 2),    // 48
        ("isatty", 1),       // 49
        ("getppid", 0),      // 50
        ("fork", 1),         // 51
        ("pipe_adopt", 3),   // 52
        ("exec_resolve", 2), // 53
        ("tcgetattr", 2),    // 54
        ("tcsetattr", 2),    // 55
        ("tcgetwinsize", 2), // 56
    ];
    let mut names = Vec::with_capacity(OPS.len());
    let mut sigs = Vec::with_capacity(OPS.len());
    for (op, (name, nargs)) in OPS.iter().enumerate() {
        // The table order must agree with `resolve` — a drift here is a bind-time refusal in
        // every vtable consumer, but pin it eagerly too.
        debug_assert_eq!(resolve(name).map(|c| c.op), Some(op as u32), "{name}");
        names.push(format!("__px_{name}"));
        let params = vec![I64; if *name == "exit" { 1 } else { *nargs }];
        let results = if *name == "exit" {
            Vec::new()
        } else {
            vec![I64]
        };
        sigs.push(temen_ir::FuncType { params, results });
    }
    (names, sigs)
}

pub fn grant(host: &mut Host, heap_base: u64, heap_end: u64, stdin: Vec<u8>) -> (i32, Posix) {
    let world = Arc::new(Mutex::new(new_world(stdin)));
    let root = Arc::new(Mutex::new(new_proc(heap_base, heap_end)));
    let posix = Posix {
        world: Arc::clone(&world),
        root: Arc::clone(&root),
    };
    // #863 slice 2 — the root is pid 1 in the process table (so a child can `kill(1, sig)` its
    // init-like parent; `kill` short-circuits to the self path when the root signals itself).
    world
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .procs
        .insert(1, ProcEntry::Live(Arc::clone(&root)));
    // #796 L2 — install the async-signal source + the shared `armed` flag (the *same* `Arc` the
    // personality mutates), so the interp can redirect into a handler at a safepoint (PROCESS.md §9 L2).
    let armed = root
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .sig_armed
        .clone();
    host.set_signal_source(Arc::new(SignalDoor(Arc::clone(&root))), armed);
    // FORK.md PR 5 / #863 — grant **forkable**: the factory clones the per-process side by POSIX's
    // rules ([`Proc::fork`]) and shares the [`World`], so a `fork()` twin gets its own fd table /
    // cwd / env / signal state over the shared memfs and open-file descriptions — real POSIX fork
    // semantics. `Host::fork_powerbox` calls this factory to carry libc into the twin, instead of
    // failing closed on an opaque closure.
    let root_for_remap = Arc::clone(&root);
    let handle = host.grant_host_proc_forkable(
        handler(Arc::clone(&world), Arc::clone(&root)),
        fork_factory(world, root),
    );
    // #801 — publish the op vtable on the grant: what lets an exec'd/spawned `__px_`-linked
    // image's manifest bind through the §3.5 coverage walk, signature-checked, with no external
    // resolver — the op knowledge travels with the grant.
    let (names, sigs) = px_vtable();
    host.set_host_proc_vtable(handle, names, sigs);
    // #972 — the root process's exec-remap hook: an execve from the root re-points its adopted
    // pipe fds at the carried ends' new handles (fork twins get theirs from the factory).
    host.push_exec_remap_hook(exec_remap_hook(root_for_remap));
    (handle, posix)
}

/// #863 — the self-replicating fork factory over one process: mints a `fork()` twin's handler +
/// signal door over a fresh [`Proc::fork`] clone (world shared), and a **replacement factory** over
/// the twin's own `Proc` — so fork-of-fork clones the twin's state, not the grandparent's.
///
/// Slice 2: the twin is born with its **pid** (the scheduler `TaskId` the parent's `fork()`
/// returns — the factory's argument) and registered in the world's process table, so `kill(pid)`
/// can target it from anywhere in the world. Pid `0` is an **anonymous** mint (the spawned-child
/// re-grant path, where no `TaskId` exists yet); slice 3 gives such a clone a pid from the
/// personality's own allocator instead — every process is table-addressable, so an embedder
/// ([`Posix::kill_pid`]) or a sibling can signal a spawned shell just like a fork twin.
/// #972 — the **exec remap hook** over one process: the core's image-replace carried the
/// process's pipe ends into fresh powerbox slots and reports the `(old, new)` handle pairs; walk
/// the fd table and re-point every [`CorePipeToken`] naming an old handle (dup groups share one
/// token — the atomic store covers the group). Policy here, mechanism in the core (invariant 4).
fn exec_remap_hook(proc_: Arc<Mutex<Proc>>) -> temen_interp::ExecRemapHook {
    Arc::new(move |pairs: &[(i32, i32)]| {
        let mut p = proc_.lock().unwrap_or_else(|e| e.into_inner());
        // #801 — a committed exec re-bases the heap to the new image's own convention (the old
        // image's allocations died with it; POSIX brk is per-image).
        if let Some((base, end)) = p.pending_exec_heap.take() {
            p.heap_next = base;
            p.heap_end = end;
            p.free_list.clear();
            p.allocated.clear();
        }
        for entry in p.fds.iter().flatten() {
            if let FdEntry::CorePipe(t) = entry {
                let cur = t.get();
                if let Some((_, nh)) = pairs.iter().find(|(o, _)| *o == cur) {
                    t.handle.store(*nh, Ordering::Relaxed);
                }
            }
        }
        // #797 interactive rung 2 — the terminal input end rides the same exec carry (it is a
        // PipeEnd binding in the old powerbox); re-point this process's terminal token like any
        // adopted pipe end, so an exec'd child's `read(0)` still taps the terminal.
        if let Some(t) = p.term_in.as_ref() {
            let cur = t.get();
            if let Some((_, nh)) = pairs.iter().find(|(o, _)| *o == cur) {
                t.handle.store(*nh, Ordering::Relaxed);
            }
        }
    })
}

fn fork_factory(world: Arc<Mutex<World>>, proc_: Arc<Mutex<Proc>>) -> HostProcFork {
    Arc::new(move |pid: u64| {
        // World before Proc — the canonical order. (This runs under the core's scheduler lock;
        // no dispatch fires a scheduler wake while holding the world lock — see [`Ctx::wake_after`]
        // — so scheduler → world here cannot cross a world → scheduler hold.)
        let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
        let mut child = proc_.lock().unwrap_or_else(|e| e.into_inner()).fork();
        // #799 — a non-zero pid IS the twin's scheduler TaskId: exactly the processes the core's
        // twin-completion wake covers, so exactly the ones blocking `waitpid` may bench on.
        child.core_task = pid != 0;
        let pid = if pid != 0 {
            pid as i32
        } else {
            // Anonymous mint: allocate from the same space spawn zombies use (skip occupied).
            while w.procs.contains_key(&w.next_pid) {
                w.next_pid += 1;
            }
            let p = w.next_pid;
            w.next_pid += 1;
            p
        };
        child.pid = pid;
        let armed = child.sig_armed.clone();
        let child = Arc::new(Mutex::new(child));
        w.procs.insert(pid, ProcEntry::Live(Arc::clone(&child)));
        drop(w);
        // #863 hygiene — the twin's exit retires its table entry: the core fires this with the
        // raw exit status when the twin's task completes, and the process becomes a reapable
        // zombie (`waitpid` then serves fork twins exactly like spawn children). Wait-encoding is
        // OUR policy — WEXITSTATUS in bits 8–15, the same encode `spawn_core` uses; a crashed
        // twin arrives as the core's crash status (128, already shell-`$?`-shaped) and encodes
        // like any exit code. #796 default actions: a twin the delivery gate terminated
        // (`term_sig` set — the core's kill is signal-blind, only this bookkeeping knows why)
        // retires in the `WIFSIGNALED` shape instead: the signal in the low 7 bits.
        let exit_world = Arc::clone(&world);
        let exit_child = Arc::clone(&child);
        let exit = Arc::new(move |status: i64| {
            let (term, pgid, ppid) = {
                let c = exit_child.lock().unwrap_or_else(|e| e.into_inner());
                (c.term_sig, c.pgid, c.ppid)
            };
            let encoded = match term {
                Some(sig) => sig & 0x7f,
                None => ((status & 0xff) << 8) as i32,
            };
            let mut w = exit_world.lock().unwrap_or_else(|e| e.into_inner());
            w.procs.insert(
                pid,
                ProcEntry::Zombie {
                    status: encoded,
                    pgid,
                    ppid,
                },
            );
            // #802 rung 3 — the exit is a child transition: pend SIGCHLD in the parent, ARM
            // ONLY (the returned fire is deliberately dropped — this hook runs under the core's
            // scheduler lock, and the parent's run-wake would re-take it). A parent benched in
            // a blocking `waitpid` is woken by the core's own twin-completion drain; a parent
            // parked elsewhere consumes the pending signal at its next delivery point.
            if let Some(ProcEntry::Live(t)) = w.procs.get(&ppid) {
                let _ = t
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .deliver_signal(SIGCHLD);
            }
        });
        ForkedProc {
            handler: handler(Arc::clone(&world), Arc::clone(&child)),
            signal: Some((Arc::new(SignalDoor(Arc::clone(&child))), armed)),
            refork: Some(fork_factory(Arc::clone(&world), Arc::clone(&child))),
            exit: Some(exit),
            exec_remap: Some(exec_remap_hook(child)),
        }
    })
}

/// Build the personality as a re-grantable **capability factory** for the powerbox model (temen-run's
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
    let world = Arc::new(Mutex::new(new_world(stdin)));
    let root = Arc::new(Mutex::new(new_proc(heap_base, heap_end)));
    let posix = Posix {
        world: Arc::clone(&world),
        root: Arc::clone(&root),
    };
    // #863 slice 2 — the root is pid 1 in the process table (so a child can `kill(1, sig)` its
    // init-like parent; `kill` short-circuits to the self path when the root signals itself).
    world
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .procs
        .insert(1, ProcEntry::Live(Arc::clone(&root)));
    // Per-backend mint over the SAME world+proc: the interp and JIT hosts are two engines of one
    // process, so they share both sides (unlike a fork, which clones the proc side).
    let make = move || handler(Arc::clone(&world), Arc::clone(&root));
    (posix, make)
}

/// #863 — the [`fork_factory`] over an existing [`cap`]/[`grant`] personality's **root process**,
/// for embedders that grant the personality through the powerbox path (`HostCap`) and want real
/// POSIX fork semantics there too (temen-run's `posix_cap`).
pub fn cap_fork_factory(posix: &Posix) -> HostProcFork {
    fork_factory(Arc::clone(&posix.world), Arc::clone(&posix.root))
}

/// The interp-facing **async-signal door** (+ its shared `armed` flag) over an existing
/// personality's root process — what [`grant`] installs via `Host::set_signal_source` on the
/// name-binding path, exposed so the powerbox path (`HostCap` — temen-run's `posix_cap`) can wire
/// the same thing. Without it the lane has no async delivery AND no **caller-request door**
/// (#799: `set_park_request` rides the signal source), so `fork` (op 51) returns `-ENOSYS` —
/// bash's fork-retry loop then spins forever (#802 slice 4 found it exactly this way).
pub fn cap_signal_source(
    posix: &Posix,
) -> (
    Arc<dyn SignalSource + Send + Sync>,
    Arc<std::sync::atomic::AtomicBool>,
) {
    let armed = posix
        .root
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .sig_armed
        .clone();
    (Arc::new(SignalDoor(Arc::clone(&posix.root))), armed)
}

/// The #972 **exec-remap hook** over an existing personality's root process (see
/// [`exec_remap_hook`]) — the powerbox-path counterpart of the install [`grant`] does.
pub fn cap_exec_remap_hook(posix: &Posix) -> temen_interp::ExecRemapHook {
    exec_remap_hook(Arc::clone(&posix.root))
}

/// The #801 op **vtable** (names + signatures, op-ordered) — what [`grant`] publishes via
/// `Host::set_host_proc_vtable` so an exec'd image's `__px_*` manifest binds through the §3.5
/// coverage walk. Exposed for the powerbox path.
pub fn cap_vtable() -> (Vec<String>, Vec<temen_ir::FuncType>) {
    px_vtable()
}

/// The **`net` capability** as a factory over an existing personality (POSIX.md §5a) — the same
/// shape as [`cap`], granted under its **own name** (e.g. `"net"`). Each call produces a `HostProc`
/// over the *same* shared state, so the socket fds it mints live in the same fd table the libc
/// `read`/`write`/`close`/`dup2` ops serve — the data plane needs no new surface.
pub fn net_cap_factory(posix: &Posix) -> impl Fn() -> HostProc + Send + Sync + 'static {
    let world = Arc::clone(&posix.world);
    let root = Arc::clone(&posix.root);
    move || net_handler(Arc::clone(&world), Arc::clone(&root))
}

/// Build the `net` capability's [`HostProc`] handler over shared `inner`: the tiny authority surface
/// (connect / bind / accept / shutdown / resolve). An unknown op is a clean `CapFault`.
fn net_handler(world: Arc<Mutex<World>>, proc_: Arc<Mutex<Proc>>) -> HostProc {
    Box::new(
        move |op, args, mem, _minter: Option<&mut dyn temen_interp::RegionMinter>| {
            let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
            let mut p = proc_.lock().unwrap_or_else(|e| e.into_inner());
            let mut st = Ctx {
                w: &mut w,
                p: &mut p,
                wake_after: Vec::new(),
            };
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

/// A fresh shared [`World`]: preloaded `stdin`, empty memfs, no delegates. Shared by [`grant`] and
/// [`cap`]; every process of the personality shares this one.
fn new_world(stdin: Vec<u8>) -> World {
    World {
        stdout: Vec::new(),
        stdout_sink: None,
        stderr: Vec::new(),
        stdin,
        stdin_pos: 0,
        files: HashMap::new(),
        explicit_dirs: HashSet::new(),
        net_listeners: HashMap::new(),
        net_next_port: 49152, // the IANA ephemeral range start
        net_delegate: None,
        #[cfg(not(target_arch = "wasm32"))]
        clock_base: std::time::Instant::now(),
        #[cfg(target_arch = "wasm32")]
        clock_tick: std::sync::atomic::AtomicI64::new(0),
        clock_fixed: None,
        commands: Vec::new(),
        executables: HashSet::new(),
        terminal: None,
        exec_stdout_handle: 0,
        exec_stdin_handle: 0,
        exec_stdin_fifo: None,
        spawn_fn: None,
        procs: HashMap::new(),
        next_pid: 1000,
        fg_pgid: 1,
    }
}

/// A fresh [`Proc`]: the window-heap arena bounded by `[heap_base, heap_end)`, the three stdio
/// sentinels seeded in the fd table, default signal state.
fn new_proc(heap_base: u64, heap_end: u64) -> Proc {
    Proc {
        pid: 1,  // the root process — init-like; fork twins get their TaskId stamped by the factory
        ppid: 0, // no recorded parent (#800 getppid)
        pgid: 1, // the root leads process group 1
        pending_exec_heap: None,
        heap_next: heap_base,
        heap_end,
        allocated: HashMap::new(),
        free_list: Vec::new(),
        fds: vec![
            Some(FdEntry::Stdin),
            Some(FdEntry::Stdout),
            Some(FdEntry::Stderr),
        ],
        dirs: Vec::new(),
        args: Vec::new(),
        cwd: "/".to_string(),
        env: HashMap::new(),
        env_ptrs: HashMap::new(),
        sig_pending: 0,
        sig_handler: HashMap::new(),
        sig_mask: 0,
        sig_action_mask: HashMap::new(),
        sig_action_flags: HashMap::new(),
        sig_stack_base: 0,
        sig_armed: Arc::new(AtomicBool::new(false)),
        wake: None,
        pipe_wake: None,
        stop: None,
        stopped_sig: None,
        stop_fresh: false,
        cont_fresh: false,
        reap_wake: false,
        kill: None,
        term_sig: None,
        handler_mask_stack: Vec::new(),
        restart_ok: false,
        park_req: None,
        core_task: false,
        term_in: None,
    }
}

/// Build the POSIX [`HostProc`] handler for one process: the shared [`World`] + this process's
/// [`Proc`]. Dispatches on the op number; an unknown op on this handle is a `CapFault` (as for any
/// capability). Both locks are taken at dispatch top in the canonical order (world, then proc — see
/// [`World`]), exactly the blanket-lock scope the pre-split single mutex had.
fn handler(world: Arc<Mutex<World>>, proc_: Arc<Mutex<Proc>>) -> HostProc {
    Box::new(
        move |op, args, mem, _minter: Option<&mut dyn temen_interp::RegionMinter>| {
            let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
            let mut p = proc_.lock().unwrap_or_else(|e| e.into_inner());
            let mut st = Ctx {
                w: &mut w,
                p: &mut p,
                wake_after: Vec::new(),
            };
            if std::env::var_os("TEMEN_PX_TRACE").is_some() {
                if op == OP_WRITE {
                    let txt = mem
                        .as_ref()
                        .and_then(|m| {
                            m.read_bytes(
                                args.get(1).copied().unwrap_or(0) as u64,
                                args.get(2).copied().unwrap_or(0).max(0) as u64,
                            )
                        })
                        .map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default();
                    eprintln!(
                        "[px] op=0 fd={} {:?}",
                        args.first().copied().unwrap_or(-1),
                        txt
                    );
                } else {
                    eprintln!("[px] op={op} args={args:?}");
                }
            }
            let res = match op {
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
                OP_ARGC => Ok(vec![st.p.args.len() as i64]),
                OP_ARGV => st.argv(args, mem),
                OP_EXEC_LOOKUP => st.exec_lookup(args, mem),
                OP_EXEC_STDOUT => Ok(vec![st.w.exec_stdout_handle as i64]),
                OP_EXEC_STDIN => st.exec_stdin(args, mem),
                OP_EXEC_WIN => st.exec_win(args),
                OP_PIPE => st.pipe(args, mem),
                OP_PIPE_ADOPT => st.pipe_adopt(args, mem),
                OP_EXEC_RESOLVE => st.exec_resolve(args, mem),
                OP_TCGETATTR => st.tcgetattr(args, mem),
                OP_TCSETATTR => st.tcsetattr(args, mem),
                OP_TCGETWINSIZE => st.tcgetwinsize(args, mem),
                OP_DUP2 => Ok(vec![st.dup2(args)]),
                OP_DUP => Ok(vec![st.dup(args)]),
                OP_FCNTL => Ok(vec![st.fcntl(args)]),
                OP_SPAWN => st.spawn(args, mem),
                OP_SPAWN2 => st.spawn2(args, mem),
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
                OP_GETPID => Ok(vec![st.p.pid as i64]),
                OP_SETPGID => Ok(vec![st.setpgid(args)]),
                OP_GETPGID => Ok(vec![st.getpgid(args)]),
                OP_TCGETPGRP => Ok(vec![st.tcgetpgrp(args)]),
                OP_TCSETPGRP => Ok(vec![st.tcsetpgrp(args)]),
                OP_ISATTY => Ok(vec![st.isatty(args)]),
                OP_GETPPID => Ok(vec![st.p.ppid as i64]),
                OP_FORK => Ok(vec![st.fork_request()]),
                _ => Err(Trap::CapFault),
            };
            if std::env::var_os("TEMEN_PX_TRACE").is_some() && op == OP_READ {
                eprintln!("[px] read -> {res:?}");
            }
            // Fire a deferred cross-process wake (see [`Ctx::wake_after`]) only after both guards
            // drop — and from a **detached thread**: the interp invoked this handler while holding
            // our domain's `Host` lock, the wake takes the scheduler lock, and scheduler-lock
            // holders lock `Host`s (fork mints, park interrupts) — a same-stack fire could close a
            // host ↔ scheduler cycle. Cross-process signals are human-frequency, so a short-lived
            // thread is the boring, provably-unentangled choice. (The embedder paths —
            // [`Posix::raise_signal`], [`Posix::kill_pid`] — hold no `Host` lock and fire inline.)
            let wakes = std::mem::take(&mut st.wake_after);
            drop(p);
            drop(w);
            if !wakes.is_empty() {
                // Both guards are dropped, so no `Host`/scheduler lock is held here. On a
                // multi-threaded driver (the tree-walker's threads, `drive_parallel`) a same-stack
                // fire could still close a host ↔ scheduler cycle *via another thread*, so we detach
                // (cross-process signals are human-frequency — a short-lived thread is the boring,
                // provably-unentangled choice). wasm32 has no `thread::spawn` and is always
                // single-threaded (the cooperative `drive` or the tree-walker), where the wake
                // closures only ring a dedicated doorbell mutex the pump never holds across an interp
                // step — so there fire inline. Without this a guest `kill(SIGCONT)` from bash's `fg`
                // (or its stopped-job exit sweep) panicked the engine to a bare `unreachable` (#1171).
                #[cfg(target_arch = "wasm32")]
                for wk in wakes {
                    wk();
                }
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::spawn(move || {
                    for wk in wakes {
                        wk();
                    }
                });
            }
            res
        },
    )
}

impl Proc {
    /// #796 L2 — nudge the interp's per-op poll to check for delivery: set the shared `armed` flag when
    /// async delivery is *possible* (a signal stack is registered). `SignalSource::take_deliverable` is
    /// #798 slice 2 — **the one signal-delivery gate**: every raise aimed at this process (`kill`
    /// self/table/table-sweep, [`Posix::kill_pid`], [`Posix::raise_signal`], the TTOU/TTIN
    /// terminal check) routes here, so the job-control actions live in one place:
    ///
    /// - `SIGCONT`: clear a stop (mark it for `WCONTINUED`) and return the **continue** fire; a
    ///   caught `SIGCONT` also pends (POSIX: the handler runs after the continue).
    /// - `SIGSTOP`: stop, unconditionally (uncatchable, unignorable).
    /// - `SIGTSTP`/`SIGTTIN`/`SIGTTOU`: default disposition ⇒ **stop** (mark for `WUNTRACED`);
    ///   ignored ⇒ dropped; caught ⇒ the ordinary pending path.
    /// - anything else: pending + arm, the wake when deliverable (the pre-slice behavior).
    ///
    /// Returns the deferred fire — the target's run-wake, or its stop/continue closure wrapped to
    /// the right direction — for the caller to run **after its locks drop** ([`Ctx::wake_after`] /
    /// the embedder paths). A missing stop closure degrades to bookkeeping-only (see
    /// [`Proc::stop`]).
    fn deliver_signal(&mut self, sig: i32) -> Option<Arc<dyn Fn() + Send + Sync>> {
        let disposition = |p: &Proc, s: i32| p.sig_handler.get(&s).copied().unwrap_or(SIG_DFL);
        match sig {
            SIGCONT => {
                let was_stopped = self.stopped_sig.take().is_some();
                if was_stopped {
                    self.cont_fresh = true;
                    self.stop_fresh = false; // an unreported stop superseded by the continue
                }
                if disposition(self, SIGCONT) > SIG_IGN {
                    self.sig_pending |= 1 << SIGCONT;
                    self.note_delivery_flags(SIGCONT); // #796 SA_RESTART
                    self.arm_signals();
                }
                if was_stopped {
                    // #796 default actions — a fatal signal held while stopped runs its action at
                    // the continue (POSIX): the kill fire subsumes the continue fire (it wakes the
                    // stopped domain itself, and death beats stop at the poll).
                    if let Some(kf) = self.dispatch_default_actions() {
                        return Some(kf);
                    }
                    self.stop
                        .clone()
                        .map(|f| -> Arc<dyn Fn() + Send + Sync> { Arc::new(move || f(false)) })
                } else if self.deliverable_now() {
                    self.wake.clone()
                } else {
                    None
                }
            }
            SIGSTOP => self.enter_stop(SIGSTOP),
            SIGTSTP | SIGTTIN | SIGTTOU => match disposition(self, sig) {
                SIG_DFL => self.enter_stop(sig),
                SIG_IGN => None,
                _ => {
                    self.sig_pending |= 1 << sig;
                    self.note_delivery_flags(sig); // #796 SA_RESTART
                    self.arm_signals();
                    if self.deliverable_now() {
                        self.wake.clone()
                    } else {
                        None
                    }
                }
            },
            _ => {
                let disp = disposition(self, sig);
                // #796 default actions. Ignored — explicitly (`SIG_IGN`) or by default
                // (CHLD/URG/WINCH) — is discarded at generation (POSIX). This replaces the L0
                // posture of pending such signals forever.
                if disp == SIG_IGN || (disp == SIG_DFL && default_ignored(sig)) {
                    return None;
                }
                if disp == SIG_DFL {
                    // Default action: TERMINATE. Masked or stopped ⇒ held pending — the action
                    // runs at unblock (`sigprocmask`) / continue (`SIGCONT`), not at raise.
                    // `SIGKILL` alone can never be held (unmaskable, kills a stopped process).
                    if sig != SIGKILL
                        && (self.stopped_sig.is_some() || (1u64 << sig) & self.sig_mask != 0)
                    {
                        self.sig_pending |= 1 << sig;
                        return None;
                    }
                    self.term_sig = Some(sig);
                    return self.kill.clone();
                }
                // Caught: pend + arm. A stopped process holds delivery until continued
                // ([`SignalDoor::take_deliverable`]'s gate).
                self.sig_pending |= 1 << sig;
                self.note_delivery_flags(sig); // #796 SA_RESTART
                self.arm_signals();
                if self.deliverable_now() {
                    self.wake.clone()
                } else {
                    None
                }
            }
        }
    }

    /// #796 `SA_RESTART` — record whether `sig`'s action wants interrupted blocking calls
    /// restarted, at every point a caught delivery can fire a park interrupt. Plain `signal()`
    /// installs leave `sa_flags` at 0 (the SysV flavor: no restart) — only a `sigaction` with
    /// `SA_RESTART` opts in.
    fn note_delivery_flags(&mut self, sig: i32) {
        self.restart_ok = (self.sig_action_flags.get(&sig).copied().unwrap_or(0) & SA_RESTART) != 0;
    }

    /// #796 default actions — run the default action for a pending, unmasked, `SIG_DFL`
    /// **terminate**-action signal (lowest number first; POSIX leaves the order unspecified):
    /// consume it, record it as this process's terminating signal, and hand back the core's
    /// terminate fire for the caller's deferred-wake path. Job-control signals are excluded
    /// (their default actions have their own arms in [`Proc::deliver_signal`]); a stopped
    /// process holds everything until continued. Called wherever a held fatal signal can
    /// become actionable: unblock (`sigprocmask`), continue (`SIGCONT`), and a disposition
    /// reset to `SIG_DFL` (`signal`/`sigaction`).
    fn dispatch_default_actions(&mut self) -> Option<Arc<dyn Fn() + Send + Sync>> {
        if self.stopped_sig.is_some() {
            return None;
        }
        let mut cand = self.sig_pending & !self.sig_mask;
        while cand != 0 {
            let s = cand.trailing_zeros() as i32;
            cand &= !(1u64 << s);
            if matches!(s, SIGCONT | SIGSTOP | SIGTSTP | SIGTTIN | SIGTTOU) || default_ignored(s) {
                continue;
            }
            if self.sig_handler.get(&s).copied().unwrap_or(SIG_DFL) == SIG_DFL {
                self.sig_pending &= !(1u64 << s);
                self.term_sig = Some(s);
                return self.kill.clone();
            }
        }
        None
    }

    /// The stop half of the gate: record the stopping signal (report-once for `WUNTRACED`) and
    /// return the domain-stop fire. Already-stopped ⇒ nothing new to do or report.
    fn enter_stop(&mut self, sig: i32) -> Option<Arc<dyn Fn() + Send + Sync>> {
        if self.stopped_sig.is_some() {
            return None;
        }
        self.stopped_sig = Some(sig);
        self.stop_fresh = true;
        self.cont_fresh = false;
        self.stop
            .clone()
            .map(|f| -> Arc<dyn Fn() + Send + Sync> { Arc::new(move || f(true)) })
    }

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
        if self.stopped_sig.is_some() {
            return false; // #798 slice 2 — a stopped process delivers at continue, never before
        }
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

    /// #863 — POSIX `fork()` of the per-process state, in one place so the rules are auditable:
    /// the fd **table** is copied entry-wise over **shared** descriptions ([`FdEntry::dup_clone`] —
    /// offsets and pipe liveness stay shared, POSIX fork-shares-open-file-descriptions), cwd / env /
    /// args are copied, the allocator bookkeeping is copied (it describes the twin's private window
    /// copy equally), signal dispositions / mask / `sigaction` extras / the signal stack are copied,
    /// and **pending signals are cleared** (POSIX: a child starts with none). The twin gets a fresh
    /// `armed` flag (its door is its own) and no wake (each run installs its own).
    fn fork(&self) -> Proc {
        Proc {
            pid: 0, // stamped by [`fork_factory`]: the twin's TaskId, or an allocated pid (re-grant)
            ppid: self.pid, // #800 getppid — the forking process is the parent
            pgid: self.pgid, // POSIX: a fork twin inherits its parent's process group
            pending_exec_heap: self.pending_exec_heap,
            heap_next: self.heap_next,
            heap_end: self.heap_end,
            allocated: self.allocated.clone(),
            free_list: self.free_list.clone(),
            fds: {
                // #972 — entry-wise copy over shared descriptions, EXCEPT CorePipe tokens, which
                // re-split per process: the twin's duplicated powerbox has its own end-counts, so
                // its last-close release decision must be its own. Intra-process dup groups are
                // preserved (one fresh token per distinct parent token).
                let mut split: HashMap<*const CorePipeToken, Arc<CorePipeToken>> = HashMap::new();
                self.fds
                    .iter()
                    .map(|s| {
                        s.as_ref().map(|e| match e {
                            FdEntry::CorePipe(t) => FdEntry::CorePipe(Arc::clone(
                                split
                                    .entry(Arc::as_ptr(t))
                                    .or_insert_with(|| Arc::new(CorePipeToken::new(t.get()))),
                            )),
                            other => other.dup_clone(),
                        })
                    })
                    .collect()
            },
            dirs: self
                .dirs
                .iter()
                .map(|s| {
                    s.as_ref().map(|d| DirStream {
                        entries: d.entries.clone(),
                        pos: d.pos,
                    })
                })
                .collect(),
            args: self.args.clone(),
            cwd: self.cwd.clone(),
            env: self.env.clone(),
            env_ptrs: self.env_ptrs.clone(),
            sig_pending: 0,
            sig_handler: self.sig_handler.clone(),
            sig_mask: self.sig_mask,
            sig_action_mask: self.sig_action_mask.clone(),
            sig_action_flags: self.sig_action_flags.clone(),
            sig_stack_base: self.sig_stack_base,
            sig_armed: Arc::new(AtomicBool::new(false)),
            wake: None,
            pipe_wake: None,
            stop: None, // the twin's own domain gets its closure at mint (like the wake)
            stopped_sig: None,
            stop_fresh: false,
            cont_fresh: false,
            reap_wake: false,
            kill: None, // ditto — the twin's own terminate closure lands at mint
            term_sig: None,
            handler_mask_stack: self.handler_mask_stack.clone(), // forked mid-handler: the twin restores on its inherited return (POSIX fork copies signal state)
            restart_ok: self.restart_ok,
            park_req: None, // the twin's own door lands at mint, like the wake/stop/kill
            core_task: false, // stamped by [`fork_factory`] beside the pid
            // #797 — the twin's own terminal token: same handle value (the twin's cloned
            // powerbox table keeps it valid), its own cell (an exec re-points per-process).
            term_in: self.term_in.as_ref().map(|t| CorePipeToken::new(t.get())),
        }
    }
}

impl Ctx<'_> {
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
    /// its shared buffer. Factored out of [`Ctx::write`] so `spawn` can route a child's captured stdout
    /// to *whatever the caller's fd 1 currently is* (the fd-inheritance path). Empty `data` is a `0` no-op.
    fn sink_write(&mut self, fd: i64, data: &[u8]) -> i64 {
        if data.is_empty() {
            return 0;
        }
        // Decide the sink first (cloning the pipe `Arc`) so we don't hold a borrow of `self.p.fds` while
        // mutating `self.w.stdout`/`self.w.stderr`/the memfs.
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
            // #972 — an adopted core pipe end: redirect the shim to the core cap-call write
            // (backpressure park, `-EPIPE`; the shim raises SIGPIPE per disposition). Note this
            // also surfaces to `spawn`'s fd-1 routing — a CorePipe stdout for a spawn child is
            // out of slice-1 scope (exec carry, #972 slice 2) and fails closed here.
            Some(FdEntry::CorePipe(t)) => return PX_TAG_BASE - t.get() as i64,
            _ => Sink::Bad,
        };
        // #798 — a background write to the proto-terminal rings SIGTTOU (doorbell; the write still
        // proceeds until slice 2's stop). Unconditional pending TOSTOP: termios lands with #797.
        if matches!(sink, Sink::Stdout | Sink::Stderr) {
            self.tty_background_check(SIGTTOU);
        }
        match sink {
            Sink::Stdout => match &self.w.stdout_sink {
                Some(s) => s
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .extend_from_slice(data),
                None => self.w.stdout.extend_from_slice(data),
            },
            Sink::Stderr => self.w.stderr.extend_from_slice(data),
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
        // #797 — with the terminal enabled, fd 0 IS the terminal: tag-redirect to the input
        // pipe (park on empty, EINTR, one-shot ^D EOF — the #972 path). The Stdin sentinel
        // stays the preloaded-stdin world's binding.
        if fd == 0 && self.w.terminal.is_some() && matches!(self.fd(fd), Some(FdEntry::Stdin)) {
            // #798/#802 interactive rung 3 — a BACKGROUND read from the terminal rings SIGTTIN
            // (default action: STOP) before the tag mints, mirroring the write-side SIGTTOU
            // doorbell.
            self.tty_background_check(SIGTTIN);
            // Rung-3 tail — if that delivery STOPPED us (or a stop was already pending), the tag
            // must NOT mint: return the restart sentinel so the guest wrapper re-issues the op
            // and the stop lands at that dispatch's safepoint poll. The previously-documented
            // "park on the empty pipe, stop before the rewound read" ordering was a race: a
            // `bg`-continued reader whose stop fire lagged one dispatch consumed the next typed
            // line before parking (POSIX stops BEFORE the I/O; the `bg` probe caught the steal).
            if self.p.stopped_sig.is_some() {
                return Ok(vec![ERESTART]);
            }
            // #797 interactive rung 2 — mint the tag from THIS process's own token (handle
            // values are per-powerbox; the world's is the root namespace's). A process without
            // one (pre-terminal, spawn delegate) keeps the world handle.
            let h = self
                .p
                .term_in
                .as_ref()
                .map(|t| t.get())
                .unwrap_or_else(|| self.w.terminal.as_ref().unwrap().input_handle);
            return Ok(vec![PX_TAG_BASE - h as i64]);
        }
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
            // #972 — an adopted core pipe end: redirect the shim to the core cap-call read
            // (blocking/EINTR/EOF). No bytes move here; a non-shim caller sees a large negative
            // "error" and fails closed.
            Some(FdEntry::CorePipe(t)) => return Ok(vec![PX_TAG_BASE - t.get() as i64]),
            _ => Src::Bad,
        };
        let chunk: Vec<u8> = match src {
            Src::Stdin => {
                // #798 — a background read from the proto-terminal rings SIGTTIN (doorbell; the
                // read still proceeds until slice 2's stop).
                self.tty_background_check(SIGTTIN);
                let avail = &self.w.stdin[self.w.stdin_pos.min(self.w.stdin.len())..];
                let n = len.min(avail.len());
                self.w.stdin_pos += n;
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
        self.p.fds.get(fd as usize).and_then(|s| s.as_ref())
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
        // #802 language differential — `/dev/fd/N` and `/dev/std{in,out,err}` (bash builds with
        // `HAVE_DEV_FD`): process substitution `<(cmd)`/`>(cmd)` substitutes `/dev/fd/N` for the
        // pipe end it set up, and the consumer `open`s it. This is a path-driven `dup`: alias the
        // existing fd (sharing its description — an `Arc` clone of a CorePipe/pipe token, so the
        // refcount and last-close EOF stay correct). An unopened target is `-EBADF`, as on Linux.
        if let Some(target) = dev_fd_target(&path) {
            let dup = match self.fd(target) {
                Some(entry) => entry.dup_clone(),
                None => return Ok(vec![EBADF]),
            };
            return Ok(vec![self.alloc_fd(dup)]);
        }
        let exists = self.w.files.contains_key(&path);
        if !exists && flags & O_CREAT == 0 {
            return Ok(vec![ENOENT]);
        }
        let file = self.w.files.entry(path.clone()).or_default();
        if flags & O_TRUNC != 0 {
            file.clear();
        }
        let pos = if flags & O_APPEND != 0 { file.len() } else { 0 };
        let acc = flags & O_ACCMODE;
        let writable = acc == O_WRONLY || acc == O_RDWR;
        Ok(vec![self.alloc_fd(FdEntry::File(Arc::new(Mutex::new(
            OpenFile {
                path,
                pos,
                writable,
            },
        ))))])
    }

    /// `close(fd) -> 0 | -errno`: release any open fd (a file, a pipe end, or a stdio sentinel — a shell
    /// closes and reuses `0`/`1`/`2` freely). An out-of-range / already-closed fd is `-EBADF`.
    fn close(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&-1);
        if fd >= 0 {
            if let Some(slot) = self.p.fds.get_mut(fd as usize) {
                if let Some(entry) = slot.take() {
                    // Closing a memnet listener releases its port — but only if the registry still
                    // points at *this* listener's queue (a stale dup must not evict a later binder).
                    if let FdEntry::NetListener(l) = &entry {
                        if self
                            .w
                            .net_listeners
                            .get(&l.addr.port)
                            .is_some_and(|q| Arc::ptr_eq(q, &l.pending))
                        {
                            self.w.net_listeners.remove(&l.addr.port);
                        }
                    }
                    // #972 — the LAST close of an adopted core pipe end (no other dup holds the
                    // token) tags "release the handle": the shim's `__vm_close` then decrements
                    // the powerbox end-count, firing the EOF/EPIPE wakes. A non-last close is a
                    // plain 0 — the description stays open through its dups (POSIX).
                    if let FdEntry::CorePipe(t) = &entry {
                        if Arc::strong_count(t) == 1 {
                            return PX_TAG_BASE - t.get() as i64;
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
        let desc = match self.fd(fd) {
            Some(FdEntry::File(of)) => Arc::clone(of),
            Some(_) => return ESPIPE,
            None => return EBADF,
        };
        let mut of = desc.lock().unwrap_or_else(|e| e.into_inner());
        let size = self.w.files.get(&of.path).map_or(0, |f| f.len()) as i64;
        let newpos = match whence {
            SEEK_SET => offset,
            SEEK_CUR => of.pos as i64 + offset,
            SEEK_END => size + offset,
            _ => return EINVAL,
        };
        if newpos < 0 {
            return EINVAL;
        }
        of.pos = newpos as usize;
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

    /// #797 — `tcgetattr(fd, attr_ptr) -> 0 | -errno`: fill the 32-byte personality termios
    /// (`{lflag, cc[8], vmin, vtime}` as 4 LE i64s) for a terminal fd (stdio, with the terminal
    /// enabled); `-ENOTTY` otherwise.
    fn tcgetattr(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let ptr = *args.get(1).ok_or(Trap::Malformed)? as u64;
        // A dup'd tty fd (bash parks the terminal at fd 255) is the terminal too — the same
        // duplicated-sentinel rule `isatty`/`tcgetpgrp` follow, not a literal 0..=2.
        let is_term = self.fd_is_terminal(fd);
        let Some(t) = self.w.terminal.as_ref().filter(|_| is_term) else {
            return Ok(vec![ENOTTY]);
        };
        let mut out = Vec::with_capacity(32);
        out.extend_from_slice(&t.lflag.to_le_bytes());
        out.extend_from_slice(&t.cc);
        out.extend_from_slice(&t.vmin.to_le_bytes());
        out.extend_from_slice(&t.vtime.to_le_bytes());
        mem.write_bytes(ptr, &out).ok_or(Trap::Malformed)?;
        Ok(vec![0])
    }

    /// #797 — `tcsetattr(fd, attr_ptr) -> 0 | -errno`: apply the 32-byte termios immediately
    /// (TCSANOW; drain/flush `when` semantics are a shim concern for now). A line buffered in
    /// canonical mode stays buffered across a switch to raw — it flushes with the next feed.
    fn tcsetattr(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let ptr = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let bytes = mem.read_bytes(ptr, 32).ok_or(Trap::Malformed)?;
        // A dup'd tty fd (bash parks the terminal at fd 255) is the terminal too — the same
        // duplicated-sentinel rule `isatty`/`tcgetpgrp` follow, not a literal 0..=2.
        let is_term = self.fd_is_terminal(fd);
        let Some(t) = self.w.terminal.as_mut().filter(|_| is_term) else {
            return Ok(vec![ENOTTY]);
        };
        t.lflag = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
        t.cc.copy_from_slice(&bytes[8..16]);
        t.vmin = i64::from_le_bytes(bytes[16..24].try_into().unwrap());
        t.vtime = i64::from_le_bytes(bytes[24..32].try_into().unwrap());
        Ok(vec![0])
    }

    /// #797 — `tcgetwinsize(fd, ws_ptr) -> 0 | -errno`: `{i32 row; i32 col}` (the `TIOCGWINSZ`
    /// shape); `-ENOTTY` off-terminal. Updated by the embedder's [`Posix::set_winsize`], which
    /// also fires SIGWINCH at the foreground group.
    fn tcgetwinsize(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let ptr = *args.get(1).ok_or(Trap::Malformed)? as u64;
        // A dup'd tty fd (bash parks the terminal at fd 255) is the terminal too — the same
        // duplicated-sentinel rule `isatty`/`tcgetpgrp` follow, not a literal 0..=2.
        let is_term = self.fd_is_terminal(fd);
        let Some(t) = self.w.terminal.as_ref().filter(|_| is_term) else {
            return Ok(vec![ENOTTY]);
        };
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&t.rows.to_le_bytes());
        out.extend_from_slice(&t.cols.to_le_bytes());
        mem.write_bytes(ptr, &out).ok_or(Trap::Malformed)?;
        Ok(vec![0])
    }

    /// #972 — `pipe_adopt(read_h, write_h, fds_ptr) -> 0 | -errno`: record two guest-minted core
    /// pipe-end handles (`__vm_pipe`'s output) as [`FdEntry::CorePipe`] fds and store
    /// `[read_fd, write_fd]` (`i32`×2, read end first — POSIX order) at `fds_ptr`. Pure
    /// bookkeeping: the handles are numbers to this table (never exercised here — a garbage
    /// handle fails the guest's own later cap-call, not this op). Negative handles are `-EINVAL`.
    fn pipe_adopt(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let rh = *args.first().ok_or(Trap::Malformed)?;
        let wh = *args.get(1).ok_or(Trap::Malformed)?;
        let ptr = *args.get(2).ok_or(Trap::Malformed)? as u64;
        if rh < 0 || wh < 0 || rh > i32::MAX as i64 || wh > i32::MAX as i64 {
            return Ok(vec![EINVAL]);
        }
        let rfd = self.alloc_fd(FdEntry::CorePipe(Arc::new(CorePipeToken::new(rh as i32))));
        let wfd = self.alloc_fd(FdEntry::CorePipe(Arc::new(CorePipeToken::new(wh as i32))));
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
        if self.p.fds.len() <= n {
            self.p.fds.resize_with(n + 1, || None);
        }
        self.p.fds[n] = Some(dup);
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
                let out = self.w.stdin[self.w.stdin_pos.min(self.w.stdin.len())..].to_vec();
                self.w.stdin_pos = self.w.stdin.len();
                out
            }
            // A file has a bounded length; read from the offset to EOF in one shot.
            Src::File => {
                let n = match self.p.fds.get(fd as usize).and_then(|s| s.as_ref()) {
                    Some(FdEntry::File(of)) => {
                        let of = of.lock().unwrap_or_else(|e| e.into_inner());
                        self.w
                            .files
                            .get(&of.path)
                            .map_or(0, |f| f.len())
                            .saturating_sub(of.pos)
                    }
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
        let (name, argv) = match self.parse_spawn_target(args, mem)? {
            Ok(t) => t,
            Err(errno) => return Ok(vec![errno]),
        };
        // Classic spawn: inherit fd 0 / 1 / 2 (the `dup2` bracket is the guest's; here it's already
        // applied to the shared fd table). `spawn_core` with the `-1` sentinels is exactly that.
        Ok(vec![self.spawn_core(&name, &argv, -1, -1, -1)])
    }

    /// [`OP_SPAWN2`] — the parallel-safe spawn+capture. Reads the 44-byte request struct at `args[0]`
    /// (command target + three fd-actions; `-1` fd = inherit fd 0 / 1 / 2), then binds the child's stdio
    /// to *those* fds inside this one locked op — never mutating the shared fd-0/1/2 table, so two vCPUs
    /// capturing concurrently on the parallel driver cannot race (#848). See [`OP_SPAWN2`] for the layout.
    fn spawn2(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let req_ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let req = mem.read_bytes(req_ptr, 44).ok_or(Trap::Malformed)?;
        let rd_u64 = |o: usize| u64::from_le_bytes(req[o..o + 8].try_into().unwrap());
        let rd_i32 = |o: usize| i32::from_le_bytes(req[o..o + 4].try_into().unwrap()) as i64;
        // The command target rides the same four-word shape `parse_spawn_target` reads for `spawn`.
        let target = [
            rd_u64(0) as i64,
            rd_u64(8) as i64,
            rd_u64(16) as i64,
            rd_u64(24) as i64,
        ];
        let (name, argv) = match self.parse_spawn_target(&target, mem)? {
            Ok(t) => t,
            Err(errno) => return Ok(vec![errno]),
        };
        let stdin_fd = rd_i32(32);
        let stdout_fd = rd_i32(36);
        let stderr_fd = rd_i32(40);
        Ok(vec![
            self.spawn_core(&name, &argv, stdin_fd, stdout_fd, stderr_fd)
        ])
    }

    /// Parse the `(name_ptr, name_len, argv_ptr, argv_len)` prefix both spawn ops carry into the command
    /// name + `argv`. `Err(Trap::Malformed)` on an unreadable pointer; `Ok(Err(EINVAL))` on a non-UTF-8
    /// name; `Ok(Ok((name, argv)))` otherwise. `argv` is the blob split on NUL with trailing empties
    /// dropped; empty ⇒ `[name]` (argv[0] = program name).
    fn parse_spawn_target(
        &self,
        args: &[i64],
        mem: &dyn GuestMem,
    ) -> Result<Result<(String, Vec<String>), i64>, Trap> {
        let name_ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let name_len = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let argv_ptr = *args.get(2).unwrap_or(&0) as u64;
        let argv_len = (*args.get(3).unwrap_or(&0)).max(0) as u64;
        let name_bytes = mem.read_bytes(name_ptr, name_len).ok_or(Trap::Malformed)?;
        let Ok(name) = String::from_utf8(name_bytes) else {
            return Ok(Err(EINVAL));
        };
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
        Ok(Ok((name, argv)))
    }

    /// The shared spawn body both [`OP_SPAWN`] and [`OP_SPAWN2`] run: invoke the embedder's delegate on
    /// the parsed command with the child's stdin drained from `stdin_fd`, then route its captured stdout/
    /// stderr to `stdout_fd`/`stderr_fd`. Each fd is a `-1` sentinel for "inherit the caller's fd 0 / 1 /
    /// 2 binding" (classic `spawn`) or an explicit fd (per-child, parallel-safe `spawn2`). Returns the
    /// synthetic pid, or an errno (`ENOSYS` when no delegate is wired — fail closed *before* draining).
    fn spawn_core(
        &mut self,
        name: &str,
        argv: &[String],
        stdin_fd: i64,
        stdout_fd: i64,
        stderr_fd: i64,
    ) -> i64 {
        // Fail closed *before* any side effect (draining stdin) if no delegate is wired.
        if self.w.spawn_fn.is_none() {
            return ENOSYS;
        }
        // #972 slice 2 — a CorePipe stdio target fails closed with a probeable errno. The capture
        // spawn is synchronous (the child runs to completion inside this dispatch), so it cannot
        // consume or feed a *live* core pipe: a CorePipe stdin would need a blocking drain this
        // dispatch cannot perform, and a CorePipe stdout/stderr would need a cap-call write it
        // cannot issue — silently dropping the bytes (the pre-fix behavior) is the one wrong
        // answer. Live-pipe wiring belongs to fork + execve (#801), where the exec-replace's
        // named re-grant carries the ends.
        for (fd, dflt) in [(stdin_fd, 0), (stdout_fd, 1), (stderr_fd, 2)] {
            let eff = if fd < 0 { dflt } else { fd };
            if matches!(self.fd(eff), Some(FdEntry::CorePipe(_))) {
                return EINVAL;
            }
        }
        // The child inherits its stdin from `stdin_fd` (fd 0 by default) — drain it before the delegate.
        let stdin = self.drain_fd(if stdin_fd < 0 { 0 } else { stdin_fd });
        // Take the delegate out to call it (a `&mut self` method cannot also borrow the boxed closure),
        // then restore it.
        let mut f = self.w.spawn_fn.take().unwrap();
        let res = f(name, argv, &stdin);
        self.w.spawn_fn = Some(f);
        // Route the child's stdout/stderr to the requested fds (default: the caller's current fd 1 / fd
        // 2 — inheritance, as a prior `dup2(_, 1)` / `dup2(_, 2)` redirect lands each in a file or pipe).
        self.sink_write(if stdout_fd < 0 { 1 } else { stdout_fd }, &res.stdout);
        self.sink_write(if stderr_fd < 0 { 2 } else { stderr_fd }, &res.stderr);
        // One pid space (#863 slice 2): allocate past any pid the table already knows (fork twins
        // occupy their `TaskId`s, the root holds 1), then park the child as a reapable zombie.
        while self.w.procs.contains_key(&self.w.next_pid) {
            self.w.next_pid += 1;
        }
        let pid = self.w.next_pid;
        self.w.next_pid += 1;
        // Wait-encode the exit status: WEXITSTATUS occupies bits 8–15, low bits 0 (a normal exit).
        self.w.procs.insert(
            pid,
            ProcEntry::Zombie {
                status: (res.status & 0xff) << 8,
                pgid: pid,        // a spawn clone is its own group leader (never setpgid'd)
                ppid: self.p.pid, // the spawner owns this child (reap-ownership, #1080)
            },
        );
        pid as i64
    }

    /// `waitpid(pid, status_ptr, options) -> pid | -errno`: reap `pid` (or any pending child when
    /// `pid == -1`), writing its wait-encoded status to `status_ptr` when non-null. `options` (e.g.
    /// `WNOHANG`) is ignored — a spawned child has already run to completion, so a reap never blocks.
    /// `-ECHILD` when there is no such child.
    fn waitpid(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let pid = *args.first().ok_or(Trap::Malformed)?;
        let status_ptr = *args.get(1).unwrap_or(&0) as u64;
        // #863 — reap from the process table: [`ProcEntry::Zombie`] entries are reapable here —
        // completed spawn-delegate children, and (hygiene slice) **exited fork twins**, whose exit
        // hook flipped them Live → Zombie. A still-running twin is `-ECHILD` (this op never blocks
        // — a guest polls, or parks in the core's servicer reap, the wait offer, which serves the
        // same twin independently; use one channel per child).
        let is_zombie = |e: Option<&ProcEntry>| matches!(e, Some(ProcEntry::Zombie { .. }));
        // #1080 pipeline rung — reap ownership: `waitpid` reaps only the caller's OWN children (POSIX).
        // The zombie carries the `ppid` it exited with; a wildcard/`-pgid` reap that ignored it would let
        // a bash pipeline stage steal the reap the shell is blocked waiting for (the `echo | cat` wedge).
        let self_pid = self.p.pid;
        let reaped = if pid == -1 {
            self.w
                .procs
                .iter()
                .filter_map(|(p, e)| {
                    matches!(e, ProcEntry::Zombie { ppid, .. } if *ppid == self_pid).then_some(*p)
                })
                .min()
        } else if pid < -1 {
            // #799 — `waitpid(-pgid)`: group-reap the lowest OWN zombie whose pgid (retained on the
            // zombie entry) matches. Non-blocking like `-1` — the park door benches only
            // specific-pid waits this rung.
            let g = (-pid) as i32;
            self.w
                .procs
                .iter()
                .filter_map(|(p, e)| {
                    matches!(e, ProcEntry::Zombie { pgid, ppid, .. } if *pgid == g && *ppid == self_pid)
                        .then_some(*p)
                })
                .min()
        } else if is_zombie(self.w.procs.get(&(pid as i32))) {
            Some(pid as i32)
        } else {
            None
        };
        // #798 slice 2 — `WUNTRACED`/`WCONTINUED`: with no zombie to reap, a freshly-stopped
        // (or freshly-continued) live process matching the pid filter is reportable — once. The
        // entry stays in the table (the process is alive); only the fresh mark clears. Status
        // encodes the Linux wait word: stopped = `sig<<8 | 0x7f`, continued = `0xffff`.
        let opts = *args.get(2).unwrap_or(&0);
        if reaped.is_none() && (opts & (WUNTRACED | WCONTINUED)) != 0 {
            let self_pid = self.p.pid;
            let mut hit: Option<(i32, i32)> = None;
            for (&tpid, e) in self.w.procs.iter() {
                if tpid == self_pid {
                    continue; // never report ourselves
                }
                let ProcEntry::Live(t) = e else { continue };
                // Respect the pid filter: a named pid, `-pgid` (#799 — the live proc's own
                // pgid), or `-1` (any).
                if pid > 0 && pid != tpid as i64 {
                    continue;
                }
                if pid < -1 {
                    let g = (-pid) as i32;
                    if t.lock().unwrap_or_else(|e| e.into_inner()).pgid != g {
                        continue;
                    }
                }
                let mut tp = t.lock().unwrap_or_else(|e| e.into_inner());
                if (opts & WUNTRACED) != 0 && tp.stop_fresh {
                    tp.stop_fresh = false;
                    let sig = tp.stopped_sig.unwrap_or(SIGSTOP);
                    hit = Some((tpid, (sig << 8) | 0x7f));
                    break;
                }
                if (opts & WCONTINUED) != 0 && tp.cont_fresh {
                    tp.cont_fresh = false;
                    hit = Some((tpid, 0xffff));
                    break;
                }
            }
            if let Some((tpid, status)) = hit {
                if status_ptr != 0 {
                    mem.write_bytes(status_ptr, &status.to_le_bytes())
                        .ok_or(Trap::Malformed)?;
                }
                return Ok(vec![tpid as i64]);
            }
        }
        let Some(p) = reaped else {
            // #799 — blocking `waitpid`: nothing to reap and the caller did not opt out
            // (`WNOHANG`) or ask for stop/continue reports (those keep polling this rung).
            // If the target is a specific, Live, core-task twin — exactly the processes the
            // core's twin-completion wake covers — request the bench: the core rewinds this
            // very op and re-runs it against the retired entry when the child exits. The
            // `-ECHILD` below then doubles as the placeholder (parking routes discard it)
            // AND the degraded poll answer (non-parking routes/tiers keep the historical
            // spin — same results, decline-never-diverge).
            // #802 interactive — `WUNTRACED`/`WCONTINUED` no longer disqualify the bench:
            // interactive bash's foreground wait is `waitpid(-1, …, WUNTRACED)` (blocking, to
            // catch ^Z stops). A fresh stop/continue already reported above without reaching
            // here; the bench wakes on the child's EXIT (the twin-completion drain) or its
            // STOP (the `Blocked::Stopped` insert drain), and the re-executed op reports.
            if (opts & WNOHANG) == 0 {
                if let Some(req) = self.p.park_req.clone() {
                    let target: Option<temen_interp::ParkEvent> = if pid > 0 {
                        match self.w.procs.get(&(pid as i32)) {
                            Some(ProcEntry::Live(t)) => t
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .core_task
                                .then_some(temen_interp::ParkEvent::TaskExit(pid as u64)),
                            _ => None,
                        }
                    } else if pid == -1 {
                        // #802 slice 4 / rung 3 — the **any-child** blocking wait
                        // (`waitpid(-1, …)`, bash's `waitchld`: subshell, command-substitution,
                        // foreground-pipeline, and — with job control — the `WUNTRACED`
                        // foreground wait all ride this): with ≥1 live core-twin child, bench
                        // on the WILDCARD key (`ParkEvent::TaskExitAny`) — every child
                        // transition (exit, stop, continue) wakes it, so a never-exiting
                        // background child cannot absorb the bench while a foreground
                        // sibling's exit goes unnoticed (the failure the old lowest-live-child
                        // approximation had: `cat &` then a foreground command hung the shell
                        // on the stopped cat). Before slice 4, `-ECHILD` came back immediately
                        // and bash raced past its unfinished subshell.
                        let self_pid = self.p.pid;
                        self.w
                            .procs
                            .iter()
                            .any(|(&tpid, e)| {
                                let ProcEntry::Live(t) = e else { return false };
                                if tpid == self_pid {
                                    return false;
                                }
                                let tp = t.lock().unwrap_or_else(|e| e.into_inner());
                                tp.core_task && tp.ppid == self_pid
                            })
                            .then_some(temen_interp::ParkEvent::TaskExitAny)
                    } else {
                        None
                    };
                    if let Some(ev) = target {
                        req(ev);
                    }
                }
            }
            return Ok(vec![ECHILD]);
        };
        let status: i32 = match self.w.procs.remove(&p) {
            Some(ProcEntry::Zombie { status: st, .. }) => st,
            _ => unreachable!("reaped pid selected as a zombie above"),
        };
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
        // #796 default actions — POSIX forbids changing SIGKILL/SIGSTOP's action.
        if signum as i32 == SIGKILL || signum as i32 == SIGSTOP {
            return EINVAL;
        }
        let prev = self
            .p
            .sig_handler
            .insert(signum as i32, handler)
            .unwrap_or(SIG_DFL);
        self.p.arm_signals(); // #796 L2 — a handler installed for an already-pending signal may deliver now
        if handler == SIG_DFL {
            // #796 default actions — a reset to `SIG_DFL` runs a pending signal's action now.
            if let Some(f) = self.p.dispatch_default_actions() {
                self.wake_after.push(f);
            }
        }
        prev
    }

    /// `kill(pid, sig) -> 0 | -errno` — #863 slice 2: **pid-targeted** through the process table.
    /// `pid == 0` or the caller's own pid is the self path (`raise(s)`); any other pid is a table
    /// lookup — a live process gets **its** pending bit set (and its run woken when the signal is
    /// deliverable by **its** dispositions, the depth-agnostic `kill(child_pid, SIGINT)`); a spawn
    /// zombie exists-until-reaped (`0`, signal dropped); an unknown pid is `-ESRCH`. `sig == 0` is
    /// the POSIX existence probe. Negative pids (process groups) are `-EINVAL` (follow-up);
    /// out-of-range `sig` is `-EINVAL`.
    fn kill(&mut self, args: &[i64]) -> i64 {
        let pid = *args.first().unwrap_or(&0);
        let sig = *args.get(1).unwrap_or(&0);
        if !(0..=63).contains(&sig) {
            return EINVAL;
        }
        if pid < 0 {
            return self.kill_pgroup((-pid) as i32, sig); // kill(-pgid): the group sweep (#798)
        }
        // Self — pid 0 (the pre-table calling convention) or the caller's own pid. Never a table
        // lock: the caller's own `Proc` is already held by this dispatch.
        if pid == 0 || pid == self.p.pid as i64 {
            if sig == 0 {
                return 0; // kill(pid, 0): liveness probe — we exist
            }
            // Through the delivery gate (#798 slice 2): a self-raised SIGTSTP stops US at the
            // next per-op poll — the fire is deferred past this dispatch's locks like any other.
            let was_stopped = self.p.stopped_sig.is_some();
            if let Some(f) = self.p.deliver_signal(sig as i32) {
                self.wake_after.push(f);
            }
            // #802 rung 3 — a stop/continue is a child transition: SIGCHLD our parent.
            if was_stopped != self.p.stopped_sig.is_some() {
                let ppid = self.p.ppid;
                self.notify_parent_chld(ppid);
            }
            return 0;
        }
        // #863 slice 2 — any other pid is a process-table lookup.
        let transitioned: Option<i32> = match self.w.procs.get(&(pid as i32)) {
            Some(ProcEntry::Live(t)) => {
                // Not self (guarded above), so this is a different mutex — World → Proc, the
                // canonical order, serialized against other multi-`Proc` paths by the world lock.
                let mut tp = t.lock().unwrap_or_else(|e| e.into_inner());
                if sig == 0 {
                    return 0; // liveness probe — the target exists
                }
                // The delivery gate (#798 slice 2) decides pending/stop/continue by the TARGET's
                // dispositions; whatever it returns fires only after this dispatch's locks drop
                // ([`Ctx::wake_after`]), never under ours.
                let was_stopped = tp.stopped_sig.is_some();
                if let Some(f) = tp.deliver_signal(sig as i32) {
                    self.wake_after.push(f);
                }
                // #802 rung 3 — target stopped or continued: SIGCHLD its parent (after the
                // target's lock drops — sibling `Proc` locks never nest).
                (was_stopped != tp.stopped_sig.is_some()).then_some(tp.ppid)
            }
            // A zombie exists until reaped (POSIX: kill succeeds) but takes no signal.
            Some(ProcEntry::Zombie { .. }) => return 0,
            None => return ESRCH,
        };
        if let Some(ppid) = transitioned {
            self.notify_parent_chld(ppid);
        }
        0
    }

    /// `kill(-pgid, sig)` (#798) — the **process-group sweep**: raise `sig` on every live table
    /// process in group `pgid`, the caller included (POSIX: the whole group, sender too when it is
    /// a member). One deferred wake per deliverable member ([`Ctx::wake_after`]). `-ESRCH` when the
    /// group has no live member; a zombie member is skipped (it takes no signal); `sig == 0` probes
    /// group existence.
    fn kill_pgroup(&mut self, pgid: i32, sig: i64) -> i64 {
        let mut any = false;
        // #802 rung 3 — members the sweep stopped or continued: their parents get SIGCHLD
        // after the iteration (never while a member's `Proc` is held — sibling locks don't nest).
        let mut transitioned: Vec<i32> = Vec::new();
        // Self first (no table lock — this dispatch already holds our `Proc`).
        if self.p.pgid == pgid {
            any = true;
            if sig != 0 {
                let was_stopped = self.p.stopped_sig.is_some();
                if let Some(f) = self.p.deliver_signal(sig as i32) {
                    self.wake_after.push(f);
                }
                if was_stopped != self.p.stopped_sig.is_some() {
                    transitioned.push(self.p.ppid);
                }
            }
        }
        let self_pid = self.p.pid;
        for (&pid, entry) in self.w.procs.iter() {
            if pid == self_pid {
                continue; // handled above (and never a second lock on our own Proc)
            }
            let ProcEntry::Live(t) = entry else {
                continue; // a zombie exists but takes no signal
            };
            let mut tp = t.lock().unwrap_or_else(|e| e.into_inner());
            if tp.pgid != pgid {
                continue;
            }
            any = true;
            if sig != 0 {
                let was_stopped = tp.stopped_sig.is_some();
                if let Some(f) = tp.deliver_signal(sig as i32) {
                    self.wake_after.push(f);
                }
                if was_stopped != tp.stopped_sig.is_some() {
                    transitioned.push(tp.ppid);
                }
            }
        }
        for ppid in transitioned {
            self.notify_parent_chld(ppid);
        }
        if any {
            0
        } else {
            ESRCH
        }
    }

    /// [`OP_SETPGID`] — `setpgid(pid, pgid)`: move `pid` (`0` = self) into group `pgid` (`0` =
    /// `pid`'s own id). The caller's own move happens on its held `Proc`; any other target must be
    /// a live table process. `-EINVAL` on a negative `pgid`, `-ESRCH` on an unknown/zombie pid.
    fn setpgid(&mut self, args: &[i64]) -> i64 {
        let pid = *args.first().unwrap_or(&0);
        let pgid = *args.get(1).unwrap_or(&0);
        if pgid < 0 || pid < 0 {
            return EINVAL;
        }
        if pid == 0 || pid == self.p.pid as i64 {
            self.p.pgid = if pgid == 0 { self.p.pid } else { pgid as i32 };
            return 0;
        }
        match self.w.procs.get(&(pid as i32)) {
            Some(ProcEntry::Live(t)) => {
                let mut tp = t.lock().unwrap_or_else(|e| e.into_inner());
                tp.pgid = if pgid == 0 { tp.pid } else { pgid as i32 };
                0
            }
            _ => ESRCH,
        }
    }

    /// [`OP_GETPGID`] — `getpgid(pid)`: `pid`'s (`0` = self) process group, from the table.
    fn getpgid(&mut self, args: &[i64]) -> i64 {
        let pid = *args.first().unwrap_or(&0);
        if pid == 0 || pid == self.p.pid as i64 {
            return self.p.pgid as i64;
        }
        match self.w.procs.get(&(pid as i32)) {
            Some(ProcEntry::Live(t)) => t.lock().unwrap_or_else(|e| e.into_inner()).pgid as i64,
            _ => ESRCH,
        }
    }

    /// Is fd a handle on the proto-terminal (the captured stdio — the pty stand-in until #797)?
    fn fd_is_terminal(&mut self, fd: i64) -> bool {
        matches!(
            self.fd(fd),
            Some(FdEntry::Stdin | FdEntry::Stdout | FdEntry::Stderr)
        )
    }

    /// [`OP_TCGETPGRP`] — `tcgetpgrp(fd)`: the terminal's foreground process group. `-ENOTTY` off
    /// the proto-terminal.
    fn tcgetpgrp(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&0);
        if !self.fd_is_terminal(fd) {
            return ENOTTY;
        }
        self.w.fg_pgid as i64
    }

    /// [`OP_ISATTY`] — `isatty(fd) -> 1 | 0`: the same proto-terminal test the `tc*` ops gate on
    /// (`fd_is_terminal`), answered as C's boolean instead of `-ENOTTY`. Bash's interactive-mode
    /// probe (#800).
    fn isatty(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&0);
        i64::from(self.fd_is_terminal(fd))
    }

    /// [`OP_FORK`] (#799) — request the return-twice clone through the caller-request door and
    /// return the no-door placeholder; see [`OP_FORK`] for the full contract. Policy lives in the
    /// fork factory (table registration, inherited signal state/pgid/`ppid`) — this op only asks.
    fn fork_request(&mut self) -> i64 {
        if let Some(req) = self.p.park_req.clone() {
            req(temen_interp::ParkEvent::ForkSelf);
        }
        ENOSYS
    }

    /// [`OP_TCSETPGRP`] — `tcsetpgrp(fd, pgid)`: make `pgid` the foreground group. `-ENOTTY` off
    /// the proto-terminal, `-EINVAL` non-positive, `-EPERM` when no live process (the caller
    /// included) is in the group — a foreground nobody occupies is a stuck terminal.
    fn tcsetpgrp(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&0);
        let pgid = *args.get(1).unwrap_or(&0);
        if !self.fd_is_terminal(fd) {
            return ENOTTY;
        }
        if pgid <= 0 {
            return EINVAL;
        }
        let self_pid = self.p.pid;
        let occupied = self.p.pgid as i64 == pgid
            || self.w.procs.iter().any(|(&pid, e)| {
                pid != self_pid
                    && matches!(e, ProcEntry::Live(t)
                        if t.lock().unwrap_or_else(|x| x.into_inner()).pgid as i64 == pgid)
            });
        if !occupied {
            return EPERM;
        }
        self.w.fg_pgid = pgid as i32;
        0
    }

    /// #798 — the **background-terminal doorbell**: a process outside the foreground group
    /// touching the proto-terminal raises `sig` (`SIGTTOU` on write, `SIGTTIN` on read) pending on
    /// ITSELF and arms — the L0 approximation of POSIX's stop-the-background-job (real
    /// stop/continue is #798 slice 2; until then the I/O proceeds and the doorbell tells a
    /// job-control-aware guest what happened).
    fn tty_background_check(&mut self, sig: i32) {
        if self.p.pgid != self.w.fg_pgid {
            // #798 slice 2 — through the delivery gate: default disposition now really STOPS the
            // background job (the doorbell became the POSIX action); ignored proceeds silently
            // (POSIX: an ignored TTOU write goes through); caught pends. The stop fires after
            // this dispatch's locks drop, so the current I/O completes first — close enough to
            // the letter (POSIX stops before the I/O) and honest about it.
            let was_stopped = self.p.stopped_sig.is_some();
            if let Some(f) = self.p.deliver_signal(sig) {
                self.wake_after.push(f);
            }
            // #802 rung 3 — the stop is a child transition: SIGCHLD the parent (POSIX), so an
            // interactive shell's handler (bash's `waitchld`) learns a background job stopped
            // for tty access before the user types `fg` (which only SIGCONTs a job it knows
            // is stopped).
            if !was_stopped && self.p.stopped_sig.is_some() {
                let ppid = self.p.ppid;
                self.notify_parent_chld(ppid);
            }
        }
    }

    /// #802 rung 3 — a child of `ppid` just transitioned (stopped, continued, or exited): raise
    /// `SIGCHLD` in the parent through its delivery gate (POSIX), deferring the fire like any
    /// other delivery ([`Ctx::wake_after`]). A parent without a handler discards it at generation
    /// (`SIGCHLD` is default-ignore); the shell that installs one (interactive bash) gets its
    /// `waitchld` run and keeps its job table live. Self-delivery when the signaler IS the
    /// parent (`fg`'s own `killpg(SIGCONT)`); an absent or dead parent is a no-op.
    fn notify_parent_chld(&mut self, ppid: i32) {
        if ppid == self.p.pid {
            self.p.reap_wake = true; // #1171 — one-shot reap re-check edge for the coop sweep
            if let Some(f) = self.p.deliver_signal(SIGCHLD) {
                self.wake_after.push(f);
            }
            // #1171 — a child's stop/continue transition must wake a parent blocked in
            // `waitpid(WUNTRACED/WCONTINUED)`, independent of whether the parent has async SIGCHLD
            // delivery armed (bash installs no sigaltstack, so its SIGCHLD is poll-only — the async
            // `deliver_signal` wake above returns `None`, and a foreground job stopped on a parked
            // read never enters the core stop-park that would drain the reap waiters). Fire the
            // parent's own domain run-wake so its blocked `waitpid` re-runs and reports the fresh stop.
            if let Some(w) = self.p.wake.clone() {
                self.wake_after.push(w);
            }
            return;
        }
        if let Some(ProcEntry::Live(t)) = self.w.procs.get(&ppid) {
            let mut tp = t.lock().unwrap_or_else(|e| e.into_inner());
            tp.reap_wake = true; // #1171 — one-shot reap re-check edge for the coop sweep
            if let Some(f) = tp.deliver_signal(SIGCHLD) {
                self.wake_after.push(f);
            }
            if let Some(w) = tp.wake.clone() {
                self.wake_after.push(w);
            }
        }
    }

    /// `sigaltstack(sp, size) -> 0` (#796 L2): register the guest's dedicated signal-handler stack (the
    /// data-SP an async handler runs on). `sp == 0` turns async delivery off (poll-only). `size` is
    /// advisory. Registering a stack may make already-pending caught signals deliverable, so re-arm.
    fn sigaltstack(&mut self, args: &[i64]) -> i64 {
        let sp = *args.first().unwrap_or(&0);
        self.p.sig_stack_base = sp.max(0) as u64;
        self.p.arm_signals();
        0
    }

    /// `sigcheck(_) -> handler | 0`: the L0 doorbell poll. Clear and return the handler pointer of the
    /// lowest-numbered pending **caught and unblocked** signal (`handler > SIG_IGN`); pending **ignored**
    /// (`SIG_IGN`) and **default** (`SIG_DFL`) signals are cleared and skipped (L0 does not deliver default
    /// actions). A pending **blocked** signal (`sig_mask`, #796) is **held** — neither delivered nor cleared
    /// — until `sigprocmask` unblocks it. `0` when nothing is deliverable — so the guest runs
    /// `((void(*)(void))handler)()` at its safe point.
    fn sigcheck(&mut self) -> i64 {
        // #798 slice 2 — a stopped process delivers nothing until continued (POSIX; also makes
        // the stop deterministic — see [`SignalDoor::take_deliverable`]).
        if self.p.stopped_sig.is_some() {
            return 0;
        }
        loop {
            let deliverable = self.p.sig_pending & !self.p.sig_mask;
            if deliverable == 0 {
                return 0; // nothing pending, or all pending signals are blocked (held)
            }
            let s = deliverable.trailing_zeros() as i32;
            self.p.sig_pending &= !(1u64 << s);
            let handler = self.p.sig_handler.get(&s).copied().unwrap_or(SIG_DFL);
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
            mem.write_bytes(oldset_ptr, &self.p.sig_mask.to_le_bytes())
                .ok_or(Trap::Malformed)?;
        }
        if set_ptr != 0 {
            let bytes = mem.read_bytes(set_ptr, 8).ok_or(Trap::Malformed)?;
            let set = u64::from_le_bytes(bytes.try_into().map_err(|_| Trap::Malformed)?);
            let new = match how {
                SIG_BLOCK => self.p.sig_mask | set,
                SIG_UNBLOCK => self.p.sig_mask & !set,
                SIG_SETMASK => set,
                _ => return Ok(vec![EINVAL]),
            };
            self.p.sig_mask = new & !UNMASKABLE; // SIGKILL/SIGSTOP can never be blocked
            self.p.arm_signals(); // #796 L2 — unblocking may free a held signal for async delivery
                                  // #796 default actions — unblocking a held fatal signal runs its
                                  // action now (deferred-fire discipline, like every kill).
            if let Some(f) = self.p.dispatch_default_actions() {
                self.wake_after.push(f);
            }
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
        // #796 default actions — POSIX forbids changing SIGKILL/SIGSTOP's action (probing with a
        // null `act` is allowed).
        if act_ptr != 0 && (s == SIGKILL || s == SIGSTOP) {
            return Ok(vec![EINVAL]);
        }
        let mem = mem.ok_or(Trap::Malformed)?;
        if oldact_ptr != 0 {
            let mut buf = [0u8; 24];
            let h = self.p.sig_handler.get(&s).copied().unwrap_or(SIG_DFL);
            let m = self.p.sig_action_mask.get(&s).copied().unwrap_or(0);
            let f = self.p.sig_action_flags.get(&s).copied().unwrap_or(0);
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
            self.p.sig_handler.insert(s, handler);
            self.p.sig_action_mask.insert(s, mask);
            self.p.sig_action_flags.insert(s, flags);
            self.p.arm_signals(); // #796 L2 — a handler for an already-pending signal may deliver now
            if handler == SIG_DFL {
                // #796 default actions — resetting a pending signal's disposition to `SIG_DFL`
                // runs its default action now (deferred-fire discipline).
                if let Some(f) = self.p.dispatch_default_actions() {
                    self.wake_after.push(f);
                }
            }
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
        Ok(vec![if self.w.files.remove(&path).is_some() {
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
        if self.w.files.contains_key(&norm) || self.is_dir(&norm) {
            return Ok(vec![EEXIST]);
        }
        let parent = match norm.rfind('/') {
            Some(0) | None => "/",
            Some(i) => &norm[..i],
        };
        if !self.is_dir(parent) {
            return Ok(vec![ENOENT]);
        }
        self.w.explicit_dirs.insert(norm);
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
        if self.w.files.contains_key(&norm) {
            return Ok(vec![ENOTDIR]);
        }
        if !self.is_dir(&norm) {
            return Ok(vec![ENOENT]);
        }
        if !self.dir_children(&norm).is_empty() {
            return Ok(vec![ENOTEMPTY]);
        }
        self.w.explicit_dirs.remove(&norm);
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
        if let Some(v) = self.w.files.remove(&old_n) {
            self.w.files.insert(new_n.clone(), v);
            self.w.explicit_dirs.remove(&new_n);
            return Ok(vec![0]);
        }
        // Directory: re-key every file and explicit subdir under `old_n`, plus the marker itself.
        if self.is_dir(&old_n) {
            let op = format!("{old_n}/");
            let np = format!("{new_n}/");
            let moved: Vec<String> = self
                .w
                .files
                .keys()
                .filter(|k| k.starts_with(&op))
                .cloned()
                .collect();
            for k in moved {
                let v = self.w.files.remove(&k).unwrap();
                self.w.files.insert(format!("{np}{}", &k[op.len()..]), v);
            }
            let dirs: Vec<String> = self
                .w
                .explicit_dirs
                .iter()
                .filter(|d| d.as_str() == old_n || d.starts_with(&op))
                .cloned()
                .collect();
            for d in dirs {
                self.w.explicit_dirs.remove(&d);
                let nd = if d == old_n {
                    new_n.clone()
                } else {
                    format!("{np}{}", &d[op.len()..])
                };
                self.w.explicit_dirs.insert(nd);
            }
            return Ok(vec![0]);
        }
        Ok(vec![ENOENT])
    }

    // ---- net ops (POSIX.md §5a) — the `net` capability's dispatch targets ----------------------

    /// The next free ephemeral port (49152..), skipping bound listeners; wraps within the range.
    fn net_alloc_ephemeral(&mut self) -> u16 {
        loop {
            let p = self.w.net_next_port;
            self.w.net_next_port = if p == u16::MAX { 49152 } else { p + 1 };
            if !self.w.net_listeners.contains_key(&p) {
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
            let Some(q) = self.w.net_listeners.get(&dst.port).map(Arc::clone) else {
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
        let Some(d) = self.w.net_delegate.as_mut() else {
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
        } else if self.w.net_listeners.contains_key(&req.port) {
            return Ok(vec![EADDRINUSE]);
        } else {
            req.port
        };
        let bound = NetAddr { port, ..req };
        if let Err(e) = Self::net_write_addr(mem, out, cap, &bound)? {
            return Ok(vec![e]);
        }
        let pending = Arc::new(Mutex::new(VecDeque::new()));
        self.w.net_listeners.insert(port, Arc::clone(&pending));
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
            match self.w.net_delegate.as_mut() {
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
            .w
            .files
            .keys()
            .chain(self.w.explicit_dirs.iter())
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
        norm.is_empty()
            || self.w.explicit_dirs.contains(norm)
            || !self.dir_children(path).is_empty()
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
        let (mode, size) = if let Some(f) = self.w.files.get(&path) {
            // #801 — a registered executable carries the exec bits; a plain file does not.
            let perms = if self.w.executables.contains(&path) {
                0o755
            } else {
                0o644
            };
            (S_IFREG | perms, f.len() as i64)
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
        if self.w.files.contains_key(&path) {
            return Ok(vec![ENOTDIR]);
        }
        if !self.is_dir(&path) {
            return Ok(vec![ENOENT]);
        }
        let entries = self.dir_children(&path);
        let stream = DirStream { entries, pos: 0 };
        let idx = match self.p.dirs.iter().position(Option::is_none) {
            Some(i) => {
                self.p.dirs[i] = Some(stream);
                i
            }
            None => {
                self.p.dirs.push(Some(stream));
                self.p.dirs.len() - 1
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
            .and_then(|i| self.p.dirs.get_mut(i)?.as_mut())
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
        if let Some(slot @ Some(_)) = usize::try_from(dir)
            .ok()
            .and_then(|i| self.p.dirs.get_mut(i))
        {
            *slot = None;
            0
        } else {
            EBADF
        }
    }

    /// `argv(i, buf, cap) -> len | -errno`: write argument `i` (NUL-terminated) into the caller's
    /// buffer and return its length (excluding the NUL). An out-of-range index is `-EINVAL`; a name
    /// that won't fit `cap` is `-ERANGE`. (`argc` is a fieldless op: `self.p.args.len()`.)
    fn argv(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let i = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let cap = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as u64;
        let Some(arg) = usize::try_from(i).ok().and_then(|i| self.p.args.get(i)) else {
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
            .w
            .commands
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, h, _)| *h as i64)
            .unwrap_or(-1);
        Ok(vec![h])
    }

    /// #801 — `exec_resolve(path_ptr, path_len) -> module_handle | -errno`: resolve a filesystem
    /// path to its registered executable's pre-granted `Module` handle. A memfs file that is not
    /// a registered executable is `-EACCES` (no exec bit); an absent path is `-ENOENT`; a
    /// non-UTF-8 path is `-EINVAL`. The errno split is what the guest `execvp` PATH walk keys on
    /// (POSIX: remember an EACCES, keep searching on ENOENT).
    fn exec_resolve(
        &mut self,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
    ) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, len).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        if self.w.executables.contains(&path) {
            if let Some((_, h, wl)) = self.w.commands.iter().find(|(n, _, _)| *n == path) {
                // Stash the command's heap re-base for the exec this resolve precedes (consumed
                // by the exec-remap hook on commit; overwritten by any later resolve, so a PATH
                // walk's misses and a refused exec never leave a stale re-base behind).
                let end = 1u64 << (*wl).min(63);
                self.p.pending_exec_heap = Some((end / 4 * 3, end));
                return Ok(vec![*h as i64]);
            }
        }
        if self.w.files.contains_key(&path) {
            return Ok(vec![EACCES]);
        }
        Ok(vec![ENOENT])
    }

    /// `exec_win(module_handle) -> size_log2 | -1`: the declared window of the registered command with
    /// this handle, so the shell carves its spawn to match (§14: carve == declared memory). `-1` when
    /// the handle is not a registered command.
    fn exec_win(&mut self, args: &[i64]) -> Result<Vec<i64>, Trap> {
        let handle = *args.first().ok_or(Trap::Malformed)? as i32;
        let wl = self
            .w
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
        let Some(fifo) = self.w.exec_stdin_fifo.clone() else {
            return Ok(vec![-1]);
        };
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, len).ok_or(Trap::Malformed)?;
        let mut q = fifo.lock().unwrap_or_else(|e| e.into_inner());
        q.clear(); // defensively drop any residue from an aborted prior filter
        q.extend(bytes);
        Ok(vec![self.w.exec_stdin_handle as i64])
    }

    /// Allocate the lowest free fd for `entry`, extending the table if needed.
    fn alloc_fd(&mut self, entry: FdEntry) -> i64 {
        self.alloc_fd_from(entry, 0)
    }

    /// Allocate the lowest free fd `>= min` for `entry` (the `F_DUPFD`/`dup2` "at or above" contract),
    /// extending the table if needed.
    fn alloc_fd_from(&mut self, entry: FdEntry, min: usize) -> i64 {
        while self.p.fds.len() < min {
            self.p.fds.push(None);
        }
        match (min..self.p.fds.len()).find(|&i| self.p.fds[i].is_none()) {
            Some(i) => {
                self.p.fds[i] = Some(entry);
                i as i64
            }
            None => {
                self.p.fds.push(Some(entry));
                (self.p.fds.len() - 1) as i64
            }
        }
    }

    /// Write `data` into fd `fd`'s memfs file at its offset (extending with zeros if the offset is
    /// past the end), advancing the offset. Returns the count, or `-EBADF` for a non-file / read-only fd.
    fn file_write(&mut self, fd: usize, data: &[u8]) -> i64 {
        let desc = match self.p.fds.get(fd).and_then(|s| s.as_ref()) {
            Some(FdEntry::File(of)) => Arc::clone(of),
            _ => return EBADF,
        };
        let mut of = desc.lock().unwrap_or_else(|e| e.into_inner());
        if !of.writable {
            return EBADF;
        }
        let pos = of.pos;
        let file = self.w.files.entry(of.path.clone()).or_default();
        let end = pos + data.len();
        if file.len() < end {
            file.resize(end, 0);
        }
        file[pos..end].copy_from_slice(data);
        of.pos = end;
        data.len() as i64
    }

    /// Read up to `len` bytes from fd `fd`'s memfs file at its offset, advancing it. `Err(-EBADF)` for
    /// a non-file fd.
    fn file_read(&mut self, fd: usize, len: usize) -> Result<Vec<u8>, i64> {
        let desc = match self.p.fds.get(fd).and_then(|s| s.as_ref()) {
            Some(FdEntry::File(of)) => Arc::clone(of),
            _ => return Err(EBADF),
        };
        let mut of = desc.lock().unwrap_or_else(|e| e.into_inner());
        let pos = of.pos;
        let file = self
            .w
            .files
            .get(&of.path)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let n = len.min(file.len().saturating_sub(pos));
        let chunk = file[pos..pos + n].to_vec();
        of.pos = pos + n;
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
        if let Some(i) = self.p.free_list.iter().position(|&(_, sz)| sz >= want) {
            let (off, sz) = self.p.free_list.swap_remove(i);
            if sz > want {
                self.p.free_list.push((off + want, sz - want));
            }
            self.p.allocated.insert(off, want);
            return off as i64;
        }
        // Bump a fresh block from the high-water mark and record it as a live allocation.
        match self.arena_bump(want) {
            Some(ptr) => {
                self.p.allocated.insert(ptr, want);
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
        let ptr = (self.p.heap_next + (ALIGN - 1)) & !(ALIGN - 1);
        match ptr.checked_add(n) {
            Some(end) if end <= self.p.heap_end => {
                self.p.heap_next = end;
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
        let mut bytes = self.p.cwd.clone().into_bytes();
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
        self.p.cwd = path;
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
        if let Some(&cached) = self.p.env_ptrs.get(&name) {
            return Ok(vec![cached as i64]);
        }
        let Some(value) = self.p.env.get(&name).cloned() else {
            return Ok(vec![0]); // unset → NULL
        };
        let mut vb = value.into_bytes();
        vb.push(0); // NUL terminator
        let Some(dst) = self.arena_bump(vb.len() as u64) else {
            return Ok(vec![0]); // no room → behave as if unset (best effort)
        };
        mem.write_bytes(dst, &vb).ok_or(Trap::Malformed)?;
        self.p.env_ptrs.insert(name, dst);
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
        // The 5-arg C ABI passes `overwrite` explicitly; a 4-arg `__vm_host_call` (the temen `std` PAL's
        // `set_var`, which always overwrites) omits it, so default to overwrite when absent.
        let overwrite = *args.get(4).unwrap_or(&1);
        let nb = mem.read_bytes(nptr, nlen).ok_or(Trap::Malformed)?;
        let vb = mem.read_bytes(vptr, vlen).ok_or(Trap::Malformed)?;
        let (Ok(name), Ok(value)) = (String::from_utf8(nb), String::from_utf8(vb)) else {
            return Ok(vec![EINVAL]);
        };
        if overwrite == 0 && self.p.env.contains_key(&name) {
            return Ok(vec![0]); // keep the existing value
        }
        self.p.env_ptrs.remove(&name); // stale cached pointer no longer reflects the value
        self.p.env.insert(name, value);
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
        let Some(value) = self.p.env.get(&name) else {
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
        self.p.env_ptrs.remove(&name);
        self.p.env.remove(&name);
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
        let mut keys: Vec<&String> = self.p.env.keys().collect();
        keys.sort();
        let Some(key) = keys.get(index as usize) else {
            return Ok(vec![-1]); // past the end
        };
        let entry = format!("{key}={}", self.p.env[*key]).into_bytes();
        if buf != 0 && (entry.len() as u64) <= cap {
            mem.write_bytes(buf, &entry).ok_or(Trap::Malformed)?;
        }
        Ok(vec![entry.len() as i64])
    }

    /// `clock(clock_id) -> nanos`: `clock_id == 1` → monotonic (nanos since this personality started),
    /// else realtime (nanos since the Unix epoch). Returns a pinned value when [`Posix::set_clock`] set
    /// one, so a differential run is reproducible; otherwise reads the real host clock.
    fn clock(&self, args: &[i64]) -> i64 {
        if let Some(fixed) = self.w.clock_fixed {
            return fixed;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if *args.first().unwrap_or(&0) == 1 {
                self.w.clock_base.elapsed().as_nanos() as i64
            } else {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0)
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            // No std time source on wasm32 (`Instant::now`/`SystemTime::now` panic): serve the
            // deterministic tick, 1 µs per read, for monotonic and realtime alike.
            let _ = args;
            self.w
                .clock_tick
                .fetch_add(1_000, std::sync::atomic::Ordering::Relaxed)
        }
    }

    /// `free(ptr)`: return `ptr`'s block to the free list for reuse, **coalescing** it with any
    /// adjacent free blocks (the #800 regex differential is the workload that finally demanded it:
    /// 30+ compile/free cycles of varied-size arenas fragmented a quarter-window heap to
    /// exhaustion). `free(NULL)` and a double / bogus free are no-ops (a bogus free never corrupts
    /// the arena — the size table is host-side).
    fn free(&mut self, args: &[i64]) {
        let ptr = *args.first().unwrap_or(&0) as u64;
        if ptr == 0 {
            return;
        }
        if let Some(size) = self.p.allocated.remove(&ptr) {
            let (mut off, mut sz) = (ptr, size);
            // Merge any free neighbor — one ending at `off`, one starting at `off + sz` — repeating
            // until neither side touches (at most two merges; the list never holds two adjacent
            // blocks after this, so adjacency can't chain further).
            while let Some(i) = self
                .p
                .free_list
                .iter()
                .position(|&(o, s)| o + s == off || o == off + sz)
            {
                let (o, s) = self.p.free_list.swap_remove(i);
                off = off.min(o);
                sz += s;
            }
            self.p.free_list.push((off, sz));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use temen_interp::{run_capture_reserved_with_host, Host, Value};
    use temen_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
    use temen_text::parse_module;

    /// Lock the personality's world + root proc (canonical order — see [`World`]) and build the
    /// [`Ctx`] dispatch view a test drives op methods through, as [`handler`] does.
    macro_rules! ctx {
        ($posix:expr, $w:ident, $p:ident, $st:ident) => {
            let mut $w = $posix.world.lock().unwrap();
            let mut $p = $posix.root.lock().unwrap();
            #[allow(unused_mut)]
            let mut $st = Ctx {
                w: &mut $w,
                p: &mut $p,
                wake_after: Vec::new(),
            };
        };
    }
    use temen_verify::verify_module;

    // Heap arena, shifted up by the #1094 NULL guard (16384) so every malloc/getenv-materialized
    // pointer lands above the unconditionally-unmapped `[0, 16384)` region: base 4096 -> 20480,
    // end 65536 -> 81920 (arena size preserved). Still well within the WIN window below.
    const HEAP_BASE: u64 = 20480;
    const HEAP_END: u64 = 80 << 10;
    const WIN: usize = 128 << 10;

    /// #863 slice 1 — the World/Proc fork contract, unit-level. [`Proc::fork`] copies the
    /// per-process side (cwd/env/dispositions copied, **pending cleared**) while fd-table entries
    /// share their open-file **descriptions** — a fork-inherited fd shares its offset with the
    /// parent (POSIX fork-shares-open-file-descriptions), and a child's cwd/env mutations are
    /// invisible to the parent (the pre-split shared blob got both wrong).
    #[test]
    fn fork_clones_the_process_side_and_shares_descriptions() {
        let world = Arc::new(Mutex::new(new_world(Vec::new())));
        let root = Arc::new(Mutex::new(new_proc(HEAP_BASE, HEAP_END)));
        let posix = Posix { world, root };
        posix.write_file("/f", b"hello world");
        let mut win = vec![0u8; 256];
        win[0..2].copy_from_slice(b"/f");
        let mut mem = temen_interp::WindowMem::new(&mut win, 256);

        let mut w = posix.world.lock().unwrap();
        let mut p = posix.root.lock().unwrap();
        let fd = {
            let mut st = Ctx {
                w: &mut w,
                p: &mut p,
                wake_after: Vec::new(),
            };
            let fd = st.open(&[0, 2, 0], Some(&mut mem)).unwrap()[0];
            assert!(fd >= 3, "a fresh fd past the stdio sentinels");
            // Advance the description to offset 5 through the parent.
            assert_eq!(st.read(&[fd, 100, 5], Some(&mut mem)).unwrap()[0], 5);
            fd
        };
        p.cwd = "/parent".to_string();
        p.env.insert("K".to_string(), "v".to_string());
        p.sig_handler.insert(2, 0x1234);
        p.sig_pending = 1 << 2;
        p.sig_mask = 1 << 10;

        let mut twin = p.fork();
        assert_eq!(
            twin.sig_pending, 0,
            "POSIX: a fork twin starts with no pending signals"
        );
        assert_eq!(
            twin.sig_handler.get(&2),
            Some(&0x1234),
            "dispositions are inherited (copied)"
        );
        assert_eq!(
            twin.sig_mask,
            1 << 10,
            "the signal mask is inherited (copied)"
        );

        // cwd/env are copies: the twin's mutations never reach the parent.
        twin.cwd = "/child".to_string();
        twin.env.insert("K".to_string(), "child".to_string());
        assert_eq!(
            p.cwd, "/parent",
            "a subshell's chdir must not move its parent"
        );
        assert_eq!(
            p.env["K"], "v",
            "a child's setenv is invisible to the parent"
        );

        // The twin's inherited fd continues at the PARENT's offset (shared description)…
        {
            let mut tc = Ctx {
                w: &mut w,
                p: &mut twin,
                wake_after: Vec::new(),
            };
            assert_eq!(tc.read(&[fd, 120, 6], Some(&mut mem)).unwrap()[0], 6);
        }
        assert_eq!(
            mem.read_bytes(120, 6).unwrap(),
            b" world",
            "twin resumes at offset 5"
        );
        // …and its read advanced the shared offset for the parent too (now EOF).
        {
            let mut pc = Ctx {
                w: &mut w,
                p: &mut p,
                wake_after: Vec::new(),
            };
            assert_eq!(
                pc.read(&[fd, 130, 8], Some(&mut mem)).unwrap()[0],
                0,
                "the twin's read moved the shared offset to EOF for the parent"
            );
            // The twin's fd TABLE is a copy: closing the parent's fd leaves the twin's open.
            assert_eq!(pc.close(&[fd]), 0);
        }
        {
            let mut tc = Ctx {
                w: &mut w,
                p: &mut twin,
                wake_after: Vec::new(),
            };
            assert_eq!(
                tc.lseek(&[fd, 0, SEEK_SET]),
                0,
                "the twin's fd survives the parent's close (per-process table)"
            );
        }
    }

    /// #863 slice 1 — `dup` shares the open-file description: the dup'd fd continues at the
    /// original's offset (POSIX; the pre-split inline description gave dups independent offsets).
    #[test]
    fn dup_shares_the_open_file_description_offset() {
        let world = Arc::new(Mutex::new(new_world(Vec::new())));
        let root = Arc::new(Mutex::new(new_proc(HEAP_BASE, HEAP_END)));
        let posix = Posix { world, root };
        posix.write_file("/f", b"hello world");
        let mut win = vec![0u8; 256];
        win[0..2].copy_from_slice(b"/f");
        let mut mem = temen_interp::WindowMem::new(&mut win, 256);
        ctx!(posix, w_g, p_g, st);
        let fd = st.open(&[0, 2, 0], Some(&mut mem)).unwrap()[0];
        let dup = st.dup(&[fd]);
        assert!(dup >= 0 && dup != fd);
        assert_eq!(st.read(&[fd, 100, 5], Some(&mut mem)).unwrap()[0], 5);
        assert_eq!(st.read(&[dup, 120, 6], Some(&mut mem)).unwrap()[0], 6);
        assert_eq!(
            mem.read_bytes(120, 6).unwrap(),
            b" world",
            "the dup continues at the original's offset — one shared description"
        );
    }

    /// #863 slice 2 — `kill(pid, sig)` reaches a fork twin's OWN pending set through the process
    /// table, end to end across two processes' handlers: the twin (minted by the fork factory
    /// under pid 7, as the core does at `fork()`) installs a SIGUSR1 handler; the ROOT's `kill(7,
    /// 10)` sets the TWIN's bit (the twin's `sigcheck` delivers it, the root's own stays empty).
    /// Plus the table edges: `kill(unknown)` = `-ESRCH`, `kill(7, 0)` probes, negative = `-EINVAL`.
    #[test]
    fn kill_targets_a_fork_twin_through_the_process_table() {
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut forked = cap_fork_factory(&posix)(7);
        // The twin installs a handler for signal 10 through its own dispatch.
        assert_eq!(
            (forked.handler)(OP_SIGNAL, &[10, 0x77], None, None).unwrap(),
            vec![0]
        );
        {
            ctx!(posix, w_g, p_g, st);
            assert_eq!(st.kill(&[7, 0]), 0, "kill(pid, 0): the twin exists");
            assert_eq!(st.kill(&[99, 10]), ESRCH, "unknown pid");
            assert_eq!(
                st.kill(&[-5, 10]),
                ESRCH,
                "kill(-pgid) of an empty group (#798)"
            );
            assert_eq!(st.kill(&[7, 10]), 0, "signal the twin by pid");
            assert_eq!(
                st.p.sig_pending, 0,
                "the ROOT's pending set stays empty — the signal went to the twin"
            );
        }
        assert_eq!(
            (forked.handler)(OP_SIGCHECK, &[], None, None).unwrap(),
            vec![0x77],
            "the twin's own sigcheck delivers the pid-targeted signal"
        );
    }

    /// #863 slice 2 — `getpid`: the root is pid 1, a fork twin reports the pid it was minted under
    /// (its scheduler `TaskId` — the same value the parent's `fork()` returned), and a twin can
    /// signal ITSELF by that pid (the self short-circuit, no table double-lock).
    #[test]
    fn getpid_reports_the_table_pid_on_both_sides_of_a_fork() {
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut root_handler = handler(Arc::clone(&posix.world), Arc::clone(&posix.root));
        assert_eq!(
            root_handler(OP_GETPID, &[], None, None).unwrap(),
            vec![1],
            "the root is pid 1"
        );
        let mut forked = cap_fork_factory(&posix)(9);
        assert_eq!(
            (forked.handler)(OP_GETPID, &[], None, None).unwrap(),
            vec![9],
            "a twin's getpid is the pid fork() returned to its parent"
        );
        // Self-kill by own pid: handler + raise + sigcheck, all through the twin's dispatch.
        assert_eq!(
            (forked.handler)(OP_SIGNAL, &[12, 0x99], None, None).unwrap(),
            vec![0]
        );
        assert_eq!(
            (forked.handler)(OP_KILL, &[9, 12], None, None).unwrap(),
            vec![0],
            "kill(own pid) short-circuits to self — no deadlock on the table entry"
        );
        assert_eq!(
            (forked.handler)(OP_SIGCHECK, &[], None, None).unwrap(),
            vec![0x99]
        );
    }

    /// #863 slice 2 — ONE pid space: spawn-delegate zombies live in the same process table as fork
    /// twins. A spawn's pid allocation skips a pid a twin already occupies; `kill` on the zombie
    /// succeeds (exists-until-reaped, signal dropped); `waitpid` reaps it out of the table, after
    /// which the pid is `-ESRCH`.
    #[test]
    fn spawn_zombies_share_the_process_table_with_fork_twins() {
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        posix.set_spawn(|_n, _a, _stdin| SpawnResult {
            stdout: Vec::new(),
            stderr: Vec::new(),
            status: 7,
        });
        // A twin squatting on pid 1000 — exactly where the spawn allocator starts.
        let _forked = cap_fork_factory(&posix)(1000);
        let mut win = vec![0u8; 256];
        win[0..4].copy_from_slice(b"prog");
        let mut mem = temen_interp::WindowMem::new(&mut win, 256);
        ctx!(posix, w_g, p_g, st);
        let pid = st.spawn(&[0, 4, 0, 0], Some(&mut mem)).unwrap()[0];
        assert_eq!(pid, 1001, "the allocator skips the twin's occupied pid");
        assert_eq!(st.kill(&[pid, 15]), 0, "a zombie exists until reaped");
        assert_eq!(
            st.waitpid(&[pid, 0, 0], Some(&mut mem)).unwrap()[0],
            pid,
            "waitpid reaps the zombie from the table"
        );
        assert_eq!(st.kill(&[pid, 15]), ESRCH, "reaped ⇒ the pid is gone");
        // The live twin at 1000 is NOT reapable through this op (core reap owns fork twins).
        assert_eq!(
            st.waitpid(&[1000, 0, 0], Some(&mut mem)).unwrap()[0],
            ECHILD
        );
    }

    /// #863 slice 2 — a deliverable pid-targeted `kill` pokes the TARGET's run wake, and only
    /// **after** the dispatch's locks are released (the deferred [`Ctx::wake_after`] — firing under
    /// the world lock would invert the fork factory's scheduler → world order). The twin has a
    /// caught handler + a signal stack (async delivery on) and a wake installed through its door;
    /// the root's `kill(7, 10)` through the ROOT handler must fire it. A masked signal must not.
    #[test]
    fn kill_wakes_the_targets_run_after_unlock() {
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut root_handler = handler(Arc::clone(&posix.world), Arc::clone(&posix.root));
        let mut forked = cap_fork_factory(&posix)(7);
        // Deliverability on the twin: caught handler + registered signal stack.
        assert_eq!(
            (forked.handler)(OP_SIGNAL, &[10, 0x77], None, None).unwrap(),
            vec![0]
        );
        assert_eq!(
            (forked.handler)(OP_SIGALTSTACK, &[0x500, 0x100], None, None).unwrap(),
            vec![0]
        );
        let (door, _armed) = forked.signal.as_ref().expect("the twin has its own door");
        let fired = Arc::new(AtomicBool::new(false));
        let f2 = Arc::clone(&fired);
        door.set_wake(Arc::new(move || {
            f2.store(true, std::sync::atomic::Ordering::SeqCst);
        }));
        // Signal 11 has no handler on the twin: pending is set but nothing is deliverable — no wake
        // (the fire is async — a detached thread — so give a false positive a moment to appear).
        assert_eq!(
            root_handler(OP_KILL, &[7, 11], None, None).unwrap(),
            vec![0]
        );
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert!(
            !fired.load(std::sync::atomic::Ordering::SeqCst),
            "an undeliverable signal never wakes the target's run"
        );
        // Signal 10 is caught + unmasked + stack registered: the wake fires — from a detached
        // thread, after the dispatch's locks dropped (slice 3), so poll for it.
        assert_eq!(
            root_handler(OP_KILL, &[7, 10], None, None).unwrap(),
            vec![0]
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !fired.load(std::sync::atomic::Ordering::SeqCst) {
            assert!(
                std::time::Instant::now() < deadline,
                "a deliverable pid-targeted kill pokes the target's scheduler wake"
            );
            std::thread::yield_now();
        }
    }

    /// #863 hygiene — a fork twin's **exit hook** retires it in the process table: firing it (as
    /// the core does at twin completion) flips `Live` → `Zombie` with the wait-encoded status, so
    /// `waitpid` reaps a fork twin exactly like a spawn child — and the pid is gone afterwards
    /// (`kill` = `-ESRCH`). Before the exit, the live twin is `-ECHILD` here (non-blocking poll).
    #[test]
    fn a_twins_exit_hook_makes_it_reapable_by_waitpid() {
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let forked = cap_fork_factory(&posix)(7);
        let mut win = vec![0u8; 256];
        let mut mem = temen_interp::WindowMem::new(&mut win, 256);
        {
            ctx!(posix, w_g, p_g, st);
            assert_eq!(
                st.waitpid(&[7, 0, 0], Some(&mut mem)).unwrap()[0],
                ECHILD,
                "a still-running twin is not reapable through this op"
            );
        }
        // The core fires the hook with the twin's raw exit status at completion.
        (forked
            .exit
            .as_ref()
            .expect("the personality installs an exit hook"))(7);
        ctx!(posix, w_g, p_g, st);
        assert_eq!(
            st.kill(&[7, 15]),
            0,
            "an exited-unreaped twin is a zombie — kill succeeds"
        );
        assert_eq!(
            st.waitpid(&[7, 100, 0], Some(&mut mem)).unwrap()[0],
            7,
            "waitpid reaps the exited fork twin"
        );
        let status = i32::from_le_bytes(mem.read_bytes(100, 4).unwrap().try_into().unwrap());
        assert_eq!(
            (status >> 8) & 0xff,
            7,
            "WEXITSTATUS is the twin's exit code"
        );
        assert_eq!(st.kill(&[7, 15]), ESRCH, "reaped ⇒ the pid is gone");
    }

    /// #798 slice 1 — the process-group surface over the table: `setpgid`/`getpgid` roundtrip
    /// (self and a table-routed twin), a fork twin **inherits** its parent's group, and
    /// `kill(-pgid)` sweeps exactly the group — every member's own pending set rings (the caller
    /// included), the non-member's stays silent, and an empty group is `-ESRCH`.
    #[test]
    fn process_groups_route_through_the_table_and_group_kill_sweeps_them() {
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        // Twins 7 and 8; 8 forks 9 AFTER being moved into group 5 — 9 must inherit 5.
        let mut t7 = cap_fork_factory(&posix)(7);
        let f8 = cap_fork_factory(&posix)(8);
        {
            ctx!(posix, w_g, p_g, st);
            assert_eq!(st.getpgid(&[0]), 1, "the root leads group 1");
            assert_eq!(st.getpgid(&[7]), 1, "a twin inherits the root's group");
            assert_eq!(st.setpgid(&[7, 5]), 0, "move twin 7 into group 5 by pid");
            assert_eq!(st.setpgid(&[8, 5]), 0, "move twin 8 into group 5 by pid");
            assert_eq!(st.getpgid(&[7]), 5);
            assert_eq!(st.setpgid(&[99, 5]), ESRCH, "unknown pid");
        }
        // Twin 8 forks twin 9: the child inherits group 5 (POSIX), not the root's 1.
        let _t9 = (f8.refork.expect("self-replicating factory"))(9);
        {
            ctx!(posix, w_g, p_g, st);
            assert_eq!(
                st.getpgid(&[9]),
                5,
                "a fork twin inherits its parent's group"
            );
            // Group kill: twins 7 and 9 catch signal 10; the root (group 1) must stay silent.
            // (Install handlers through the table procs directly — unit-level.)
        }
        assert_eq!(
            (t7.handler)(OP_SIGNAL, &[10, 0x71], None, None).unwrap(),
            vec![0]
        );
        {
            ctx!(posix, w_g, p_g, st);
            assert_eq!(st.kill(&[-5, 10]), 0, "sweep group 5");
            assert_eq!(
                st.p.sig_pending, 0,
                "the root is outside group 5 — its pending set stays empty"
            );
            assert_eq!(st.kill(&[-77, 10]), ESRCH, "an empty group");
        }
        assert_eq!(
            (t7.handler)(OP_SIGCHECK, &[], None, None).unwrap(),
            vec![0x71],
            "group member 7's own doorbell rang"
        );
    }

    /// #798 slice 1 — the proto-terminal: `tcgetpgrp`/`tcsetpgrp` on stdio fds (`-ENOTTY` on a
    /// file), foreground validation (`-EINVAL` non-positive, `-EPERM` for an unoccupied group),
    /// and the background doorbells — a background write rings `SIGTTOU`, a background read rings
    /// `SIGTTIN` (both proceed — the L0 approximation), and a foreground process rings nothing.
    #[test]
    fn the_proto_terminal_foreground_group_gates_with_ttou_ttin_doorbells() {
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, b"input".to_vec());
        let _twin = cap_fork_factory(&posix)(7); // occupies group 1 alongside the root
        let mut win = vec![0u8; 256];
        win[0..2].copy_from_slice(b"/f");
        let mut mem = temen_interp::WindowMem::new(&mut win, 256);
        ctx!(posix, w_g, p_g, st);
        assert_eq!(
            st.tcgetpgrp(&[1]),
            1,
            "foreground starts as the root's group"
        );
        assert_eq!(
            st.tcgetpgrp(&[0]),
            1,
            "any stdio fd names the one proto-terminal"
        );
        let file_fd = st.open(&[0, 2, 65], Some(&mut mem)).unwrap()[0]; // O_CREATE|O_WRITE
        assert_eq!(
            st.tcgetpgrp(&[file_fd]),
            ENOTTY,
            "a file is not the terminal"
        );
        assert_eq!(st.tcsetpgrp(&[1, 0]), EINVAL, "a non-positive pgid");
        assert_eq!(st.tcsetpgrp(&[1, 42]), EPERM, "an unoccupied group");
        // Foreground I/O rings nothing.
        assert_eq!(st.write(&[1, 0, 2], Some(&mut mem)).unwrap()[0], 2);
        assert_eq!(st.read(&[0, 100, 2], Some(&mut mem)).unwrap()[0], 2);
        assert_eq!(st.p.sig_pending, 0, "foreground terminal I/O is silent");
        // Move the root into its own group 9 and foreground the twin's group 1: root backgrounded.
        // CATCH the job-control signals first — a caught TTOU/TTIN takes the pending path (the
        // doorbell); the DEFAULT disposition now stops (#798 slice 2, asserted further down).
        assert_eq!(st.signal(&[SIGTTOU as i64, 0x70]), 0);
        assert_eq!(st.signal(&[SIGTTIN as i64, 0x71]), 0);
        assert_eq!(st.setpgid(&[0, 9]), 0);
        assert_eq!(
            st.tcgetpgrp(&[1]),
            1,
            "foreground group unchanged by setpgid"
        );
        assert_eq!(
            st.write(&[1, 0, 2], Some(&mut mem)).unwrap()[0],
            2,
            "the write proceeds"
        );
        assert_ne!(
            st.p.sig_pending & (1 << SIGTTOU),
            0,
            "a background terminal write rings caught SIGTTOU"
        );
        assert_eq!(
            st.read(&[0, 100, 2], Some(&mut mem)).unwrap()[0],
            2,
            "the read proceeds"
        );
        assert_ne!(
            st.p.sig_pending & (1 << SIGTTIN),
            0,
            "a background terminal read rings caught SIGTTIN"
        );
        // #798 slice 2 — back to the DEFAULT disposition: a background write now records a real
        // stop (bookkeeping-only here — a unit Ctx has no core stop closure installed).
        assert_eq!(st.signal(&[SIGTTOU as i64, SIG_DFL]), 0x70);
        st.p.sig_pending = 0;
        assert_eq!(st.write(&[1, 0, 2], Some(&mut mem)).unwrap()[0], 2);
        assert_eq!(
            st.p.stopped_sig,
            Some(SIGTTOU),
            "default-disposition TTOU stops the background writer"
        );
        assert!(st.p.stop_fresh, "the stop is fresh for WUNTRACED");
        assert_eq!(
            st.p.sig_pending, 0,
            "a default-action stop is not a pending bit"
        );
        // SIGCONT through the gate clears the stop and marks the continue.
        assert!(
            st.p.deliver_signal(SIGCONT).is_none(),
            "no closure to fire in a unit Ctx"
        );
        assert_eq!(st.p.stopped_sig, None, "continued");
        assert!(st.p.cont_fresh, "the continue is fresh for WCONTINUED");
        // The root takes the terminal back for its own group — foreground again, silence again.
        assert_eq!(
            st.tcsetpgrp(&[1, 9]),
            0,
            "the caller's own group is occupied"
        );
        st.p.sig_pending = 0;
        assert_eq!(st.write(&[1, 0, 2], Some(&mut mem)).unwrap()[0], 2);
        assert_eq!(st.p.sig_pending, 0, "foreground once more — no doorbell");
        // A write to the FILE from the background never rings (not the terminal).
        assert_eq!(st.setpgid(&[0, 1]), 0); // background again (fg is 9... self now group 1)
        assert_eq!(
            st.tcsetpgrp(&[1, 9]),
            EPERM,
            "group 9 is now unoccupied — EPERM"
        );
        assert_eq!(st.write(&[file_fd, 0, 2], Some(&mut mem)).unwrap()[0], 2);
        assert_eq!(
            st.p.sig_pending & (1 << SIGTTOU),
            0,
            "file I/O is not terminal I/O — no TTOU"
        );
    }

    /// #798 slice 2 — stop/continue through the delivery gate, end to end at the unit level: a
    /// #796 default actions — the delivery gate's terminate side, driven through a mock kill door:
    /// an unhandled `SIGTERM` fires the twin's **kill closure** (deferred like every wake) and the
    /// exit hook retires the zombie in the `WIFSIGNALED` shape (`waitpid` status = the signal);
    /// `SIGCHLD` (default-ignore) and a `SIG_IGN`'d signal are discarded at generation — a later
    /// reset to `SIG_DFL` finds nothing pending; a **masked** `SIGTERM` is held and runs its action
    /// at `sigprocmask`-unblock; a **stopped** process holds a fatal signal until `SIGCONT` (whose
    /// fire is then the death, not the continue) while `SIGKILL` cuts straight through a stop; and
    /// `signal`/`sigaction` refuse to change `SIGKILL`/`SIGSTOP`'s action (`-EINVAL`).
    #[test]
    fn default_terminate_fires_the_kill_door_and_waitpid_reports_the_signal() {
        use std::sync::atomic::{AtomicI32, Ordering};
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut root_handler = handler(Arc::clone(&posix.world), Arc::clone(&posix.root));
        let mut forked = cap_fork_factory(&posix)(7);
        let kills = Arc::new(AtomicI32::new(0));
        let k2 = Arc::clone(&kills);
        let (door, _armed) = forked.signal.as_ref().expect("the twin has a door");
        door.set_kill(Arc::new(move || {
            k2.fetch_add(1, Ordering::SeqCst);
        }));
        let mut win = vec![0u8; 256];
        let mut mem = temen_interp::WindowMem::new(&mut win, 256);

        // SIGCHLD (17): default-ignore — discarded, no fire, nothing pends.
        assert_eq!(
            root_handler(OP_KILL, &[7, 17], None, None).unwrap(),
            vec![0]
        );
        // SIG_IGN'd SIGUSR2 (12): discarded at generation — a reset to SIG_DFL finds nothing.
        assert_eq!(
            (forked.handler)(OP_SIGNAL, &[12, SIG_IGN], None, None).unwrap(),
            vec![SIG_DFL]
        );
        assert_eq!(
            root_handler(OP_KILL, &[7, 12], None, None).unwrap(),
            vec![0]
        );
        assert_eq!(
            (forked.handler)(OP_SIGNAL, &[12, SIG_DFL], None, None).unwrap(),
            vec![SIG_IGN]
        );
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(
            kills.load(Ordering::SeqCst),
            0,
            "ignored signals never kill"
        );

        // Masked SIGTERM (15): held pending; the unblock runs the default action.
        mem.write_bytes(64, &(1u64 << 15).to_le_bytes()).unwrap();
        assert_eq!(
            (forked.handler)(OP_SIGPROCMASK, &[SIG_BLOCK, 64, 0], Some(&mut mem), None).unwrap(),
            vec![0]
        );
        assert_eq!(
            root_handler(OP_KILL, &[7, 15], None, None).unwrap(),
            vec![0]
        );
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(
            kills.load(Ordering::SeqCst),
            0,
            "a masked fatal signal is held"
        );
        assert_eq!(
            (forked.handler)(OP_SIGPROCMASK, &[SIG_UNBLOCK, 64, 0], Some(&mut mem), None).unwrap(),
            vec![0]
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while kills.load(Ordering::SeqCst) != 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the unblock runs the default action (kill fire)"
            );
            std::thread::yield_now();
        }
        // The core's task completion now fires the exit hook (crash status 128 — what a
        // `ThreadFault` self-termination produces); the personality's bookkeeping wins.
        (forked.exit.take().expect("the twin has an exit hook"))(128);
        let r = root_handler(OP_WAITPID, &[7, 100, 0], Some(&mut mem), None).unwrap();
        assert_eq!(r, vec![7], "the killed twin is reapable");
        let status = i32::from_le_bytes(mem.read_bytes(100, 4).unwrap().try_into().unwrap());
        assert_eq!(status & 0x7f, 15, "WIFSIGNALED: the terminating signal");
        assert_eq!((status >> 8) & 0xff, 0, "not an exit-code encode");

        // A second twin: a stop holds SIGTERM; SIGCONT's fire is the death, not the continue.
        let forked2 = cap_fork_factory(&posix)(8);
        let kills2 = Arc::new(AtomicI32::new(0));
        let stops2 = Arc::new(AtomicI32::new(0));
        let (door2, _armed2) = forked2.signal.as_ref().expect("door");
        let k = Arc::clone(&kills2);
        door2.set_kill(Arc::new(move || {
            k.fetch_add(1, Ordering::SeqCst);
        }));
        let s = Arc::clone(&stops2);
        door2.set_stop(Arc::new(move |stopped| {
            s.store(if stopped { 1 } else { -1 }, Ordering::SeqCst);
        }));
        assert_eq!(
            root_handler(OP_KILL, &[8, SIGSTOP as i64], None, None).unwrap(),
            vec![0]
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while stops2.load(Ordering::SeqCst) != 1 {
            assert!(std::time::Instant::now() < deadline, "the stop fires");
            std::thread::yield_now();
        }
        assert_eq!(
            root_handler(OP_KILL, &[8, 15], None, None).unwrap(),
            vec![0]
        );
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(
            kills2.load(Ordering::SeqCst),
            0,
            "a stopped process holds a fatal signal"
        );
        assert_eq!(
            root_handler(OP_KILL, &[8, SIGCONT as i64], None, None).unwrap(),
            vec![0]
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while kills2.load(Ordering::SeqCst) != 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the continue runs the held default action"
            );
            std::thread::yield_now();
        }
        assert_eq!(
            stops2.load(Ordering::SeqCst),
            1,
            "the kill fire subsumed the continue fire"
        );

        // A third twin: SIGKILL cuts straight through a stop.
        let mut forked3 = cap_fork_factory(&posix)(9);
        let kills3 = Arc::new(AtomicI32::new(0));
        let (door3, _armed3) = forked3.signal.as_ref().expect("door");
        let k = Arc::clone(&kills3);
        door3.set_kill(Arc::new(move || {
            k.fetch_add(1, Ordering::SeqCst);
        }));
        assert_eq!(
            root_handler(OP_KILL, &[9, SIGSTOP as i64], None, None).unwrap(),
            vec![0]
        );
        assert_eq!(
            root_handler(OP_KILL, &[9, SIGKILL as i64], None, None).unwrap(),
            vec![0]
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while kills3.load(Ordering::SeqCst) != 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "SIGKILL kills a stopped process"
            );
            std::thread::yield_now();
        }

        // SIGKILL/SIGSTOP dispositions are immutable.
        assert_eq!(
            (forked3.handler)(OP_SIGNAL, &[SIGKILL as i64, 0x77], None, None).unwrap(),
            vec![EINVAL]
        );
        mem.write_bytes(128, &[0u8; 24]).unwrap();
        assert_eq!(
            (forked3.handler)(
                OP_SIGACTION,
                &[SIGSTOP as i64, 128, 0],
                Some(&mut mem),
                None
            )
            .unwrap(),
            vec![EINVAL]
        );
        assert_eq!(
            (forked3.handler)(
                OP_SIGACTION,
                &[SIGKILL as i64, 0, 128],
                Some(&mut mem),
                None
            )
            .unwrap(),
            vec![0],
            "probing with a null act stays allowed"
        );
    }

    /// default-disposition `SIGTSTP` at a twin fires its **stop closure** (`f(true)`, deferred
    /// past the dispatch's locks like a wake) and `waitpid(-1, WUNTRACED)` reports the stop once
    /// (`sig<<8 | 0x7f`); while stopped, further signals are held (`sigcheck` empty, no wake);
    /// `SIGCONT` fires `f(false)`, `waitpid(-1, WCONTINUED)` reports once (`0xffff`), and the held
    /// signal delivers after the continue. A CAUGHT `SIGTSTP` never stops (ordinary pending).
    #[test]
    fn sigtstp_stops_fire_the_domain_closure_and_waitpid_reports_them() {
        use std::sync::atomic::{AtomicI32, Ordering};
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut root_handler = handler(Arc::clone(&posix.world), Arc::clone(&posix.root));
        let mut forked = cap_fork_factory(&posix)(7);
        // A mock core stop closure on the twin's door: records the last direction.
        let dir = Arc::new(AtomicI32::new(0));
        let d2 = Arc::clone(&dir);
        let (door, _armed) = forked.signal.as_ref().expect("the twin has a door");
        door.set_stop(Arc::new(move |stopped| {
            d2.store(if stopped { 1 } else { -1 }, Ordering::SeqCst);
        }));
        let mut win = vec![0u8; 256];
        let mut mem = temen_interp::WindowMem::new(&mut win, 256);

        // Default-disposition SIGTSTP: the twin stops; the closure fires (detached thread — poll).
        assert_eq!(
            root_handler(OP_KILL, &[7, SIGTSTP as i64], None, None).unwrap(),
            vec![0]
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while dir.load(Ordering::SeqCst) != 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the stop closure fires f(true)"
            );
            std::thread::yield_now();
        }
        // WUNTRACED reports the stop, once.
        let r = root_handler(OP_WAITPID, &[-1, 100, WUNTRACED], Some(&mut mem), None).unwrap();
        assert_eq!(
            r,
            vec![7],
            "waitpid(-1, WUNTRACED) reports the stopped twin"
        );
        let status = i32::from_le_bytes(mem.read_bytes(100, 4).unwrap().try_into().unwrap());
        assert_eq!(status & 0xff, 0x7f, "the stopped marker");
        assert_eq!((status >> 8) & 0xff, SIGTSTP, "the stopping signal");
        assert_eq!(
            root_handler(OP_WAITPID, &[-1, 100, WUNTRACED], Some(&mut mem), None).unwrap(),
            vec![ECHILD],
            "report-once"
        );
        // While stopped: an ordinary signal is HELD — the twin's sigcheck stays empty.
        assert_eq!(
            (forked.handler)(OP_SIGNAL, &[10, 0x77], None, None).unwrap(),
            vec![0]
        );
        assert_eq!(
            root_handler(OP_KILL, &[7, 10], None, None).unwrap(),
            vec![0]
        );
        assert_eq!(
            (forked.handler)(OP_SIGCHECK, &[], None, None).unwrap(),
            vec![0],
            "a stopped process delivers nothing"
        );
        // SIGCONT: the closure fires f(false); WCONTINUED reports once; the held 10 now delivers.
        assert_eq!(
            root_handler(OP_KILL, &[7, SIGCONT as i64], None, None).unwrap(),
            vec![0]
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while dir.load(Ordering::SeqCst) != -1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the continue fires f(false)"
            );
            std::thread::yield_now();
        }
        let r = root_handler(OP_WAITPID, &[-1, 100, WCONTINUED], Some(&mut mem), None).unwrap();
        assert_eq!(r, vec![7], "waitpid(-1, WCONTINUED) reports the continue");
        let status = i32::from_le_bytes(mem.read_bytes(100, 4).unwrap().try_into().unwrap());
        assert_eq!(status, 0xffff, "the continued status word");
        assert_eq!(
            root_handler(OP_WAITPID, &[-1, 100, WCONTINUED], Some(&mut mem), None).unwrap(),
            vec![ECHILD],
            "report-once"
        );
        assert_eq!(
            (forked.handler)(OP_SIGCHECK, &[], None, None).unwrap(),
            vec![0x77],
            "the held signal delivers after the continue"
        );
        // A CAUGHT SIGTSTP never stops: ordinary pending.
        assert_eq!(
            (forked.handler)(OP_SIGNAL, &[SIGTSTP as i64, 0x99], None, None).unwrap(),
            vec![0]
        );
        dir.store(0, Ordering::SeqCst);
        assert_eq!(
            root_handler(OP_KILL, &[7, SIGTSTP as i64], None, None).unwrap(),
            vec![0]
        );
        assert_eq!(
            (forked.handler)(OP_SIGCHECK, &[], None, None).unwrap(),
            vec![0x99],
            "caught TSTP takes the pending path"
        );
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(
            dir.load(Ordering::SeqCst),
            0,
            "no stop fired for a caught TSTP"
        );
    }

    /// func 0 `(host_proc_handle) -> i64`: `malloc(2)`, store `"hi"` into the returned buffer,
    /// `write(1, ptr, 2)`, then encode `write_result * 1_000_000 + ptr`. `malloc` hands out the aligned
    /// heap base (`20480`, above the #1094 NULL guard), `write` returns `2`, so the result is
    /// `2_020480` — and stdout is `"hi"`.
    const MALLOC_WRITE: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vsz = i64.const 2\n\
  vptr = call.cap 13 2 (i64) -> (i64) vph (vsz)\n\
  vh = i32.const 104\n\
  i32.store8 vptr vh\n\
  vone = i64.const 1\n\
  vp1 = i64.add vptr vone\n\
  vi = i32.const 105\n\
  i32.store8 vp1 vi\n\
  vfd = i64.const 1\n\
  vn = call.cap 13 0 (i64, i64, i64) -> (i64) vph (vfd, vptr, vsz)\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vn vk\n\
  vr = i64.add vt vptr\n\
  return vr\n\
  }\n\
}\n";

    fn run_interp(src: &str, stdin: &[u8]) -> (Result<Vec<Value>, temen_interp::Trap>, Vec<u8>) {
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
            temen_run::cap_thunk,
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
  vbuf = call.cap 13 2 (i64) -> (i64) vph (veight)\n\
  vfd0 = i64.const 0\n\
  vn = call.cap 13 1 (i64, i64, i64) -> (i64) vph (vfd0, vbuf, veight)\n\
  vfd1 = i64.const 1\n\
  vw = call.cap 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vbuf, vn)\n\
  return vn\n\
  }\n\
}\n";

    /// func 0 `(handle) -> i64`: `exit(42)` — never returns.
    const EXIT_42: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vc = i64.const 42\n\
  vx = call.cap 13 4 (i64) -> (i64) vph (vc)\n\
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
  va = call.cap 13 2 (i64) -> (i64) vph (vsz)\n\
  vb = call.cap 13 2 (i64) -> (i64) vph (vsz)\n\
  vf = call.cap 13 3 (i64) -> (i64) vph (va)\n\
  vc = call.cap 13 2 (i64) -> (i64) vph (vsz)\n\
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
        assert_eq!(ir, Err(temen_interp::Trap::Exit(42)), "interp: exit(42)");
        assert!(
            matches!(jo, JitOutcome::Exited(42)),
            "jit: exit(42) must terminate the domain, got {jo:?}"
        );
    }

    /// #800 — `free` coalesces adjacent blocks: three 1 KiB neighbors freed out of order merge back
    /// into one span that serves a 3 KiB request no single freed block could. (The regex
    /// differential's 30+ varied-size compile/free cycles exhausted a quarter-window heap without
    /// this.)
    #[test]
    fn free_coalesces_adjacent_blocks() {
        let mut w = new_world(Vec::new());
        let mut p = new_proc(4096, 8192);
        let mut cx = Ctx {
            w: &mut w,
            p: &mut p,
            wake_after: Vec::new(),
        };
        let a = cx.malloc(&[1024]);
        let b = cx.malloc(&[1024]);
        let c = cx.malloc(&[1024]);
        assert!(a > 0 && b > 0 && c > 0, "three live blocks");
        assert_eq!(cx.malloc(&[3072]), 0, "no room while all three are live");
        cx.free(&[a]);
        cx.free(&[c]);
        cx.free(&[b]);
        let big = cx.malloc(&[3072]);
        assert_eq!(
            big, a,
            "freed neighbors merged into one span serving a request bigger than any single block"
        );
    }

    #[test]
    fn malloc_store_write_matches_across_backends() {
        let (ir, iout) = run_interp(MALLOC_WRITE, b"");
        let (jo, jout) = run_jit(MALLOC_WRITE, b"");
        // Interpreter reference: malloc → 20480, write → 2, so 2*1_000_000 + 20480 = 2_020480; "hi" out.
        assert_eq!(
            ir,
            Ok(vec![Value::I64(2_020_480)]),
            "interp: malloc+write result"
        );
        assert_eq!(
            iout, b"hi",
            "interp: bytes written to stdout via the personality"
        );
        // JIT parity: the HostProc dispatches through the same Host path, so identical result + output.
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2_020_480]),
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
  vpath = i64.const 16384\n\
  vfch = i32.const 102\n\
  i32.store8 vpath vfch\n\
  vplen = i64.const 1\n\
  vflags = i64.const 66\n\
  vfd = call.cap 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
  a16 = i64.const 16400\n\
  cH = i32.const 72\n\
  i32.store8 a16 cH\n\
  a17 = i64.const 16401\n\
  ci = i32.const 105\n\
  i32.store8 a17 ci\n\
  a18 = i64.const 16402\n\
  cbang = i32.const 33\n\
  i32.store8 a18 cbang\n\
  vwlen = i64.const 3\n\
  vw = call.cap 13 0 (i64, i64, i64) -> (i64) vph (vfd, a16, vwlen)\n\
  vzero = i64.const 0\n\
  vsk = call.cap 13 7 (i64, i64, i64) -> (i64) vph (vfd, vzero, vzero)\n\
  a32 = i64.const 16416\n\
  veight = i64.const 8\n\
  vr = call.cap 13 1 (i64, i64, i64) -> (i64) vph (vfd, a32, veight)\n\
  vfd1 = i64.const 1\n\
  vso = call.cap 13 0 (i64, i64, i64) -> (i64) vph (vfd1, a32, vr)\n\
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
            temen_run::cap_thunk,
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
  vpath = i64.const 16384\n\
  vg = i32.const 103\n\
  i32.store8 vpath vg\n\
  vplen = i64.const 1\n\
  vu = call.cap 13 8 (i64, i64) -> (i64) vph (vpath, vplen)\n\
  vflags = i64.const 0\n\
  vo = call.cap 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
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
            temen_run::cap_thunk,
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
  vp = i64.const 16484\n\
  vf = i32.const 102\n\
  i32.store8 vp vf\n\
  vlen = i64.const 1\n\
  vflags = i64.const 577\n\
  vfd = call.cap 13 5 (i64, i64, i64) -> (i64) vph (vp, vlen, vflags)\n\
  vone = i64.const 1\n\
  vd = call.cap 13 24 (i64, i64) -> (i64) vph (vfd, vone)\n\
  vsz = i64.const 2\n\
  vbuf = call.cap 13 2 (i64) -> (i64) vph (vsz)\n\
  vY = i32.const 89\n\
  i32.store8 vbuf vY\n\
  vbuf1 = i64.add vbuf vone\n\
  voo = i32.const 111\n\
  i32.store8 vbuf1 voo\n\
  vn = call.cap 13 0 (i64, i64, i64) -> (i64) vph (vone, vbuf, vsz)\n\
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
            temen_run::cap_thunk,
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
        ctx!(posix, w_g, p_g, st);
        let mut win = vec![0u8; WIN];
        win[16..21].copy_from_slice(b"hello");
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);

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
            ctx!(posix, w_g, p_g, st);
            let mut win = vec![0u8; WIN];
            win[0..2].copy_from_slice(b"up");
            let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
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
            ctx!(posix, w_g, p_g, st);
            let mut win = vec![0u8; WIN];
            win[0..2].copy_from_slice(b"up"); // name
            win[8..13].copy_from_slice(b"up\0-n"); // argv blob: ["up", "-n"]
            let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
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
  vp0 = i64.const 16494\n\
  voo = i32.const 111\n\
  i32.store8 vp0 voo\n\
  vp1 = i64.const 16495\n\
  vu2 = i32.const 117\n\
  i32.store8 vp1 vu2\n\
  vp2 = i64.const 16496\n\
  vtt = i32.const 116\n\
  i32.store8 vp2 vtt\n\
  vn0 = i64.const 16484\n\
  vuu = i32.const 117\n\
  i32.store8 vn0 vuu\n\
  vn1 = i64.const 16485\n\
  vpp = i32.const 112\n\
  i32.store8 vn1 vpp\n\
  vpath = i64.const 16494\n\
  vplen = i64.const 3\n\
  vflags = i64.const 577\n\
  vfd = call.cap 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
  vone = i64.const 1\n\
  vd = call.cap 13 24 (i64, i64) -> (i64) vph (vfd, vone)\n\
  vnm = i64.const 16484\n\
  vnl = i64.const 2\n\
  vz = i64.const 0\n\
  vpid = call.cap 13 27 (i64, i64, i64, i64) -> (i64) vph (vnm, vnl, vz, vz)\n\
  vsb = i64.const 16504\n\
  vr = call.cap 13 28 (i64, i64, i64) -> (i64) vph (vpid, vsb, vz)\n\
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
            temen_run::cap_thunk,
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
        ctx!(posix, w_g, p_g, st);

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

        // The embedder's door (a terminal ^C) reaches the still-installed SIGINT handler. Release
        // the guards (the `Ctx` view only borrows them) before `raise_signal` re-locks the root proc.
        let _ = st;
        drop(p_g);
        drop(w_g);
        posix.raise_signal(2);
        ctx!(posix, w_g, p_g, st);
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
        st.p.sig_mask = 1 << 4; // block it
        st.kill(&[0, 4]); // raise -> pending but blocked
        assert_eq!(
            st.sigcheck(),
            0,
            "a blocked pending signal is held, not delivered"
        );
        assert_eq!(
            st.p.sig_pending & (1 << 4),
            1 << 4,
            "the held signal stays pending across the poll"
        );
        st.p.sig_mask = 0; // unblock
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
  vprev = call.cap 13 30 (i64, i64) -> (i64) vph (vsig, vh)\n\
  vz = i64.const 0\n\
  vk = call.cap 13 31 (i64, i64) -> (i64) vph (vz, vsig)\n\
  vc = call.cap 13 32 (i64) -> (i64) vph (vz)\n\
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
    /// heap base (`20480`, above the #1094 NULL guard) and returns it; stdout is `"/bin"`.
    const GETENV_ECHO: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vp = i64.const 16384\n\
  vP = i32.const 80\n\
  i32.store8 vp vP\n\
  vp1 = i64.const 16385\n\
  vA = i32.const 65\n\
  i32.store8 vp1 vA\n\
  vp2 = i64.const 16386\n\
  vT = i32.const 84\n\
  i32.store8 vp2 vT\n\
  vp3 = i64.const 16387\n\
  vH = i32.const 72\n\
  i32.store8 vp3 vH\n\
  vnlen = i64.const 4\n\
  vptr = call.cap 13 11 (i64, i64) -> (i64) vph (vp, vnlen)\n\
  vfd1 = i64.const 1\n\
  vfour = i64.const 4\n\
  vw = call.cap 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vptr, vfour)\n\
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
            temen_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;
        // getenv materializes "/bin\0" at the aligned heap base (20480) and returns it.
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

    /// FORK.md PR 5 — `temen_posix::grant` is **forkable**: the fork factory (`cap`'s `make`) mints libc
    /// handlers over the *same* shared `Inner` (memfs + fd table + cwd/env), so a `fork()` twin's libc
    /// inherits the parent's open-file state — POSIX fork-shares-open-file-descriptions. This pins the
    /// factory half at the personality level: a handler minted by the factory sees a file the shared
    /// `Posix` handle wrote host-side. (The interp half — `fork_powerbox` carrying the factory across a
    /// twin — is pinned by `temen-interp`'s `fork_carries_a_forkable_host_proc_via_its_factory`.)
    ///
    /// func 0 `(handle) -> i64`: stage `"greet"` at offset 16384 (above the #1094 NULL guard),
    /// `open(., 5, O_RDONLY)` → fd, `read(fd, buf=16416, 3)`, `write(1, buf, 3)` echoing to stdout,
    /// return the byte count.
    const READ_GREET: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vp0 = i64.const 16384\n\
  vg = i32.const 103\n\
  i32.store8 vp0 vg\n\
  vp1 = i64.const 16385\n\
  vr = i32.const 114\n\
  i32.store8 vp1 vr\n\
  vp2 = i64.const 16386\n\
  ve = i32.const 101\n\
  i32.store8 vp2 ve\n\
  vp3 = i64.const 16387\n\
  ve2 = i32.const 101\n\
  i32.store8 vp3 ve2\n\
  vp4 = i64.const 16388\n\
  vt = i32.const 116\n\
  i32.store8 vp4 vt\n\
  vpath = i64.const 16384\n\
  vplen = i64.const 5\n\
  vflags = i64.const 0\n\
  vfd = call.cap 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
  vbuf = i64.const 16416\n\
  vcap = i64.const 3\n\
  vn = call.cap 13 1 (i64, i64, i64) -> (i64) vph (vfd, vbuf, vcap)\n\
  vfd1 = i64.const 1\n\
  vw = call.cap 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vbuf, vn)\n\
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

    /// func 0 `(handle) -> i64`: `chdir("/tmp")` (path bytes staged at offset 16384, above the #1094
    /// NULL guard), then `getcwd(buf, 8)` into a scratch window buffer, echo the result (minus its NUL)
    /// to stdout, and return `chdir_result * 1_000_000 + getcwd_ptr`. A working roundtrip: `chdir` → `0`,
    /// `getcwd` writes `"/tmp\0"` and returns the buffer offset (16416) → `0*1_000_000 + 16416 = 16416`;
    /// stdout is `"/tmp"`.
    const CHDIR_GETCWD: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vph: i32) {\n\
  vp = i64.const 16384\n\
  vsl = i32.const 47\n\
  i32.store8 vp vsl\n\
  vp1 = i64.const 16385\n\
  vt = i32.const 116\n\
  i32.store8 vp1 vt\n\
  vp2 = i64.const 16386\n\
  vm = i32.const 109\n\
  i32.store8 vp2 vm\n\
  vp3 = i64.const 16387\n\
  vpc = i32.const 112\n\
  i32.store8 vp3 vpc\n\
  vplen = i64.const 4\n\
  vcd = call.cap 13 10 (i64, i64) -> (i64) vph (vp, vplen)\n\
  vbuf = i64.const 16416\n\
  veight = i64.const 8\n\
  vgc = call.cap 13 9 (i64, i64) -> (i64) vph (vbuf, veight)\n\
  vfd1 = i64.const 1\n\
  vfour = i64.const 4\n\
  vw = call.cap 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vbuf, vfour)\n\
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
        // chdir 0, getcwd returns buf (16416) → 0*1_000_000 + 16416 = 16416; stdout "/tmp".
        assert_eq!(
            ir,
            Ok(vec![Value::I64(16416)]),
            "interp: chdir then getcwd roundtrip"
        );
        assert_eq!(iout, b"/tmp", "interp: getcwd wrote the new cwd");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[16416]),
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
        ctx!(posix, w_g, p_g, st);
        // Stage the name "K" at offset 0 and value "v2" at offset 8 in a scratch window.
        let mut win = vec![0u8; WIN];
        win[0] = b'K';
        win[8] = b'v';
        win[9] = b'2';
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
        // setenv("K", "v1", overwrite=1): name@0 len 1, value staged separately — reuse offset 8 with "v1".
        st.p.env.insert("K".to_string(), "v1".to_string());
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
            st.p.env.get("K").map(String::as_str),
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
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
        ctx!(posix, w_g, p_g, st);
        let rd = |mem: &temen_interp::WindowMem, off: u64| {
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
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
        win_write(&mut mem, 0, b"/data/sub");
        win_write(&mut mem, 16, b"/data/seed");
        win_write(&mut mem, 32, b"/gone/x");
        win_write(&mut mem, 48, b"/data");
        win_write(&mut mem, 64, b"/data/renamed");
        win_write(&mut mem, 80, b"/data/d");
        win_write(&mut mem, 96, b"/data/e");
        win_write(&mut mem, 112, b"/data/e/inner");
        win_write(&mut mem, 128, b"/data/d/inner");
        ctx!(posix, w_g, p_g, st);
        let rd = |mem: &temen_interp::WindowMem, off: u64| {
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
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
        // bind blob: v4 loopback :0 at offset 0 → [4, 0,0, 127,0,0,1]
        win_write(&mut mem, 0, &[4u8, 0, 0, 127, 0, 0, 1]);
        ctx!(posix, w_g, p_g, st);

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
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
        ctx!(posix, w_g, p_g, st);

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
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
        win_write(&mut mem, 0, b"prog");
        ctx!(posix, w_g, p_g, st);

        let pid = st.spawn(&[0, 4, 0, 0], Some(&mut mem)).unwrap()[0];
        assert!(pid >= 0, "spawn returns a pid");
        assert_eq!(st.w.stdout, b"to-out", "stdout routed to fd 1's sink");
        assert_eq!(st.w.stderr, b"to-err", "stderr routed to fd 2's sink");

        assert_eq!(st.waitpid(&[pid, 200, 0], Some(&mut mem)).unwrap()[0], pid);
        let status = i32::from_le_bytes(mem.read_bytes(200, 4).unwrap().try_into().unwrap());
        assert_eq!(
            (status >> 8) & 0xff,
            3,
            "WEXITSTATUS is the delegate's exit code"
        );
    }

    /// #972 slice 2 — a **CorePipe stdio target fails closed** on the capture spawn: the child runs
    /// to completion inside one dispatch, which can neither drain a live core pipe (blocking) nor
    /// cap-call bytes into one — so a CorePipe stdin/stdout/stderr is `-EINVAL` up front (the
    /// pre-fix behavior silently dropped the child's bytes). Adoption never exercises handles, so
    /// fake handle numbers suffice here. Live-pipe wiring is fork+execve territory (#801).
    #[test]
    fn spawn2_fails_closed_on_a_core_pipe_stdio_target() {
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        posix.set_spawn(|_n, _a, _s| SpawnResult {
            stdout: b"out".to_vec(),
            stderr: Vec::new(),
            status: 0,
        });
        let mut win = vec![0u8; WIN];
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
        ctx!(posix, w_g, p_g, st);
        win_write(&mut mem, 0, b"prog");
        st.pipe_adopt(&[7, 8, 8], Some(&mut mem)).unwrap(); // fake handles; [rfd@8, wfd@12]
        let wfd = i32::from_le_bytes(mem.read_bytes(12, 4).unwrap().try_into().unwrap()) as i64;
        win_write(&mut mem, 100, &0u64.to_le_bytes());
        win_write(&mut mem, 108, &4u64.to_le_bytes());
        win_write(&mut mem, 116, &0u64.to_le_bytes());
        win_write(&mut mem, 124, &0u64.to_le_bytes());
        win_write(&mut mem, 132, &(-1i32).to_le_bytes());
        win_write(&mut mem, 136, &(wfd as i32).to_le_bytes()); // stdout -> a CorePipe fd
        win_write(&mut mem, 140, &(-1i32).to_le_bytes());
        let r = st.spawn2(&[100], Some(&mut mem)).unwrap()[0];
        assert_eq!(
            r, EINVAL,
            "a CorePipe spawn target refuses probeably, never a silent drop"
        );
        assert!(
            st.w.stdout.is_empty(),
            "nothing ran: fail closed happened before the delegate"
        );
    }

    #[test]
    fn spawn2_routes_per_child_fds_without_touching_the_shared_stdio() {
        // #848: `spawn2` binds the child's stdio to fds named in its request struct, atomically inside
        // the one op — so a capture never mutates the shared fd-1/fd-2 binding (the parallel-driver race
        // the `dup2` bracket had). Wire a delegate that uppercases the inherited stdin to stdout and
        // writes a fixed stderr; route both to *capture pipes* and prove the global stdout/stderr sinks
        // stay empty (the child's bytes went to the pipes, not fd 1 / fd 2).
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, b"data".to_vec());
        posix.set_spawn(|_n, _a, stdin| SpawnResult {
            stdout: stdin.to_ascii_uppercase(),
            stderr: b"E".to_vec(),
            status: 5,
        });
        let mut win = vec![0u8; WIN];
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
        ctx!(posix, w_g, p_g, st);

        win_write(&mut mem, 0, b"prog"); // command name
                                         // Two capture pipes: their write ends receive the child's stdout / stderr, their read ends drain.
        st.pipe(&[8], Some(&mut mem)).unwrap(); // [rfd_out@8, wfd_out@12]
        st.pipe(&[16], Some(&mut mem)).unwrap(); // [rfd_err@16, wfd_err@20]
        let rd = |mem: &mut temen_interp::WindowMem, o: u64| {
            i32::from_le_bytes(mem.read_bytes(o, 4).unwrap().try_into().unwrap()) as i64
        };
        let (rfd_out, wfd_out) = (rd(&mut mem, 8), rd(&mut mem, 12));
        let (rfd_err, wfd_err) = (rd(&mut mem, 16), rd(&mut mem, 20));

        // Build the 44-byte request at offset 100: name="prog"@0, argv empty, stdin inherit (-1),
        // stdout→wfd_out, stderr→wfd_err.
        win_write(&mut mem, 100, &0u64.to_le_bytes()); // name_ptr = 0
        win_write(&mut mem, 108, &4u64.to_le_bytes()); // name_len = 4
        win_write(&mut mem, 116, &0u64.to_le_bytes()); // argv_ptr = 0
        win_write(&mut mem, 124, &0u64.to_le_bytes()); // argv_len = 0
        win_write(&mut mem, 132, &(-1i32).to_le_bytes()); // stdin_fd = inherit fd 0
        win_write(&mut mem, 136, &(wfd_out as i32).to_le_bytes());
        win_write(&mut mem, 140, &(wfd_err as i32).to_le_bytes());

        let pid = st.spawn2(&[100], Some(&mut mem)).unwrap()[0];
        assert!(pid >= 0, "spawn2 returns a pid");

        // The child's stdout/stderr landed in the capture pipes, NOT the shared fd-1/fd-2 sinks.
        assert!(
            st.w.stdout.is_empty() && st.w.stderr.is_empty(),
            "per-child routing never wrote the shared stdout/stderr sinks"
        );
        let n_out = st.read(&[rfd_out, 200, 64], Some(&mut mem)).unwrap()[0];
        assert_eq!(
            mem.read_bytes(200, n_out as u64).unwrap(),
            b"DATA",
            "the child's stdout drained from its capture pipe (uppercased inherited stdin)"
        );
        let n_err = st.read(&[rfd_err, 300, 64], Some(&mut mem)).unwrap()[0];
        assert_eq!(
            mem.read_bytes(300, n_err as u64).unwrap(),
            b"E",
            "the child's stderr drained from its own capture pipe"
        );

        // An all-`-1` request is exactly `spawn`: the child inherits fd 0 / 1 / 2 (routes to the sinks).
        // fd 0's preloaded stdin is already drained, so the delegate sees empty input this time.
        win_write(&mut mem, 132, &(-1i32).to_le_bytes());
        win_write(&mut mem, 136, &(-1i32).to_le_bytes());
        win_write(&mut mem, 140, &(-1i32).to_le_bytes());
        let pid2 = st.spawn2(&[100], Some(&mut mem)).unwrap()[0];
        assert_eq!(pid2, pid + 1, "a second spawn mints the next pid");
        assert_eq!(
            st.w.stderr, b"E",
            "an all-(-1) request inherits fd 2 → the shared stderr sink"
        );
    }

    /// Write `bytes` into `mem` at `off` (test helper — `WindowMem` has no direct slice setter).
    fn win_write(mem: &mut temen_interp::WindowMem, off: u64, bytes: &[u8]) {
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
        let mut mem = temen_interp::WindowMem::new(&mut win, WIN as u64);
        ctx!(posix, w_g, p_g, st);

        assert_eq!(st.p.args.len() as i64, 3, "argc");
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

    /// The manifest form a chibicc/`temen-llvm` frontend emits for unresolved libc symbols: the
    /// module *imports* the libc names `malloc`/`write` (never hand-writes a `call.cap`), each
    /// call site carries a dummy `i32.const 0` handle operand (vestigial in static dispatch —
    /// IMPORTS.md §2.5), and the entry takes **no capability parameters at all** — the granted
    /// handle arrives through the slot binding ([`bind`]), never through an entry argument
    /// (→ `2_020480`, `"hi"` — malloc base above the #1094 NULL guard).
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
            temen_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;

        assert_eq!(
            ir,
            Ok(vec![Value::I64(2_020_480)]),
            "interp: bound-handle malloc+write"
        );
        assert_eq!(
            iposix.stdout(),
            b"hi",
            "interp: the write reached the personality"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2_020_480]),
            "jit: must match interp, got {jo:?}"
        );
        assert_eq!(jposix.stdout(), b"hi", "jit: stdout must match interp");
    }
}
