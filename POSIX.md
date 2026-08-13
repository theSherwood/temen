# POSIX personality — libc as host capabilities

> Status: **core surface landed.** `svm-posix` provides ops 0–20 (stdio, `malloc`/`free`,
> `exit`, the memfs + fd table, cwd, env, argv, and the Stage-1 `exec` surface) as a `HostFn`
> capability, differential-tested on both backends. A compiled-C shell runs on it
> (`crates/svm/tests/c_shell.rs`) and dispatches external commands to spawned confined
> children (`stage1_posix_spawn.rs`; the spawn work is tracked in `STAGE1.md`). Update the
> **ABI table** and the **Status** as ops land.

## 1. The thesis: libc is not one thing

Running a shell (BusyBox `ash` → Bash) needs a libc. We do **not** bake a libc into the VM
or the guest as a fixed, trusted blob. The insight is that "libc" splits along one line —
**authority** — and only one side is naturally a host capability:

- **The authority / OS surface** — `open`/`read`/`write`/`lseek`/`stat`/`readdir`, `mmap`,
  `fork`/`exec`/`waitpid`, `signal`, `time`, `exit`, `pipe`. These carry authority, so they
  are **host-provided capabilities**, reached only through granted, masked, type-checked
  handles (§7). Most already exist as first-class caps (`Stream`, `Memory`, the fs ops,
  `Instantiator`, `Pipe`, `Clock`, `Exit`); the rest are `HostFn` ops. **This is the part
  that must never be baked in — and it isn't.**
- **The pure-computation bulk** — `strlen`/`memcpy`/`snprintf`-formatting, `qsort`, `ctype`,
  `math`, and malloc's *free-list logic*. These carry **no authority**. Whether they are
  host caps or ordinary guest code is a **performance / binary-size choice, not a security
  one**: guest code is re-verified and untrusted like any guest — it is *not* in the TCB.
  A host cap for `strlen` is a boundary crossing for a loop, so the default is to keep pure
  compute as **guest code** (compiled from the C source), and reach for a host cap only when
  binary size or startup demands it.

"Baked in" means *in the TCB*. Neither guest code nor a `HostFn` is in the TCB. The whole
personality — every byte of libc semantics — lives outside the escape-TCB match, exactly the
boundary DESIGN.md §7 draws.

## 2. The mechanism already exists: §7 named imports → `HostFn`

`svm-wasi` is the working template (a 2-op WASI shim); `svm-posix` generalizes it.

1. The shell (C) compiles to SVM IR with its libc calls left as **unresolved named imports**
   — `CallImport { "env.malloc" }`, `"posix.open"`, … (chibicc / `svm-llvm` emit these for
   any unresolved symbol).
2. At **load**, `svm_ir::resolve_imports` (driven by `svm_run::resolve_capability_imports`)
   binds each name to a `(type_id, op)` on a capability handle and lowers `CallImport` →
   `cap.call`. `svm_posix::resolve(name)` supplies the name → `(HOST_FN, op)` map.
3. A `HostFn` handler implements each op **host-side**, reading/writing the guest window
   through the masked `GuestMem`. All names share **one** `HOST_FN` handle; the op number
   distinguishes the call (svm-wasm/chibicc thread a single capability handle).

Nothing here touches the verifier or the confinement lowering. A `HostFn` is untrusted host
code reached only through a masked handle — a translation or personality bug is a clean
capability error, never an escape.

## 3. Where the state lives: bytes in the window, bookkeeping host-side

The elegant split for the stateful ops (`malloc`, `stdio`, the fd table):

- **Bytes live in the guest window.** A `malloc`'d buffer, a `FILE`'s scratch, an `iovec`
  target — all are ordinary window memory the guest reads and writes at **native speed** (no
  crossing per byte). `malloc(n)` returns a **window offset**.
- **Bookkeeping lives host-side**, in the `HostFn`'s state: the allocator's free list, the
  fd table (fd → `Stream`/`Pipe`/fs handle), `FILE` buffering, `errno`. This is the small,
  swappable part; it never enters the guest's address space, so the guest cannot corrupt it.

The allocator manages a **heap region of the guest window** the embedder configures at grant
(`[heap_base, heap_end)`): a first-fit **free list** — `malloc` reuses a freed block (splitting
off any remainder) before bumping the high-water mark, and `free` returns a block for reuse
(coalescing adjacent frees is a follow-up). The **fs** is an in-memory `path → bytes` map (a
memfs) with a host-side fd table (`open`/`close`/`read`/`write`/`lseek`); it keeps the
personality self-contained for the playground, and a native embedder routing to a real `fs`
cap is a follow-up.

## 4. One ABI, two bindings — how this unifies with "personality = guest library"

PROCESS.md frames personalities as guest libraries; this doc frames libc as host caps. The
§7 named-import ABI makes these **the same interface with different bindings**: a name can
resolve to a **host** cap (`HostFn` — fast, the playground path) *or* to a cap a **parent
serves** (an endpoint — the self-similar / interposition path, PROCESS.md Stage 2.5). The
shell's IR is identical; only the resolver's target differs. So the durable decision is to
**pin the ABI** — the function list and each function's shape — and bind it host-side now,
guest-serve the same ABI later.

**The handle binds by name too — no powerbox slot (PROCESS.md S15).** The personality is a
per-domain **singleton**, so its handle is supplied by the resolver, not threaded by the
guest: each libc import's handle operand is a `ConstI32` **placeholder** patched at resolve
(`svm_ir::Resolved::CapBound`, via `svm_posix::resolve_bound(handle)` — grant first, then
resolve; DESIGN.md §7's "late binding is the general form of the powerbox"). Consequences:
the guest's libc has **real C signatures** (`open(path, flags)`, `getenv(name)`, `malloc(n)`
— the NUL→`(ptr,len)` adaptation is a thin guest wrapper); the module's **import section is
the discoverable capability manifest** (explicit names + signatures, fail-closed — never a
silent slot numbering agreed out-of-band); and nothing about the personality touches the
fixed 8-slot `_start`, which S15 retires. Capabilities with **many** live objects (streams,
regions, pipe ends) keep the handle a call-site operand — resolver-bound handles are the
singleton case, not a replacement for first-class handles.

## 5. The ABI (POSIX subset for a fork-less shell)

Op numbers on the shared `HOST_FN` handle. `ptr`/`buf` are **window offsets**; `-errno` on
failure (`< 0`), a `>= 0` count / handle on success (except `malloc`, which returns `0` for
"no memory", the C `NULL`). Pure-compute functions are **guest code** (no cap) and are listed
only to mark the boundary.

| # | Function | Shape | Backed by | Status |
|---|----------|-------|-----------|--------|
| 0 | `write(fd, buf, len)` | `-> n \| -errno` | host sinks / fd table → `Stream`/`Pipe` | **done (spike)** — fd 1/2 → captured stdout/stderr |
| 1 | `read(fd, buf, len)` | `-> n \| -errno` | host sinks / fd table → `Stream`/`Pipe` | **done (spike)** — fd 0 → preloaded stdin |
| 2 | `malloc(size)` | `-> ptr \| 0` | window-heap arena (host bookkeeping) | **done** — first-fit free list |
| 3 | `free(ptr)` | `-> 0` | window-heap arena | **done** — reclaims for reuse (no coalescing yet) |
| 4 | `exit(code)` | `noreturn` | `Trap::Exit` (→ `Exit` cap) | **done** |
| 5 | `open(path, len, flags)` | `-> fd \| -errno` | memfs + host fd table | **done** — `O_CREAT`/`O_TRUNC`/`O_APPEND`, `-ENOENT` |
| 6 | `close(fd)` | `-> 0 \| -errno` | host fd table | **done** |
| 7 | `lseek(fd, off, whence)` | `-> pos \| -errno` | host fd table | **done** — `SEEK_SET`/`CUR`/`END` |
| 8 | `unlink(path, len)` | `-> 0 \| -errno` | memfs | **done** — `-ENOENT` if absent (aka `remove`) |
| 9 | `getcwd(buf, size)` | `-> buf \| -errno` | host cwd | **done** — NUL-terminated, `-ERANGE`/`-EINVAL` |
| 10 | `chdir(path, len)` | `-> 0 \| -errno` | host cwd | **done** — flat memfs, no existence check yet |
| 11 | `getenv(name, len)` | `-> ptr \| 0` | host env map | **done** — stable NUL-terminated `char*` in arena |
| 12 | `setenv(name, nlen, val, vlen, overwrite)` | `-> 0 \| -errno` | host env map | **done** — invalidates `getenv` cache |
| 13 | `stat(path, len, statbuf)` | `-> 0 \| -errno` | memfs | **done** — minimal `{ st_mode, st_size }`; `S_IFREG`/`S_IFDIR`, `-ENOENT` (aka `lstat`) |
| 14 | `opendir(path, len)` | `-> dir \| -errno` | memfs | **done** — snapshots immediate children; `-ENOTDIR`/`-ENOENT` |
| 15 | `readdir(dir, buf, cap)` | `-> namelen \| 0 \| -errno` | dir stream | **done** — NUL-terminated name; `0` at end, `-ERANGE`/`-EBADF` |
| 16 | `closedir(dir)` | `-> 0 \| -errno` | dir stream | **done** — `-EBADF` on a stale handle |
| 17 | `argc()` | `-> n` | host arg vector | **done** — personality ext. (the `sh -c` path; `argv[0]` = program name) |
| 18 | `argv(i, buf, cap)` | `-> len \| -errno` | host arg vector | **done** — NUL-terminated arg `i`; `-EINVAL`/`-ERANGE` |
| 19 | `exec_lookup(name, len)` | `-> module \| -1` | host PATH registry (`register_command`) | **done** — Stage 1 exec (STAGE1.md §5); the spawn itself is the shell's `Instantiator` op 13 + `join` |
| 20 | `exec_stdout()` | `-> stream` | host stdout `Stream` | **done** — the handle the shell re-grants to a child under the name `"stdout"` |
| 21 | `exec_stdin(ptr, len)` | `-> stream` | host input-pipe `Stream` + FIFO | **done** — a filter command's `"stdin"`: pushes `[ptr,len)` into the FIFO, returns the read end |
| 22 | `exec_win(module)` | `-> size_log2 \| -1` | host PATH registry | **done** — the granted command's declared window, so the shell carves each spawn to match |
| 23 | `pipe(fds_ptr)` | `-> 0 \| -errno` | host fd table + shared byte FIFO | **done** — stores `[read_fd, write_fd]` (`i32`×2); intra-personality, non-blocking (empty reads `0`) |
| 24 | `dup2(oldfd, newfd)` | `-> newfd \| -errno` | host fd table | **done** — the redirect primitive; pipe ends share the buffer, a `File` copies its description |
| 25 | `dup(oldfd)` | `-> fd \| -errno` | host fd table | **done** — clone onto the lowest free fd |
| 26 | `fcntl(fd, cmd, arg)` | `-> res \| -errno` | host fd table | **done** — `F_DUPFD`/`F_DUPFD_CLOEXEC` (dup ≥ arg); `F_GET/SETFD`/`F_GET/SETFL` accepted no-ops |
| 27 | `spawn(name, nlen, argv, alen)` | `-> pid \| -errno` | embedder spawn delegate (`set_spawn`) | **done** — fork-free child; inherits fd 0/1 (drains stdin, routes stdout); `-ENOSYS` unwired. `posix_spawn(p)` bind here |
| 28 | `waitpid(pid, status, opts)` | `-> pid \| -errno` | host process table | **done (#863)** — reaps `pid` (or `-1` = any) from the one process table: spawn children **and exited fork twins** (their exit hook parks them as zombies); wait-encoded status (`WEXITSTATUS` bits 8–15); `WUNTRACED`/`WCONTINUED` report fresh stops/continues (`sig<<8\|0x7f` / `0xffff`, once each — #798); `-ECHILD` unknown/still-running (non-blocking poll; the core wait offer is the parking channel) |
| 29 | `wait(status)` | `-> pid \| -errno` | host child table | **done** — `waitpid(-1, status, 0)` |
| 30 | `signal(signum, handler)` | `-> prev \| -errno` | host signal state | **done (L0)** — records disposition (`SIG_DFL`/`SIG_IGN`/handler ptr); returns previous; `SIGKILL`/`SIGSTOP` immutable (`-EINVAL`, #796); a reset to `SIG_DFL` runs a pending signal's default action |
| 31 | `kill(pid, sig)` | `-> 0 \| -errno` | host process table | **done (#863/#798)** — pid-targeted: `0`/own pid = self (`raise`), a table pid = THAT process's pending set (+ its run woken when deliverable), `-pgid` = the **group sweep**; zombies exist-until-reaped; `-ESRCH` unknown. (`kill(0,s)` = raise-to-self, a deliberate POSIX divergence every pre-table guest relies on) |
| 32 | `sigcheck(_)` | `-> handler \| 0` | host signal state | **done (L0)** — doorbell poll: clears+returns the next pending **caught** handler; ignored/default dropped |
| 33 | `clock(clock_id)` | `-> nanos` | host clock (pinnable, `set_clock`) | **done** — monotonic (`id 1`) / realtime; the svm `std` `time` PAL path |
| 34 | `getenv_r(name, nlen, buf, cap)` | `-> len \| -1` | host env map | **done** — buffer-writing `getenv` (no arena); size-then-fetch |
| 35 | `unsetenv(name, nlen)` | `-> 0 \| -errno` | host env map | **done** — absent name is a success no-op |
| 36 | `environ(i, buf, cap)` | `-> len \| -1` | host env map | **done** — `i`-th `KEY=VALUE`, keys sorted; `-1` past the end |
| 37 | `mkdir(path, plen, mode)` | `-> 0 \| -errno` | memfs explicit-dir set | **done** — `-EEXIST`/`-ENOENT` (parent); `mode` ignored |
| 38 | `rename(old, olen, new, nlen)` | `-> 0 \| -errno` | memfs | **done** — file move or whole-subtree re-key; `-ENOENT` |
| 39 | `rmdir(path, plen)` | `-> 0 \| -errno` | memfs explicit-dir set | **done** — `-ENOTDIR`/`-ENOTEMPTY`/`-EINVAL` (root) |
| 40 | `sigprocmask(how, set, oldset)` | `-> 0 \| -errno` | host signal state | **done (#796)** — the blocked set (`SIG_BLOCK`/`UNBLOCK`/`SETMASK`); a pending **blocked** signal is held by `sigcheck`, not delivered, until unblocked. `sigset_t` = a `u64` bitset; `SIGKILL`/`SIGSTOP` unblockable |
| 41 | `sigaction(signum, act, oldact)` | `-> 0 \| -errno` | host signal state | **done (#796)** — the richer `signal`: records the disposition (delivered by the doorbell) + `sa_mask`/`sa_flags`, round-tripped through `oldact`. `struct sigaction` = `{sa_handler:i64, sa_mask:u64, sa_flags:i64}` |
| 42 | `sigaltstack(sp, size)` | `-> 0` | host signal state | **done (#796 L2)** — register the dedicated **signal-handler stack** an async handler runs on (the interp can't reuse the interrupted frame's stack). `sp == 0` ⇒ async off (poll-only). Enables L2 delivery |
| 43 | `spawn2(req_ptr)` | `-> pid \| -errno` | embedder spawn delegate | **done (#848)** — per-spawn fd-actions (44-byte request: target + stdin/stdout/stderr fds, `-1` = inherit); parallel-safe capture — never mutates the shared fd 0/1/2 |
| 44 | `getpid()` | `-> pid` | host process table | **done (#863)** — `1` root, the `TaskId` a `fork()` returned for a twin, an allocated pid for a re-grant clone; one space with `kill`/`waitpid` |
| 45 | `setpgid(pid, pgid)` | `-> 0 \| -errno` | host process table | **done (#798)** — `0` = self / own-id; table-routed; a `fork` twin **inherits** its parent's group |
| 46 | `getpgid(pid)` | `-> pgid \| -errno` | host process table | **done (#798)** |
| 47 | `tcgetpgrp(fd)` | `-> pgid \| -errno` | proto-terminal (captured stdio) | **done (#798)** — the foreground group; `-ENOTTY` off-terminal. The pty proper is #797 |
| 48 | `tcsetpgrp(fd, pgid)` | `-> 0 \| -errno` | proto-terminal | **done (#798)** — foreground the group (`-EINVAL`/`-EPERM` empty); background terminal I/O then **stops** the offender (default disposition; caught rings the pending doorbell, ignored proceeds — POSIX). Stop/continue is real: the domain parks at its next safepoint (#798 slice 2) |
| — | `fstat` | / `-errno` | memfs + host fd table | todo |
| — | default actions + `EINTR` | doorbell (§9 L1/L2) | host signal state | **partial** — **L2 async delivery to a running loop is DONE** (#796): with a handler + `sigaltstack`, the interp redirects a caught, unmasked signal into `void handler(int)` at a per-op safepoint and resumes. **`EINTR` on blocked calls is DONE** (#863: a deliverable signal — embedder `^C`, `kill(pid)` — interrupts the target domain's blocked pipe I/O and `wait` reaps with `-EINTR`, domain-scoped per INVARIANTS.md #12). **Stop/continue is DONE** (#798: `SIGSTOP`/`SIGTSTP`/`SIGTTIN`/`SIGTTOU` default actions park the domain at a safepoint; `SIGCONT` resumes; a stopped process holds all delivery until continued). **Default-action terminate is DONE** (#796: an unhandled fatal signal fires the core's `SignalSource::set_kill` door — the domain's term flag, death at the next per-op poll, `waitpid` reports `WIFSIGNALED` — while ignored/default-ignore signals are discarded at generation; a masked or stopped process holds a fatal signal until unblock/continue, `SIGKILL` excepted). **Block-during-handler + nested delivery are DONE** (#796: delivery pushes the mask and blocks the signal + its `sa_mask` for the handler's duration, restored via `SignalSource::handler_returned`; a different unmasked signal nests, same-signal reentry cannot). **`SA_RESTART` is DONE** (#796: a restart-flagged delivery re-issues an interrupted blocking pipe op — handler runs promptly, the op re-parks — and leaves a parked `wait` parked, the handler landing at the wait's completion; plain `signal()` stays SysV no-restart). **Still parked:** L1 EINTR for the remaining park kinds (futex/cap/completion), JIT/bytecode parity for the async safepoint |
| — | `fork/vfork/execve` | Stage 3 | durable clone (§7) | **parked** — return-twice / image-replace need the durable-clone capstone (R8 ✓); `spawn`+`waitpid` (ops 27–29) cover the fork-free process model a shell drives today |
| — | `strlen/memcpy/snprintf/qsort/ctype/math` | pure | **guest code** (no cap) | n/a |

## 5a. The `net` capability — sockets without growing the libc table

Networking follows the WASI 0.2 lesson (`wasi:sockets`): **authority is an explicit granted
handle, and the data plane reuses the generic fd path**. It is a **separate named capability**
(`"net"`, resolved like `"posix"`) over the *same* shared personality state — not more entries
in the libc op table above. Socket-ness lives only at connection setup; a connected socket is
an ordinary fd, read and written through ops 0/1 (`read`/`write`) and closed/`dup2`'d like any
other, so redirects (`dup2(sock, 1)`) work unchanged.

**Request/refuse, not preopen-only.** The guest *requests* (`bind :8080`, `connect host:port`)
and the host side grants, remaps, or refuses — mechanism in the personality, policy in the
embedder:

- **Loopback = the memnet** (the memfs analog): binds and connects on `127.0.0.1`/`::1` are
  served in-personality — private per-instance byte FIFOs (the `pipe` machinery), ephemeral
  `:0` assignment, deterministic, playground-safe. No external authority exists here, so it
  needs no grant beyond the `net` cap itself.
- **Beyond loopback = the embedder's `NetDelegate`** (`Posix::set_net`, the `set_spawn`
  analog): `connect`/`resolve` route to it — a real socket, a scripted table, an allowlisting
  proxy. **No delegate ⇒ fail closed** (`-ECONNREFUSED`/`-ENOENT`), exactly like spawn's
  `-ENOSYS`. Non-loopback `bind` (a delegate-granted real listener) is the noted follow-up —
  the *op* carries the request today; the delegate hook is what lands later.

Blocking: a memnet `read`/`accept` on empty returns `-EAGAIN` (a single cooperative guest
blocking on itself would deadlock; lockstep guests never see it, `set_nonblocking` programs
get `WouldBlock`). A delegate-backed `recv` may block **host-side** inside the call, like a
spawn running its child. A socket address travels as a tiny blob — `[family u8 (4|6),
port u16 LE, addr 4|16 bytes]` — sized for the 4-arg call ABI.

Ops on the `net` handle (own numbering; `-errno` on failure):

| # | Function | Shape | Notes |
|---|----------|-------|-------|
| 1 | `connect(addr, alen, laddr_out, cap)` | `-> fd \| -errno` | loopback → memnet peer; else delegate or `-ECONNREFUSED`; writes the local addr |
| 2 | `bind(addr, alen, bound_out, cap)` | `-> fd \| -errno` | bind+listen folded; `:0` assigns an ephemeral port; writes the actual bound addr; non-loopback `-EACCES` (slice 1) |
| 3 | `accept(fd, peer_out, cap)` | `-> fd \| -EAGAIN \| -errno` | next pending memnet connection; writes the peer addr |
| 4 | `shutdown(fd, how)` | `-> 0 \| -errno` | write-shutdown makes the peer's empty reads return `0` (EOF), not `-EAGAIN` |
| 5 | `resolve(name, nlen, out, cap)` | `-> nbytes \| -errno` | `localhost` → loopback; else delegate or `-ENOENT`; writes addr blobs |

UDP (`sendto`/`recvfrom` on the memnet) is a follow-up slice on the same cap.

## 6. Roadmap

1. **Spike (done):** `svm-posix` crate — `write`/`read`/`malloc`/`free`/`exit` as a `HostFn`,
   differential interp↔JIT (`svm-posix` tests). Proves the arena-in-window + host-bookkeeping
   model and the cross-backend parity.
2. **Named-import binding (done):** a real C `main` (via chibicc) links its libc calls to the
   personality through named imports — the real linking path, not hand-written cap.calls
   (`crates/svm/tests/c_posix.rs`). Bound in the §7 **general form**: `resolve_bound` supplies
   the handle at resolve (`Resolved::CapBound`), so the guest libc has real C signatures and
   no powerbox slot (§4 above; the fixed-`_start` retirement is PROCESS.md S15).
3. **fs + fd table (done):** `open`/`read`/`write`/`close`/`stat`/`readdir` over the memfs,
   with a host-side fd table (ops 5–16 above); a real free-list allocator.
4. **A first shell (done — a compiled-C shell, not BusyBox):** fork-less at Stage 0 — `sh -c`,
   builtins, redirection, pipelines (staged through memfs temp files; `ls | grep` via the
   `Pipe` cap is still open) — `crates/svm/tests/c_shell.rs`. This is the playground target.
5. **Signals (L0), time** (env landed — ops 11/12), then Stage 3 (`fork`/`exec`) on top of
   `Instantiator` / clone. The `exec` half landed first as Stage 1 spawn — ops 19/20 plus the
   shell's own `Instantiator` op 13 + `join` — tracked in `STAGE1.md`; `fork` remains.

Testing follows the repo standard: every op is an interp ↔ bytecode ↔ JIT differential
(errno paths included), because a `HostFn` dispatches through the same `cap_dispatch_slots`
the JIT's `cap.call` thunk calls — parity for free.
