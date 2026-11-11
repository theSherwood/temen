# The `exec` capability — one interface, host processes or domains

Design record for the subprocess capability (first consumer: **jacl**,
`theSherwood/jacl_impl/docs/TEMEN_EXEC_ASK.md` — shell-out for `!cmd` /
`[exec …]`). Settled 2026-07-22 with the §3.6 consumer pinning
(IMPORTS.md §3.6); implementation follows the `temen-fs` mold.

## The one-interface decision

A guest imports **one** interface, `"exec"`: a flat import of the one
entry (or `self.resolve("exec")` + `call.cap HOST_PROC 0`, which reaches
only a host servicer), with its message region `"exec.msg"` beside it.
*What runs* is the wirer's choice, invisible to the guest — the
interposition-invisibility property the import model already guarantees:

| backend | what a spawn is | status |
|---|---|---|
| `host_exec(allowlist)` | a real host OS process, attenuated by an explicit program allowlist (the capability *is* the list, as `host_fs`'s *is* the root) | **BUILT 2026-07-22** (`temen-run/src/exec.rs`) |
| `scripted_exec(table)` | no process at all: a `(argv-prefix → {stdout, stderr, exit})` table — the `mem_fs` analog; what differential tests and wasm/browser embedders grant | **BUILT 2026-07-22** (wasm-safe `temen-exec` crate) |
| `domain_exec` | a **child temen domain** (host-served: the embedder implements the same ops over the Instantiator machinery it already has) | **BUILT 2026-07-23** (`temen-run/src/exec.rs`) |
| guest-served | a parent domain serves its child's `"exec"` with **its own code** — the none-the-wiser nested shell | **BUILT** (#744; with real argv and output through the message region, #2962 — `crates/temen/tests/exec_region.rs` runs one chibicc-compiled client against `host_exec` and against a guest servicer on every driver): a detached spawn's live self-serve grant (`GRANT_SERVE_LIVE_TAG`), served natively by the tree-walker, the Cranelift JIT, the cooperative bytecode driver (`drive`, and the browser's `CoopRun`, whose tier-up regions end before a spawn, `svc.wait` or live call) and the native parallel driver (`drive_threads`, on the pump's rules since #1414 3e); the debug driver declines it until #1414 gives it the cooperative pump. A granted module a child runs is admitted by the same rule as the run's own program, so a child image, which keeps its program's serve loop and spawns, runs where its program does (#2284) |

This is how "a shell that manages real host processes" and "the same
shell nested, its parent handling process-like domains, none the wiser"
are the *same program*: the shell's manifest names `"exec"`; the wirer
decides. A shell that *knowingly* wants both at once is not
none-the-wiser — that is a visible, honest choice: two imports (e.g.
`exec.host`, `exec.domain` — dotted names are convention), each wired,
shared, or denied (fail-closed; the guest's fallback runs) by the parent.
Interposition invisibility is per-slot; the manifest is the requirement
statement.

Un-granted, `__vm_cap_resolve("exec")` stays negative and the guest's
fallback runs (jacl: a catchable error value) — a subprocess is pure
authority; without a grant the honest behavior is an error, not
emulation.

## Op protocol (v1 — blocking one-shot)

One entry, the op in its first argument; errors are negative errno-style
returns, never traps (§3e D42). `argv` is NUL-separated bytes. **Every
byte crosses the message region** (#2962, universal B): a `SharedRegion`
the wirer grants beside the slot under the name `exec.msg`
(`EXEC_MSG_LEN` = 64 KiB), and every offset below is region-relative.

```
exec(RUN,      off, argv_len, stdin_len) -> job | -errno          ; argv at msg[off..], its stdin right after
exec(READ_OUT, job, off, cap)            -> n | 0 = EOF | -errno   ; the next ≤ cap bytes land at msg[off..]
exec(READ_ERR, job, off, cap)            -> n | 0 = EOF | -errno
exec(STATUS,   job)                      -> exit code | -errno
exec(CLOSE,    job)                      -> 0 | -errno
```

(`RUN` = 0 … `CLOSE` = 4; the shape is `(i64, i64, i64, i64) -> (i64)`.)

- **Why one entry.** A flat import of this shape binds the same way to a
  host procedure (op 0) and to a guest's live offer (its one op, matched
  by signature). A call that names the op in its immediate —
  `call.cap HOST_PROC op`, what `__vm_host_call` lowers to — never reaches
  a guest servicer, so no client written that way could be interposed.
- **Why a region.** A guest servicer cannot read its caller's window
  (PROCESS.md, wire discipline), and a host servicer that reads the
  window serves a different data plane than a guest one could. With the
  region, both read the same bytes, and the host servicer never touches
  the window at all.
- **Who makes it, and when.** The wirer, at grant time: each host backend
  mints it as the capability is granted (charged to the grantee) and
  registers `exec.msg`; a guest parent mints one, maps it into its own
  window, and grants it with its live `exec` grant. One region per grant,
  so a servicer with several callers holds one each.
- **Concurrent calls.** The servicer touches only the ranges a call names,
  only between the call and its reply. Keeping in-flight calls on disjoint
  ranges (per-thread slices, or a lock) is the caller's library's job; a
  library that splits the region aligns slices to 64 B.
- **Too big.** A `run` whose argv and stdin together pass the region's end
  is `-E2BIG`; output drains in chunks, so it never needs more. A range
  past the region is `-EFAULT`.
- **The bounce.** A caller's library copies argv and stdin in before `run`
  and each chunk out after `read_out`. A program that keeps its data in
  the region pays nothing: a driver can read one job's output straight into
  the next job's stdin slot (`multibinary.rs`).
- **v1 `run` is synchronous-to-completion**: the process runs with the
  given stdin, its output captured; `read_out`/`read_err` then drain the
  captured bytes in chunks; `status` reports the exit code. Blocking
  one-shot is what shell-out needs (the ask, verbatim); pipelines,
  incremental stdin, and signals extend this surface later without
  breaking it.
- **Streaming is reserved, not promised**: the contract permits a future
  backend to return `read_out` bytes *before* the process exits (at
  which point `status` may block). v1 callers that only read after
  `status` are forward-compatible with that; do not write guests that
  *depend* on reads being complete-at-run-return.
- **A program outside the allowlist is a refused op** (negative return),
  not a trap — probeable, like every authority miss. An **empty
  allowlist means "any"** and is a choice the embedder must spell out.

## Contract corners (unified across backends, stated honestly)

- **Exit code domain**: `status` returns an `i64`. `host_exec` reports
  the host's wait status collapsed the POSIX-shell way (exit code
  0–255; signal death reported as `128 + signo`). `domain_exec` reports
  the child domain's entry result verbatim. `scripted_exec` reports the
  table entry. A portable guest treats 0 as success and nonzero as
  failure and nothing more.
- **argv resolution**: `host_exec` matches `argv[0]` against the
  allowlist (the capability's attenuation); `scripted_exec` matches the
  longest argv-*prefix* in its table; `domain_exec` resolves `argv[0]`
  through a Module registry (the `exec_lookup` pattern, personality
  policy).
- **Environment**: not part of v1. A backend that wants env passes it at
  grant time (embedder-side), the way `host_fs` fixes its root — never
  ambient.
- **Blocking profile** (same interface, different resource): `host_exec`'s
  `run` blocks a host thread — it is `Blocking`-shaped and offloadable
  through the §12 offload pool (a parked completion); `domain_exec`'s `run` parks a
  fiber. Guests see one synchronous op either way.
- **Determinism**: `scripted_exec` is fully deterministic (the test and
  browser grant); `domain_exec` is deterministic under fuel;
  `host_exec` is not deterministic and never claimed to be.

## Trust placement

The exec capability is also the trust ladder's bottom rung made
concrete (IMPORTS.md §3.6 consumer pinning): genuinely hostile code
belongs in a **separate OS process** — which is exactly what
`host_exec` grants the ability to create. In-process domains (with
`attest` + attenuated grants) cover trusted and limited-trust code;
tiers 0/1 are never a Spectre boundary (§1a).

## Acceptance (from the ask)

A guest that resolves `"exec"` and runs `echo hi` sees `hi\n` + exit 0
under `host_exec(["echo"])`, byte-identical under a `scripted_exec`
seeded with the same entry, **interp == jit** on both; un-granted,
resolve stays negative and the fallback runs; an allowlist miss is a
negative return, not a trap.

## Implementation shape (the temen-fs mold, exactly)

Protocol constants + the deterministic `scripted_exec` handler live in a
**wasm-safe crate** (`temen-exec`, mirroring `temen-fs`) so the browser
cdylib can grant it; the real `host_exec` backend and the `HostCap`
constructors live in `temen-run` (`exec.rs`, mirroring `fs.rs`).
`domain_exec` lands beside them when built; the guest-served backend is
§3.6/Endpoint work and is recorded there.

**As built (2026-11-11, #2962).** The protocol moved whole to the one-entry,
message-region form above. `temen-exec` now holds the one dispatcher every
host backend serves through (`serve`: the entry, the region, the job table),
so the backends differ only in how a job is made (`RunJob`); `temen-run`'s
`exec_cap` mints and grants the region at grant time for all three.

**As built (2026-07-22).** `temen-exec` holds the protocol (op codes, errno,
NUL-argv parsing) plus the shared `JobTable` — every backend routes its
non-`run` ops through the one table, so read/status/close semantics are
backend-identical by construction — and the scripted backend (longest
argv-prefix wins; a miss is `-EPERM`, the *same* refusal an allowlist miss
produces, so the failure mode doesn't reveal the backend either).
`temen-run/src/exec.rs` adds `host_exec(allowlist)` (blocking one-shot spawn,
stdin fed then closed, both streams captured, POSIX-shell status collapse)
and the `HostCap` constructors, re-exporting `temen-exec` as one surface.
Acceptance pinned by `crates/temen-run/tests/exec_cap.rs`: `echo hi` →
`hi\n` + exit 0, byte-identical host vs scripted, on all three backends;
un-granted resolve negative with the fallback running; allowlist miss
probeable, never a trap (the host test is unix-only — Windows has no
`echo` executable; the protocol is covered everywhere by the scripted
differential).

**As built (2026-07-23): `domain_exec`.** The third row lands beside
`host_exec` in `temen-run/src/exec.rs`: `domain_exec(Vec<DomainProgram>)`
— each entry a `{name, Arc<Instance>, Limits}` — where a `run` is a
**fresh temen domain** (its own window, powerbox, fuel; no OS process
anywhere): `argv[0]` resolves by exact name through the registry (the
registry *is* the attenuation; a miss is the same `-EPERM` as an
allowlist/table miss), the wire `stdin` seeds the child's `Stream{In}`,
the full argv rides the §3e args buffer (a `main(int, char**)` program
reads it exactly as standalone), both output streams are captured into
the shared `JobTable`, and the exit code is the child's entry result
**verbatim** (`Exit(code)` or the first returned value). A child that
**traps** (or exhausts its fuel) is a *failed `run`* — probeable
`-EINVAL`, never a trap in the caller and never an invented exit code
(v1; a trap→exit-code mapping is a reserved refinement). Runs on the
reference interpreter — deterministic under the per-program fuel bound,
as the contract claims — and blocking one-shot, exactly `host_exec`'s
profile (the §3.6 fiber-parking refinement is the Endpoint/guest-served
row's business). Acceptance extends `exec_cap.rs`: the **same consumer**
that ran real and scripted `echo` runs a domain `echo` **byte-identical**
on all three backends (three kinds of spawn, one guest, no way to tell);
stdin round-trips a cat-like domain program; a trapping program probes
negative. The guest-served backend (a parent domain serving its child's
`"exec"` with its own code) remains the §3.6 Endpoint consumer.
