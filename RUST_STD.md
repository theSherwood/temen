# Rust `std` on the svm-llvm on-ramp — design & tracking

The plan, the decisions, and the prioritized slices for running **full-`std` Rust as an svm
guest** through the LLVM on-ramp, bound to the **POSIX personality** (`svm-posix`) — not WASI.
The analog of `NIM.md` for this workstream; fold completed sections into `DESIGN.md`/`LLVM.md`
and drop this file once the actionable gaps close (the repo convention, cf. the former
`WASM.md`).

> Status: **scoping complete, no slices started.** `core`+`alloc` Rust is done and
> byte-identical to native (`LLVM.md` slices AH–AM); full `std` has never run
> (`NIM.md` §3e names the blockers). Route decision recorded below (§1, §4). First
> implementation slice is S0+S1 (§8).

Section numbers like "§7" refer to `DESIGN.md` unless prefixed with a file; "D54" etc. are its
Decision Log; "I55" is `ISSUES.md`.

---

## 1. Goal and the route decision

**Goal:** an ordinary `std` Rust program — `println!`, `File`, `env`, `Instant`, `HashMap` —
compiles with `rustc`, translates through `svm-llvm`, and runs as a guest **byte-identical to
native**, with its authority reaching the host only through the svm-posix capability ABI.

**Route decision (owner, 2026-08-11): svm-posix, not WASI.** The WASI route (build
`wasm32-wasip1`, fatten `svm-wasi` from 2 ops to ~30, ride `svm-wasm`) would inherit upstream's
maintained `std` port, but couples our Rust story to the wasm/WASI toolchain and spec surface —
a dependency we don't control and don't otherwise need (`svm-wasi` stays the deliberately-thin
2-op shim it is today). The svm-posix route keeps everything on the proven LLVM path
(`rustc --emit=llvm-ir` → `svm-llvm`, the same lane as slices AH–AM) and binds `std` to the
personality ABI we already pin and test (`POSIX.md` §5, ops 0–32). Cost: we author and maintain
a small `std` platform layer ourselves (§4, §5).

**Trust framing:** everything in this workstream is untrusted, re-verified frontend +
personality + guest code — the translator (`svm-llvm`), the `HostFn` personality (`svm-posix`),
and `std` itself compiled as guest code. Zero escape-TCB, same class as chibicc/`svm-wasm`
(§2a, D54; INVARIANTS §2/§9 untouched). A `std` or personality bug is a clean capability/verify
error, never an escape.

## 2. What already works (don't re-plan it)

- **`core` + `alloc` Rust, byte-identical to native** — trait objects, slices, enums, `Vec`
  growth through a guest allocator, checked arithmetic, `i64` discriminant `switch`, the jsmn
  capstone (`LLVM.md` slices AH–AM; `w5_rust_guest.rs` runs the leng-shaped surface).
- **No toolchain pin.** The on-ramp reads **textual `.ll`**, so it ingests whatever LLVM the
  default `rustc` bundles — the container's `rustc 1.94` (LLVM 21) flows straight through
  (`translate.rs` "Milestone 2" header). The old LLVM-18/1.81 bitcode pin applies only to the
  legacy `llvm-ir`-binding lanes (`peval_*`), not this workstream.
- **Panic-path lowering** for `panic=abort`: external `core::panicking::*` entries → trap
  (`is_rust_abort_call`, slice AI). With `-Zbuild-std` these become *defined* guest functions
  instead — see the risk in §9.
- **The whole svm-posix ABI is reachable from translated code today** with no on-ramp changes:
  `__vm_cap_resolve("posix")` → handle, then `__vm_host_call(handle, OP, a, b, c, d)` drives
  any op (`posix_cap.rs` exercises pipe/dup2/spawn/waitpid end-to-end on all three engines).
- **Auto-vectorized `rustc -O2` output** is ingested (slices AN–AT) — no `-C` opt gymnastics
  needed.
- Most of what "std" means to a program — `Vec`/`String`/collections/`fmt`/`iter`/`error` —
  is re-exported `core`/`alloc` and **already runs**. The new work is confined to the platform
  layer: `std::{io, fs, env, time, process, sync, thread, net}`.

## 3. The seam: how `std` reaches svm-posix

Two existing binding mechanisms; the plan uses both, each where it's strongest:

1. **The generic host-call bridge (primary).** `int __vm_cap_resolve(const char *name, long
   len)` resolves the embedder-granted `"posix"` cap to a handle (§7 `cap.self.resolve`);
   `long __vm_host_call(int handle, int op, long a, long b, long c, long d)` →
   `cap.call HOST_PROC` (`svm-llvm/src/lib.rs` `"__vm_host_call"`). The `std` platform layer
   declares these two `extern "C"` symbols and speaks the `POSIX.md` §5 op table directly.
   **No allowlist growth, no per-name plumbing** — new ops cost zero translator changes.
   Constraints to respect:
   - `op` must be a **compile-time constant** at the LLVM call site. The shim gives each op
     its own `#[inline(always)]` wrapper with a literal op (never a shared `fn call(op: u32)`
     helper that could stay a runtime value).
   - Fixed `(i64×4) -> i64` payload. Every op the `std` surface needs fits except `setenv`
     (5-ary) — see §7.
   - Errno is **in-band** (`-> n | -errno`, INVARIANTS §5): the shim maps negative returns
     straight to `io::Error`. **No `__errno_location`, no errno TLS** — a genuine advantage
     of this ABI over a libc-shaped one.
2. **Named imports (secondary).** The `cap_import_name` allowlist + `svm_posix::resolve`
   name-binding path (slice N). Used only where it already wins: `malloc`/`free` left as
   external calls get the **synthesized guest bump allocator** (`synth_malloc`, slice S) —
   zero host crossings per allocation. The `std` platform's `sys::alloc` therefore calls
   `extern "C" malloc/free` rather than posix ops 2/3 (which cross the boundary per call;
   revisit only if the measured difference says otherwise).

State the ops need — fd table, memfs, cwd, env map, argv — already lives host-side in
`svm-posix` (ops 0–32 done; `POSIX.md` status). The embedder wiring exists
(`svm_run::posix::posix_cap` + `run_with_caps(&[("posix", cap)])`).

## 4. Alternatives considered — final review (recorded so we don't re-litigate)

- **A. Custom `std::sys` platform ("PAL") for an svm target — CHOSEN.** A target JSON
  (`x86_64-unknown-svm.json`-shaped, `panic-strategy: "abort"`) + `-Zbuild-std` + a small
  platform module patched into the toolchain's `rust-src`. The in-tree precedent is exactly
  this shape: a PAL is a contained directory (`library/std/src/sys/`), and the `wasi`/`sgx`
  PALs show the pattern of re-exporting the `unsupported` PAL for everything not implemented,
  overriding only `stdio`/`fs`/`env`/`args`/`time`/`alloc`. The PAL talks `__vm_host_call`
  directly (§3), so struct layouts are **ours** (no reconciliation): `stat` is svm-posix's
  `{st_mode, st_size}`, timestamps are whatever the Clock op returns. Cost: a rust-src overlay
  patch to rebase when the pinned nightly moves — contained, mechanical, and entirely in
  guest/untrusted code.
- **B. Reuse `std::sys::unix` via a Linux-claiming triple + a guest `svm-libc` shim —
  REJECTED.** No std fork, but the price is the libc-crate ABI: ~50–80 symbols including
  `pthread_mutex_*`/`pthread_key_*` TLS, `sigaction`, `mmap`, glibc struct layouts to
  reconcile — and, decisively, **raw syscalls**: `std::sys::unix` and the `getrandom` crate
  reach for `libc::syscall(SYS_*)` on Linux, which would force a syscall-number emulation
  layer. That is precisely the surface the personality model exists to avoid (`POSIX.md` §1:
  authority rides *named, typed* capability ops, never a numeric syscall multiplexer).
  Lying about `target_os` also leaks into every ecosystem crate's `cfg`. Violates "do less"
  twice over.
- **C. WASI route — REJECTED for this workstream** (owner call, §1). Recorded trade: we give
  up upstream's maintained `std` port; we keep toolchain independence and one on-ramp.
  Note the wasm on-ramp remains the stated Path-W end-goal for *self-hosting* (`NIM.md` §3e
  slice 5) — that workstream is unaffected; it targets `wasm32-unknown-unknown` (no WASI)
  precisely because `svm-wasi` stays thin.
- **D. `restricted_std` (custom target, unpatched std) — S0 ONLY.** `-Zbuild-std` against an
  unknown target compiles `std` with the `unsupported` PAL: programs link and run but every
  platform call errors. Useless as a destination, **perfect as the S0 smoke test** that the
  target JSON + build-std + translate + link lane works before the PAL exists.
- **E. No-`std`-at-all forever (status quo) — REJECTED** by the premise: `no_std + alloc`
  blocks every crate that touches `std::io`/`fs`/`time` even incidentally, which is most of
  crates.io — the breadth D54 exists to buy.

## 5. ABI pins (decide once, up front)

| Pin | Value | Why |
|---|---|---|
| Panic strategy | `panic=abort` (target JSON + `-Zbuild-std=std,panic_abort`) | Sidesteps unwinding entirely; EH (`invoke`/`landingpad`) is planned substrate, not built (`LLVM.md` §"setjmp/longjmp + C++ EH"). Panic *messages* still format and print via the PAL's stderr before the abort → trap. |
| Threading | **Single-threaded v1.** `thread::spawn` → unsupported `io::Error`; `Mutex`/`RwLock`/`Once` single-thread trivial; TLS via plain statics (`has-thread-local` off, PAL key-based TLS over a static table) | The one big deferrable. svm has a threading model (`THREADS.md`) but wiring std's thread/futex/TLS backend is its own project (S5). |
| Errno | In-band negative returns → `io::Error`; no errno TLS | INVARIANTS §5; §3 above. |
| Allocator | PAL `sys::alloc` → `extern "C" malloc/free` → synthesized guest bump allocator | Zero host crossings; already built (slice S). |
| `stat`/time layouts | svm-posix's, verbatim (`{st_mode, st_size}`; Clock op's epoch/units) | The PAL is the only consumer — no reconciliation layer. |
| HashMap seeding | v1: fixed-seed `RandomState` in the PAL; v2: `getrandom` op (§7) | Unblocks `HashMap` without new ops (`NIM.md` §3e flagged this exact blocker). Fixed seed is *per-guest determinism*, which the differential harness wants anyway. |
| Toolchain | One pinned **nightly** (build-std requires it) + `rust-src`; overlay the PAL patch onto the toolchain's `rust-src` (scripted, `scripts/`) | The textual-`.ll` reader frees us from LLVM-version pins (§2); the nightly pin is for build-std reproducibility only. Pin, don't drift (`LLVM.md` §2). |

## 6. Missing svm-posix surface (all small, all additive)

| Gap | Op | Notes |
|---|---|---|
| `clock_gettime` | new op → existing `Clock` cap | Already `todo` in the `POSIX.md` ABI table. Needed by `Instant`/`SystemTime` (S2). Pin monotonic vs realtime as two ops or an arg. |
| `getrandom` | new op → a randomness cap | Deferred behind the fixed-seed pin (S4). When added: embedder-supplied, so tests stay deterministic. |
| `setenv` arity | none | 5-ary; `__vm_host_call` carries 4. `std::env::set_var` always overwrites → the PAL can ride a 4-ary call with `overwrite` implied, via the *named-import* binding for this one name, or a new 4-ary `setenv_ov` op. Decide in S2; do less. |
| `fsync`/`ftruncate`/`rename` | ops exist in the `fs` cap world (`fs_cap.rs`) but check the posix table | Audit in S3 against what `std::fs::File` actually calls. |

Everything else `std::{io, fs, env, args, process::exit}` needs is **already an op** (0–18).

## 7. What stays out of scope (v1 non-goals)

- **`std::thread` / real sync / TLS** — S5, its own project over `THREADS.md`.
- **Unwinding** (`catch_unwind` that returns, `panic=unwind`) — blocked on the EH substrate;
  `catch_unwind` under `panic=abort` compiles to a plain call and is fine.
- **`std::net`** — no socket personality; nothing needs it yet (no named consumer,
  INVARIANTS §1).
- **`std::process::Command`** — spawn/waitpid ops exist (27–29), but wiring std's
  child-process model is post-v1; `exec`-style dispatch stays the shell's lane (`STAGE1.md`).
- Windows/second-arch target JSONs, `build-std` for stable, upstreaming the PAL.

## 8. Slices (smallest-first; every slice lands with its differential test)

| # | Slice | Needs | Proves / test |
|---|---|---|---|
| **S0** | Target JSON + pinned nightly + `-Zbuild-std` (restricted-std, no PAL); `fn main(){}` → `.ll` → translate → verify → run on both backends | toolchain script; asset-lane CI check (guest builds live outside the per-PR gate — I55, the `chibicc.svmb` template) | The build lane exists. Also answers the §9 `llvm.trap`/defined-panic-symbol risks on a real `std` module. |
| **S1** | Minimal PAL: `stdio` (posix ops 0/1), `exit` (4), `args` (17/18), `alloc` (extern malloc); `println!` + `process::exit` + `env::args` | S0 | **First `std` hello-world byte-identical to native rustc** — the `check_powerbox_vs_native` analog, in `crates/svm-llvm/tests/` beside `w5_rust_guest.rs`. |
| **S2** | PAL `env` (11/12, setenv decision §6) + `time` (new Clock op) | clock op | `env::var` round-trip + `Instant` monotonicity vs native. |
| **S3** | PAL `fs`: `File` open/read/write/seek/close, `metadata`, `read_dir`, `remove_file` (ops 5–16) | op audit (§6) | Full-file-I/O program over the memfs, byte-identical vs native on a real dir tree. |
| **S4** | `HashMap` end-to-end (fixed-seed `RandomState`; optionally the getrandom op) | — | Retires the `NIM.md` §3e `RandomState` blocker; a `HashMap`-heavy program byte-identical to native. |
| **S5** | *(deferred, separate designs)* threads/sync/TLS; unwinding; net; `Command` | new ops + EH substrate | Each gets its own slice plan when a consumer names it. |

**W5 payoff (cross-link):** S1–S4 is exactly the surface `svm-leng` needs to run as a `std`
crate *without* the `no_std + alloc` rework `NIM.md` §3e slice 2 currently assumes — the
fixed-seed hasher lands here as the PAL default instead of a per-crate patch. Re-scope that
slice when S4 lands.

## 9. Risks / things S0 must answer

- **`op` constant-folding.** If rustc outlines a shared host-call helper, `op` reaches the
  call site as a runtime value → clean `Unsupported`. Mitigation pinned in §3 (per-op
  `#[inline(always)]` wrappers); S0/S1 asserts it holds at `-O2` **and** `-O0`.
- **Defined panic machinery.** With build-std, `core::panicking::*` are guest-defined, so the
  slice-AI external-recognizer no longer fires; the panic path *executes* (format → PAL
  stderr → `panic_abort` → `llvm.trap`/`unreachable`). S0 verifies the translator handles
  the terminal intrinsic; if not, that's a one-case lowering (→ `Trap`), not a design change.
- **`lang_start` surface.** `std::rt` init (argc/argv plumb-through, `sys::init`,
  stack-guard setup, `sigaltstack` on unix — ours is a no-op) compiles as ordinary guest
  code; S1's hello-world is the existence proof. Unknowns land as `unsupported`-PAL errors,
  which fail loud, not wrong (INVARIANTS §9's decline-never-diverge, applied to a runtime).
- **std-internal `thread_local!`** (stdout buffering, panic counters) must work
  single-threaded — the static-table key TLS in §5 covers it; S1 exercises it via `println!`
  (stdout's `RefCell` buffer is TLS-backed).
- **Nightly rebase cost.** The PAL overlay must re-apply when the pin moves. Keep the patch
  minimal (re-export `unsupported`, override six modules); the S0 CI lane catches drift.
- **CI drift (I55).** The guest-std build lives outside the workspace; without an asset-lane
  check it rots silently — S0 lands the lane check *with* the first artifact, not after.

## 10. Invariants respected

- **Zero escape-TCB** — translator, personality, PAL, and `std` itself are all untrusted /
  re-verified (INVARIANTS §9; §2a). No verifier or masking change anywhere in the plan
  (INVARIANTS §2).
- **Authority = named capability ops, never a syscall multiplexer** (`POSIX.md` §1) — the
  §4-B rejection is this invariant applied.
- **Host = mechanism** (INVARIANTS §4): new ops (clock, getrandom) are bookkeeping over
  existing caps; no policy, no scheduling, no priorities.
- **Errors are values** (INVARIANTS §5): errno stays in-band; the PAL never adds a trap
  reachable from a benign failure.
- **Do less** (INVARIANTS §1): every deferred row in §7 stays deferred until a named consumer
  demands it.
