# Rust `std` on the svm-llvm on-ramp — design & tracking

The plan, the decisions, and the prioritized slices for running **full-`std` Rust as an svm
guest** through the LLVM on-ramp, bound to the **POSIX personality** (`svm-posix`) — not WASI.
The analog of `NIM.md` for this workstream; fold completed sections into `DESIGN.md`/`LLVM.md`
and drop this file once the actionable gaps close (the repo convention, cf. the former
`WASM.md`).

> Status: **real `std` runs on svm (S0 + S1a + S1b core done, 2026-08-11).** A `std`
> binary — `lang_start` + heap `Vec` + iterators — built for `x86_64-unknown-svm` via
> `-Zbuild-std` (`crates/svm-llvm/rust-svm/`), translated through the on-ramp, runs on the
> powerbox to the correct computed exit code, byte-identical to native
> (`crates/svm-llvm/tests/std_guest.rs`). Getting there took **one on-ramp change** — parsing
> call operand bundles (§9, S1a) — plus the build lane; malloc-synth + `lang_start` worked as-is
> off the bin's `main`. **Remaining for S1: the PAL** so actual I/O (`println!`, args) works
> over `__vm_host_call`; pure-compute `std` is already live. Route decision in §1/§4.

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

- **A. Custom `std::sys` platform ("PAL") for an svm target — CHOSEN, S0 done.** A target JSON
  (`x86_64-unknown-svm.json`, `panic-strategy: "abort"`, `singlethread: true`) + `-Zbuild-std` +
  a small overlay patched into the toolchain's `rust-src`. **S0 built `std` this way**
  (`crates/svm-llvm/rust-svm/`): the PAL (`sys/pal/`) already has a clean `_ => unsupported`
  fallback, so the overlay is just **five one-line `cfg_select!` arm additions** routing
  `target_os="svm"` to the minimal no-OS/single-thread leaf-module impls (the same ones
  `vexos`/`zkvm` use, in `sys/{alloc,thread_local,random,io/error}`) plus one ~85-line
  allocator `imp` forwarding to the C `malloc` family. The PAL talks `__vm_host_call` directly
  (§3), so struct layouts are **ours** (no reconciliation): `stat` is svm-posix's
  `{st_mode, st_size}`, timestamps are whatever the Clock op returns. Cost: a rust-src overlay
  patch to rebase when the pinned nightly moves — contained (13 added lines), mechanical, and
  entirely in guest/untrusted code.
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
- **D. `restricted_std` (custom target, *unpatched* std) — FOUND INSUFFICIENT.** The original
  plan was to use `restricted_std` as a no-PAL S0 smoke test. **S0 disproved this for a novel
  OS:** modern std's leaf modules (`sys/{alloc,thread_local,random,io/error}`) enumerate
  `target_os` with **no catch-all**, so an unknown `os=svm` fails to *build* with missing-symbol
  errors — even though the PAL itself falls back to `unsupported`. The minimum buildable overlay
  is therefore the five `svm` leaf-arms + one alloc `imp` (Alternative A, now done). This is why
  S0 and S1 partly merge: "std that builds but errors at runtime" is not free for a new OS. The
  overlay still uses the `unsupported` PAL for I/O, so S0's std *runs* only pure compute; real
  stdio/fs/args wait for the S1 PAL.
- **E. No-`std`-at-all forever (status quo) — REJECTED** by the premise: `no_std + alloc`
  blocks every crate that touches `std::io`/`fs`/`time` even incidentally, which is most of
  crates.io — the breadth D54 exists to buy.

## 5. ABI pins (decide once, up front)

| Pin | Value | Why |
|---|---|---|
| Panic strategy | `panic=abort` (target JSON + `-Zbuild-std=std,panic_abort`) | Sidesteps unwinding entirely; EH (`invoke`/`landingpad`) is planned substrate, not built (`LLVM.md` §"setjmp/longjmp + C++ EH"). Panic *messages* still format and print via the PAL's stderr before the abort → trap. |
| Threading | **Single-threaded v1** via `singlethread: true` in the target JSON (the `zkvm` posture: single-thread, atomics still available). Selects std's `no_threads` sync (`Mutex` = `Cell`) and static TLS automatically. `thread::spawn` → unsupported `io::Error` | The one big deferrable. svm has a threading model (`THREADS.md`) but wiring std's thread/futex/TLS backend is its own project (S5). **S0 note:** `singlethread` is load-bearing — without it std sees `target_has_threads` and `compile_error!`s the `no_threads` impls. |
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
| **S0** | ✅ **DONE (2026-08-11).** Target JSON + nightly `-Zbuild-std` + the minimal svm overlay (`crates/svm-llvm/rust-svm/`); `std` compiles for `os=svm`, and a fat-LTO'd pure-compute crate emits one self-contained `.ll` whose undefined externals are all on-ramp-supported. | overlay (5 leaf-arms + alloc `imp`); asset-lane CI check still TODO (I55) | The build lane exists; the route is validated. **Surfaced the first S1 gap** (parser, §9) rather than the anticipated `llvm.trap` one (that intrinsic is already handled). |
| **S1a** | ✅ **DONE.** On-ramp parser accepts call **operand bundles** (`[ "nonnull"(…) ]` on `llvm.assume`) — `skip_call_trailing` in `ll/parse.rs`, the real first gap (§9). | — | Std IR parses past the panic/assume machinery; unit-tested, no regression. |
| **S1b** | ✅ **core DONE.** Entry/powerbox for std works **as-is** off a bin's C `main` — no on-ramp change needed: malloc-synth + `Memory` grant + `lang_start` all fire. A pure-compute `std` bin (`Vec` + iterators, computed `ExitCode`) runs on the powerbox byte-identical to native (`tests/std_guest.rs`, gated on the build-std lane). | S1a | Real `std` (lang_start + heap) runs on svm. |
| **S1c** | **The PAL for real I/O**: a small `std::sys::svm` (replacing `unsupported`) whose `stdio`/`args`/`exit` call svm-posix ops (0/1, 17/18, 4) via `__vm_host_call`; then `println!`+`process::exit`+`env::args`. | S1b; posix `write`/`exit`/`argv` (present) | **First `std` hello-world byte-identical to native** — the `check_powerbox_vs_native` analog. Also the first exercise of the §9 `op`-constant-folding assumption. |
| **S2** | PAL `env` (11/12, setenv decision §6) + `time` (new Clock op) | clock op | `env::var` round-trip + `Instant` monotonicity vs native. |
| **S3** | PAL `fs`: `File` open/read/write/seek/close, `metadata`, `read_dir`, `remove_file` (ops 5–16) | op audit (§6) | Full-file-I/O program over the memfs, byte-identical vs native on a real dir tree. |
| **S4** | `HashMap` end-to-end (fixed-seed `RandomState`; optionally the getrandom op) | — | Retires the `NIM.md` §3e `RandomState` blocker; a `HashMap`-heavy program byte-identical to native. |
| **S5** | *(deferred, separate designs)* threads/sync/TLS; unwinding; net; `Command` | new ops + EH substrate | Each gets its own slice plan when a consumer names it. |

**W5 payoff (cross-link):** S1–S4 is exactly the surface `svm-leng` needs to run as a `std`
crate *without* the `no_std + alloc` rework `NIM.md` §3e slice 2 currently assumes — the
fixed-seed hasher lands here as the PAL default instead of a per-crate patch. Re-scope that
slice when S4 lands.

## 9. Risks — S0 results and what's now live for S1

- **First real gap — call operand bundles — FIXED (S1a, 2026-08-11).** The `expected
  \`%dest =\`, found LBracket` failure was *not* the packed-struct globals (those parse fine);
  it was **operand bundles** on `llvm.assume`: `rustc` emits `call void @llvm.assume(i1 true) [
  "nonnull"(ptr %p) ]`, and the parser didn't consume the `[ … ]` after a call's arg list, so
  the leading `[` desynced it into the next instruction. Fixed by `skip_call_trailing` /
  `skip_operand_bundle` in `ll/parse.rs` (bundles are optimization/annotation hints the on-ramp
  doesn't lower → parse-and-drop; unit test `call_operand_bundles_are_dropped`, 326 translate
  tests still green). Benefits every LLVM frontend, not just std.
- **Entry/powerbox — RESOLVED, no on-ramp change (S1b).** The `malloc` undefined-call stop
  seen on the S0 *lib* probe (`need_malloc = needs_malloc && has_main`, `lib.rs:528`) was purely
  an artifact of building `#![no_main]`. A real std **bin** emits the C `main` the powerbox
  recognizes (arity 3 = `main(argc, argv)`), so `_start` synthesis, the `Memory` grant, and
  malloc-synth all fire, and `lang_start` runs on top. A pure-compute std bin now runs to the
  right exit code on the powerbox (`tests/std_guest.rs`). No `lang_start`-specific work was
  needed — the earlier "lang_start surface" risk is retired for the compute path.
- **`llvm.trap`/panic machinery — NOT a risk (S0 cleared it).** The LTO'd std module's only
  undefined externals are `malloc`/`free`/`realloc` (synth, slice S) and intrinsics the on-ramp
  already handles: `llvm.trap`, `memcpy`, `sadd.with.overflow.i64`, `umax`/`umin.i64`,
  `vector.reduce.add.*`, `lifetime`/`assume`/`noalias.scope.decl` (dropped). No new intrinsic
  lowering is needed for the pure-compute path.
- **`op` constant-folding (still to verify in S1).** If rustc outlines a shared host-call
  helper, `op` reaches the call site as a runtime value → clean `Unsupported`. Mitigation
  pinned in §3 (per-op `#[inline(always)]` wrappers); S1 asserts it holds at `-O2` **and**
  `-O0`. (Not yet exercised — S0's program does no host calls.)
- **`lang_start` surface (S1).** `std::rt` init compiles as ordinary guest code; S1's
  hello-world is the existence proof. Unknowns land as `unsupported`-PAL errors, which fail
  loud, not wrong (INVARIANTS §9's decline-never-diverge, applied to a runtime).
- **Nightly rebase cost — measured small.** The overlay is **13 added lines across 4 files +
  one `imp`**, sitting next to the long-lived `vexos`/`zkvm` arms; `apply-overlay.sh` is
  idempotent and dry-run-verified. The CI asset-lane check catches drift.
- **CI drift (I55) — still open.** The guest-std build lives outside the workspace; without an
  asset-lane check it rots silently. The lane check is the remaining S0 loose end — land it
  with S1's first translating artifact.

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
