# Stage 1 — external commands: `fork`/`exec`/`wait` for the shell

Stage 0 (PROCESS.md §10 / S7) proved a real command interpreter runs on the
`svm-posix` personality: redirection, pipelines, lists, variables, globbing,
`if`/`then`/`else`, and a dozen builtins, all differential-tested interp==JIT
(`crates/svm/tests/c_shell.rs`). Everything there is **in-process** — pipelines
stage through memfs temp files, and there are no child processes.

Stage 1 gives the shell **external commands**: run a program that is *not* a
builtin as its own confined child domain, deliver it `argv`, inherit stdio, and
collect its exit status into `$?`. This is `posix_spawn` + `wait` — sequential,
no fork-returns-twice yet.

## The substrate already exists

The child-domain machinery is built and CI-gated; Stage 1 is **personality
glue, not new substrate**:

- **`Instantiator` (iface 6)** — `instantiate_module` (op 5) spawns a child
  running a **separate host-verified `Module`** confined to a carve of the
  parent's window; `join` (op 1) parks the caller until the child completes and
  yields its result. Proven on both backends (`separate_module.rs`,
  `jit_separate_module.rs`), including data-segment materialization and
  confinement (a child cannot touch bytes outside its carve).
- **`Module` (iface 8)** — host-verified code a guest may instantiate. The host
  grants a `Module` capability (`Host::grant_module`); on the JIT the child is
  resolved through `svm_run::module_resolver` (never guest-reachable) and
  compiled **at instantiate** — §14's "nesting cost paid at setup".
- **Grants into children** — `instantiate_granted` (op 8) / `instantiate_named`
  (op 11) re-grant the parent's own capabilities (stdout/stderr/stdin) into the
  child, discovered by name via `cap.self.resolve` (`instantiate_granted.rs`,
  `instantiate_named.rs`). This is stdio inheritance.
- **argv delivery** — a child runs over the **parent's shared window backing**;
  the carve is not zeroed, so the parent seeds `argv` bytes into the child's
  carve before `instantiate` and the child reads them at low offsets. (The
  child's own data segments materialize over the carve at spawn, so `argv` goes
  where no segment lands.) Proven by `stage1_spawn_wait.rs`.
- **exit status** — `join`'s `i64` result is the child's return value; the
  personality maps it to POSIX's 8-bit `$?` convention.

## The mapping — BusyBox-multicall in miniature

A shell's `exec` and BusyBox's applet dispatch are the same shape: a program
image that, run with a different `argv[0]`, *is* a different command. In this
substrate an "external program" is a verified `Module`; running one is
`instantiate_module` + `join`. The shell's `PATH` becomes a **name → `Module`
map** the personality holds; command lookup is a map lookup; `exec` is spawn.

## Slice plan

0. **Unmodified `main(argc, argv)` on chibicc** *(done —
   `stage1_argv_main.rs`)* — the "as close to native as the security model
   allows" milestone. An **ordinary** C program — `int main(int argc, char
   **argv)`, `write(1, …)` to an *ambient* fd, no capability threading in the
   source — compiles through chibicc and runs with a real `argv`. chibicc's
   synthesized `_start` now parses the §3e powerbox args buffer at
   `POWERBOX_ARGS_BASE` (`{argc, envc}` + packed strings) into an `argv[]`
   pointer array parked at the entry SP, then calls `main(main_sp, argc, argv)`
   with `main`'s frame relocated a page above; writable globals shift past
   `POWERBOX_ARGS_END` so the seeded argv never collides with a global (both
   opt-in for a `main`-with-params, so every `main(void)` program — incl. the
   Stage-0 shell — is byte-identical). Modeled on svm-llvm's `synth_start_argv`
   (already vs-native there, but that frontend is an excluded LLVM-dependent
   crate; the self-contained demo needs chibicc parity). Output tracks argv,
   exit code = `argc`, differential interp==JIT; no confinement-path change. This
   is the crt that makes a compiled C program a **first-class bash command**.
1. **Spawn/wait spike** *(done — `stage1_spawn_wait.rs`)* — a differential
   (interp+JIT) test proving the core: a parent seeds `argv` bytes into a child's
   carve, `instantiate_module`s it, `join`s, and the child's return (a function
   of the seeded bytes) is the parent's result — with the child's output also
   readable from the shared carve. No shell yet; de-risks the mechanism.
   - **argv[] vector ABI** *(done — `stage1_argv_vector.rs`)* — pins the real
     `main(argc, argv)` marshalling: `argc` (i32), an `argv[]` pointer array, and
     a string blob laid in the child's carve. An applet reads `argc`, follows
     `argv[1]`'s pointer to its bytes, and echoes them to the granted stdout —
     proving pointer-array *indirection* (the output tracks `argv[1]`, not a flat
     read), status = `argc`, differential interp==JIT. This is the layout the
     personality's `spawn` will lay down.
2. **stdio-inherited child** *(done — `stage1_stdio_child.rs`)* — a same-module
   BusyBox-applet child inherits a granted `stdout` (`instantiate_named`, op 11)
   and echoes its parent-seeded `argv` to it: a real external `echo` — argv in,
   bytes out through inherited stdio, status back — differential interp==JIT.
   - **Foreign-program variant** *(done — `stage1_foreign_command.rs`)* — the
     general `exec` case: the command is a *separate* verified `Module` (a
     distinct binary), spawned via `instantiate_module` (op 5). Separate-module
     children have no stdio-grant op, so the shell uses the **parent-as-pager**
     model — the child writes output into its carve and the parent forwards
     those bytes (length = the child's `join` return) to its own stdout.
     Differential interp==JIT.
3. **multi-applet dispatch guarantee** *(done — `stage1_applet_dispatch.rs`)* —
   one binary carries several applets (`true`→0, `false`→1, `echo`→writes+3),
   and spawning a chosen entry yields that applet's own `(stdout, status)`. This
   is the substrate guarantee the shell's command dispatch rests on: look a
   command up, spawn its entry, thread its exit code into `$?`. Differential
   interp==JIT. The name→entry map itself is trivial glue and lands in slice 4.
   - **C-applet ABI** *(done — `stage1_granted_argv_applet.rs`)* — the applet
     receives its `stdout` as an *entry argument* (via `instantiate_granted`,
     op 8 — the handle is the child's 3rd arg) and writes through it, rather than
     resolving by name. This is the shape a **chibicc-compiled** applet must take:
     the frontend's generic capability import passes the handle as the *first C
     argument at runtime* (`codegen_ir.c` §7) and cannot emit `cap.self.resolve`,
     so `applet(inst, addrspace, stdout_h)` writing through `stdout_h` is the
     natural form. Proven with a seeded-argv echo, differential interp==JIT.
4. **exec a compiled-C command (pure status)** *(done —
   `stage1_exec_command.rs`)* — a "shell" parent spawns a *separate*, unmodified
   `int main(int argc, char **argv)` C program via `instantiate_module` (op 5),
   delivers `argv` through the §3e args buffer seeded into the child's carve, and
   `join`s for `main`'s return — the value a shell records in `$?`. The one
   enabler was a chibicc **`--child-entry`** flag: it emits function 0 with the
   §14 child ABI (`(i64 starter) -> (i64 status)`, `main`'s int widened) instead
   of the paramless top-level powerbox entry, so the program is spawnable while
   still parsing the args buffer into `main(argc, argv)`. Status tracks argv
   (real delivery), differential interp==JIT; **no new substrate op** (rides
   op 5 + op 1). Empirically found: an unmodified powerbox `_start` (`() -> i32`)
   ThreadFaults under `instantiate_module` — the child-entry signature is the fix.

4b. **exec a compiled-C command with inherited stdout** *(done —
   `stage1_exec_stdout.rs`)* — the full external `echo`. This closes slice 4's
   gap with a new substrate op, **`instantiate_module_named` (Instantiator
   op 13)**: the union of `instantiate_module` (op 5 — resolve + compile a
   host-granted `Module`, materialize its data into the carve) and
   `instantiate_named` (op 11 — re-grant caps into the child's powerbox by name).
   It is the only op that runs a foreign program **and** hands it capabilities,
   so a compiled command resolves an inherited `stdout` by name and `write(1, …)`
   lands in the shell's sink. The shell parent re-grants its own `stdout` under
   the name `"stdout"`; the command's `--child-entry` `_start` resolves it. Built
   on **both backends** (interp dispatch decode + JIT `instantiate_module_named`
   thunk, the JIT given both the module resolver and the named-grant hooks),
   differential interp==JIT, output tracking argv. **Authority-TCB, not
   escape-TCB (§2a): the D63/D38 carve masking is untouched** — op 13 is literally
   the union of two existing, fuzzed decode paths through the same confined-child
   spawn. Existing instantiate suites unchanged.

4c. **op 13 driven by a compiled-C shell** *(done — `crates/svm/tests/c_shell_exec.rs`)*
   — where 4b's parent was hand-written IR, here a **chibicc shell** (`main(argc,
   argv)`) parses its own powerbox args, looks the command up via a host fn,
   seeds the command's `argv` into a 128 KiB carve, lays a `"stdout"` grant record,
   and drives `instantiate_module_named` (op 13) + `join` through capability
   imports (`Resolved::CapBound`, the `Instantiator` baked in) — the whole
   external-command path emitted by the frontend, differential interp==JIT.
   This surfaced a **latent JIT/interp differential gap** (now fixed, no
   confinement code touched): the JIT's `lower_instantiator` demanded an
   *exact-width* Instantiator contract (i32 child handle / i32 `join` arg), but
   chibicc widens every scalar to an i64 slot (`int __spawn(...)` → `… -> (i64)`),
   so **no compiled-C program could drive the Instantiator on the JIT** — every
   op-13/op-1/op-5/op-8/op-11 call fell to a `CapFault` the interpreter never
   raised. The interpreter already tolerated the width (it reads args as i64 slots
   and coerces each result to the declared type, `slot_to_val`); the JIT now
   mirrors that with `slot_i64`/`slot_i32`/`result_as` coercions and a
   width-tolerant shape gate (scalar-int, arg-count) across every Instantiator arm.
   A non-scalar shape, too-few args, or unknown op still lowers to a runtime
   `CapFault` (never a compile-time rejection). Guarded portably by the
   `wide` arm of `stage1_exec_stdout.rs` (i64-declared op-13 result + join arg,
   hand-IR, no chibicc). *Follow-up:* folding this into the full `c_shell.rs`
   builtin dispatch (its personality-heap-at-`win/2` layout vs. a 128 KiB command
   carve).

### Power 2 — the endpoint direction (deferred, S9)

Op 13 is **forwarding** (capability model "Power 1"): a parent hands a child a
capability it already holds. That covers `cmd` writing straight to the terminal.
It does **not** cover the shell *intercepting* a command's output — `cmd > file`,
`cmd | other`, capturing into a shell buffer — because there the shell must not
forward the real `stdout` but **serve** the child's stdout with its own code
(capturing bytes, feeding a pipe). That is capability-model **Power 2**: a guest
minting a capability whose implementation *is* its own code, so a child's
`cap.call` on it parks and wakes the parent (§14 "the parent's own handler /
pay-for-what-you-virtualize"). The primitive is the **`Endpoint`** (PROCESS.md
§4, `[PROPOSED]`; S9 on the roadmap): `mint(sig) -> (serve_end, client_template)`,
`serve`/`reply`. A guest-served endpoint is what makes a parent a **personality /
kernel for its children** (parent-as-POSIX-kernel, parent-as-pager) — the
keystone of self-similarity. It is **not built**. Until it lands:

- **`cmd` → terminal**: op 13, forwarding the real stdout. **Done.**
- **`cmd > file`, `cmd | cmd2`** (shell-side interception): needs `Endpoint`
  (Power 2). The stopgap is the parent-as-pager model (`stage1_foreign_command.rs`
  — the command writes to its carve, the parent forwards), which requires a
  command written to output to memory rather than an ambient fd, so it is **not**
  a drop-in for unmodified compiled commands. Real redirection/pipelines of
  external commands wait on the endpoint work.

5. **`spawn` in the personality** *(substrate done — `crates/svm/tests/stage1_posix_spawn.rs`)*
   — `svm-posix` gained a minimal **`exec` surface**: a `PATH` registry (`name →
   Module` handle, `Posix::register_command`) reached by an `exec_lookup` op, plus
   an `exec_stdout` op handing back the forwardable stdout `Stream`. A compiled-C
   shell **running on the real personality** now dispatches an unknown command to
   a spawned external child instead of `<cmd>: not found`: it reads its own `argv`
   (`argc`/`argv`), looks the command up, and — the spawn being the shell's own
   `Instantiator` op 13 + `join` (`Resolved::CapBound`) — re-grants `stdout` by
   name and threads the child's `argc` into the status. The **two stdout models
   are unified**: `Posix::set_stdout_sink(host.shared_stdout())` routes the
   personality's fd-1 writes and the child's re-granted `Stream` writes into one
   `Host` sink, so shell and command output interleave. Differential interp==JIT,
   three paths (builtin / external / not-found). Confinement untouched (op 13 is
   the existing fuzzed spawn path; the personality is authority-TCB, §2a).

   **Folded into the full Stage-0 shell** *(done — `crates/svm/tests/c_shell.rs`)*:
   the real `c_shell.rs` shell now spawns external commands from its command
   dispatch — the `else` (was `<cmd>: not found`) branch does `exec_lookup` and,
   on a hit, `spawn_cmd` (grant record + args carve + op 13 + `join`), threading
   the child's status into `$?`. The layout tension is resolved as the focused
   test does: a 384 KiB `pool` static forces a window with a 128 KiB-aligned
   command carve **below the stack**, and the personality heap moves to the top of
   the window (the shell never `malloc`s, so it is never touched). `run_shell`
   takes an optional PATH of `(name, C source)` commands; with none registered
   `exec_lookup` always misses, so the 24 existing shell tests are unchanged. Two
   new tests cover a spawned command's argv delivery + `$?` and its status flowing
   through `&&`/`||`. A `>`/`|` redirect on an *external* command is not honored
   (the command always writes to the terminal sink) — that is the Power-2
   `Endpoint` gap below, not a regression.
6. **Pipelines across real children** — replace the memfs-temp pipeline staging
   with concurrent OS-thread children communicating through a granted
   `SharedRegion` + canonical-key futex (PROCESS.md §4 "revised async-children
   plan"). This is the jump from sequential spawn/wait to true concurrency.
   **[PROMOTED 2026-07-22 — an svm-owned todo, consumer-pinned.]** jacl (the
   first shell-like language targeting svm) needs concurrent stages soon after
   sequential; this does not wait for a further request. The remaining build is
   step 2 of the revised plan (OS-thread children in own guarded windows; the
   canonical-key futex, step 1, landed as S1b) — sequential-first, concurrency
   promptly after.

   **[BUILT 2026-07-23 — the substrate composition, reference interpreter.]**
   Two changes closed the gap, then the pipeline just composed: (1) the
   **cross-domain canonical futex key** — the interp's `FutexKey::Region` first
   field was the *per-window* region id (the S1b caveat), so two domains
   mapping one granted backing produced different keys and every wake missed;
   it now keys on the **backing allocation's identity**, closing the S1c
   residue. (2) **`SharedRegion` re-grants into spawned children** —
   `can_regrant`/`regrant_into_child` alias the same backing into the child's
   powerbox (the pipe-end pattern), so op-8/11/13 named grants (and detached
   spawns) carry the data plane; previously only coroutine children could
   receive a region. Pinned by `svm-interp/tests/concurrent_stages.rs`: a
   parent mints a region, spawns a producer and a consumer as separate carves
   (own windows, own address spaces), and the two move four items through a
   **one-slot bounded ring** (flag + datum in the region), parking on the flag
   and waking each other — the shape sequential spawn/wait deadlocks on
   outright. Regressions surface loudly (30 s wait timeouts fold into the
   result ×1000), and the run completes in well under a second — real wakes.
   Also pinned: the same ring between two **detached** stages (op-15 spawns —
   private windows + the explicit shared channel compose, the §5 model
   sentence as a test) and bytecode-entry equality via the standing oracle
   fallback.

   **[BUILT 2026-07-23 — the JIT pipeline.]** The fast backend runs the same
   ring pin (`svm/tests/jit_concurrent_stages.rs`, byte-identical module,
   differential vs the interp) — and the deep PAL turned out to be *already
   built*: `MprotectWindow::map_region` does real aliasing (memfd
   `mmap(MAP_SHARED|MAP_FIXED)` on unix, placeholder + `MapViewOfFile3` on
   windows) against any `GuestWindow`, the child powerbox host is already the
   child's baked `cap.call` ctx (so `map` reaches its regranted region), and
   the region-canon hook self-installs per child window at the child's first
   thunk call. What was actually missing: (1) **granted children ran
   synchronously** — ops 8/11/13 now take the S1c OS-thread path (uncached
   per-spawn code; the powerbox host is released from the child's thread via
   `run_child_code_then`'s teardown hook, *before* the child window's VA is
   freed, so the host's region-canon purge can never erase a reused address's
   fresh entries); and (2) **children had no futex** — a nesting module's run
   now always stands up the thread `Domain` (futex-only when the module has no
   `thread.*` ops: `Env.call_tramp` became optional), the nursery carries its
   address, and child compiles get a wait/notify-only `ThreadEnv` over the
   **parent's shared futex table** (spawn/join/fibers stay rejected;
   `Func::uses_fibers_or_threads`/`uses_futex` split `uses_concurrency`).
   Spawned children register in the domain's live count so wait/join deadlock
   detection stays sound.

   **[BUILT 2026-07-23 — the `c_shell` personality `|` wiring.]** The Stage-0
   shell's `run_pipeline` now runs eligible pipelines **concurrently over
   rings** (`crates/svm/tests/c_shell.rs`): when every stage after the first is
   a pure filter and the `__stage` runner is on PATH, the shell mints one
   region per `|`, spawns each later stage as an op-13 child of the runner
   (grants: `stdout` + input ring + output ring by name), runs stage 0 *in the
   shell* (full builtin power — files, redirects, globs) pumping into ring 0
   through its own mapped alias, and joins the children; the status is the
   last stage's, as in bash. The ring is a byte SPSC FIFO in the region
   (head/tail/done/reader-closed words + data), parking on real futexes with
   the loud 5 s-timeout bail; `head`'s early exit sets reader-closed so its
   producer stops (SIGPIPE-lite) instead of wedging. The `__stage` runner is
   an ordinary `--child-entry` C program — grants discovered by `cap.self`
   reflection, rings mapped with the `__vm_region_*` builtins into its own
   256 KiB window, filters (`cat`/`grep [-v -c]`/`wc`/`head`/`tail`/`sort`/
   `uniq`) byte-matching the shell's builtins — and holds no memfs capability
   at all, which is exactly why only pure filters ride the ring path:
   anything else (redirects, `$`/glob tokens, file args, >4 stages, no runner
   registered) **falls back** to the sequential memfs-temp staging with
   identical semantics. Zero substrate change — the slice is personality glue
   over #422/#424's regions-into-children + async op-13 children + shared
   futex. Pinned differentially interp==JIT: a 3-stage `cat|sort|uniq`, a
   4-stage pipeline with ring→ring middles, `grep` status into `$?` from a
   ring child, `grep -c`, `head` early-exit, and the redirect fallback
   coexisting with ring runs in one script. (Found on the way: a chibicc
   indexed-post-increment-store miscompile, worked around and logged —
   ISSUES.md I35.) **Remaining:** redirects on external commands stay
   Power-2-gated (the Endpoint work), below.
7. **`fork`/`clone`** — the parked-domain clone path (PROCESS.md §7), the last
   piece for shells that fork *themselves*.

Security posture is unchanged: children keep their **own guarded windows**; the
D38 confinement lowering (the most sensitive code in the tree) is not touched.
Stage 1 only *composes* existing, fuzzed primitives.

## Resolved — `poll` is WNOHANG; its terminal status is backend-portable

A crashing command must not crash the shell, which needs `poll` (op 9:
`0` running / `1` returned / `2` trapped) to detect a trapped child and
`detach` instead of `join` (a `join` propagates the child's trap to the
parent). The original concern was that an *immediate* `poll` after a
synchronous spawn reads differently across backends: the interpreter's M:N
scheduler defers a child (so an immediate `poll` reads `0` running); the JIT
runs it eagerly on its own OS thread (so an immediate `poll` may already read
`1`/`2`).

That is the **defined semantics of a non-blocking probe (WNOHANG)**, not a
divergence: `0` ("not done yet") is a valid answer at any time, and a caller
does not control how many `0`s it sees first. Making the interpreter "eager"
cannot converge the immediate poll for the deterministic single-worker configs
anyway (the JIT is 1:1 OS-thread; a single-worker interp run has no thread to
run the child ahead of the parent). The **portable idiom** is to loop `poll`
(yielding the worker between probes) until non-zero; the **terminal** value is
identical across backends — `1` returning, `2` trapping. That is now pinned by
`crates/svm/tests/lifecycle_poll_convergence.rs` (interp vs JIT, both cases).
See ISSUES.md I43. The `$?` = 128 + signal crash-status mapping is a shell/guest
convention (not a substrate contract) and remains guest-personality work.

## Shipped: the Stage-0 shell in the browser playground

The Stage-0 shell now runs as an **interactive card** in the browser playground — type a script,
click Run, see its output, client-side in the sandbox. This is the same shell `c_shell.rs`
differential-tests; the browser plumbing lives in the detached `svm-browser` crate (see
`BROWSER.md` → "POSIX-personality on-ramp"). The pieces:

- **`svm-posix` reaches the browser runtime.** `svm-posix` is a dependency of the `svm-browser`
  cdylib; `svm_run_shell` / `posix_shell_exec` grant the personality and run the shell module on the
  **bytecode** engine (the tree-walker uses OS threads + a wall clock, absent under wasm), with the
  editor text as stdin and the personality's captured stdout as output.

- **Sequential subset only.** The browser's bytecode compiler rejects `Instantiator`/`SharedRegion`
  cap.calls, so the browser fixture is built with `-DSVM_SHELL_SEQUENTIAL`: `#ifndef` guards in
  `shell_main.c` drop the **external-command spawn** (slice 5, above) and the **concurrent ring
  pipelines** (slice 6). What ships is the full Stage-0 surface — builtins, redirection, in-window
  memfs pipelines, `;`/`&&`/`||` lists, `if`, variables, globbing, and `#` comments. External
  commands and concurrent pipelines stay native-only (they need the caps bytecode can't compile);
  the full shell keeps both.

- **One source of truth.** The shell C source lives in `crates/svm-run/demos/shell/*.c`
  (`shim.c`/`ring.c`/`shell_main.c`), `include_str!`d by `c_shell.rs` and compiled into the committed
  `browser/tests/fixtures/shell.svmb` by the (ignored) `gen_browser_shell_fixture` test — reusing the
  same chibicc compile + by-name import resolution the differential uses.

- **The 64 KiB-page fix.** A chibicc `main(void)` guest with read-only data faulted in the browser
  until chibicc gained a **`--data-page 65536`** flag (D40 / RO-vs-writable isolation at the wasm
  64 KiB page). Detailed in `BROWSER.md`; it unblocks *any* chibicc guest for the playground, not just
  the shell.

Tested: native `c_shell` (differential interp==JIT) + `browser/tests/shell.rs` (bytecode) +
`browser/browser-shell-test.mjs` (real Chromium over the wasm, in CI). External commands, concurrent
pipelines, and `fork` (items 5–7 above) remain the native-only frontier.

## Remaining work — from the Stage-0 shell to *actual bash* in the playground

The card above is a **hand-written Stage-0 shell**, not GNU Bash. Getting real bash — the ~150k-line C
program, not something bash-like — into the playground is two problems stacked: the **native** path
to bash (mostly captured in `PROCESS.md`), *plus* a **browser-only** constraint that is not captured
anywhere else and is the harder, partly-open half.

### The native path to bash (see PROCESS.md)

Bash is a different architecture from the current shell: where this shell mints **confined children
and grants them capabilities** ("children born destitute", the `Instantiator` model), bash assumes
ambient **`fork()` + `exec()` + `wait()`**, signals, and job control. The roadmap for that lives in
`PROCESS.md` (the S8/S11/S12 stage table, the §7 fork plan, the POSIX-coverage census, the L0/L1/L2
signals ladder). In dependency order:

1. **Guest libc for bash** — assemble bash's link surface (`glob`/`fnmatch`, regex, `getpwnam`/
   `getgrnam`, `setjmp` — proven — on top of the Postgres shim set). *Medium; incremental.*
   **[Slice 4a done 2026-07-29 — the two-worlds bridge.]** The POSIX personality (the process/fd/signal
   ABI of slices 1–3) lived only in the chibicc/name-binding world; the LLVM on-ramp reaches host caps by
   name via `__vm_cap_resolve` + `__vm_host_call` (the `fs`-cap idiom). `svm_run::posix::posix_cap`
   (over new `svm_posix::cap`) now exposes the personality as a named powerbox `HostCap`, so an on-ramp
   guest resolves `"posix"` and drives `pipe`/`dup2`/`posix_spawn`/`waitpid` through it — proven
   cross-backend (interp==bytecode==JIT) by `crates/svm-llvm/tests/posix_cap.rs` (a compiled-C mini-shell
   spawns a child with fd inheritance). This is the substrate a bash-on-LLVM `proc_shim` will call
   instead of the Postgres demo's inert process stubs. *(A program that reaches the host only through
   `__vm_host_call` still needs a standard-libc call — e.g. `printf` — to trip svm-llvm's
   `needs_powerbox_entry` and get the synthesized `_start`; a real shell links libc, so it always does.)*
2. **Build `bash.svmb`** — autoconf cross-config for the svm "platform", `--noediting`, through the
   `clang → svm-llvm-translate` on-ramp (S8), the same path Postgres/SQLite/QuickJS already ride.
   *Medium/mechanical.* Gets bash to *link and start*.
3. **`fork` (the gate)** — the substrate has **no fork-returns-twice**. The plan (§7 / Stage 3 / S11)
   implements it at the **personality** level over durable freeze → clone-window → thaw, with two hard
   prereqs: **R8 closure** (durable `call_indirect` to may-suspend targets — bash dispatches builtins
   through function-pointer tables, so R8 is squarely on fork's critical path) and a
   durable-instrumented build + full window copy (CoW later). *Large; the single biggest item.* Until
   it lands, bash runs only a fork-free subset — effectively no real script (`ls | wc` already forks).
4. **`exec`/`wait`/`waitpid`/`pipe`/`dup2`** wired to the child machinery. *Medium.*
   **[Slice 1 done 2026-07-29 — the fd surface.]** `pipe`/`dup2`/`dup`/`fcntl` landed as real
   personality ops (POSIX.md ops 23–26) over a generalized fd table (stdio `0`/`1`/`2` are now ordinary
   sentinels, so `dup2(pipe_w, 1)` redirect and `close`/reuse of a stdio fd work as POSIX). Pipes are
   intra-personality (a single guest's two ends share one buffer, non-blocking).
   **[Slice 2 done 2026-07-29 — spawn/wait + fd inheritance.]** `spawn`/`waitpid`/`wait` landed as
   personality ops (POSIX.md 27–29). A spawn is *authority* the libc personality does not hold, so the
   instantiate+run is an **embedder-wired delegate** (`Posix::set_spawn`) — opt-in like the stdout
   `Stream`, `-ENOSYS` unwired. The child **inherits the caller's fd 0 and fd 1**: `spawn` drains the
   current fd-0 binding as the child's stdin and routes its captured stdout to the current fd-1 binding,
   so a `dup2(pipe_w, 1)` / `dup2(file, 1)` redirect before the spawn lands the child's output exactly
   where POSIX would (proven cross-backend). **[Slice 2.5 done 2026-07-29 — end-to-end with a real
   child.]** A **compiled-C shell** now drives a **separate compiled-C command** through `spawn`/`waitpid`
   (`c_posix_spawn.rs`, interp==JIT): the embedder wires the spawn delegate to instantiate + run the
   child domain on its own Posix personality, and the child's uppercased stdin output flows to the
   shell's fd 1 while its exit status returns through `waitpid`. (The delegate is the test embedder —
   promoting a reusable builder into `svm-run` waits on `svm-run` gaining an `svm-posix` dep, deferred
   until a second consumer needs it.) **Remaining:** `fork`/`vfork`/`execve` (return-twice /
   image-replace) on the durable-clone capstone.
5. **Signals** — L0 doorbell (a word bash polls at command boundaries; exact for `trap`, ships
   cheaply) → L1 interruptible parks → L2 safepoint handlers (Ctrl-C a running loop; parked, S13).
   **[Slice 3 done 2026-07-29 — the L0 doorbell.]** `signal`/`kill`/`sigcheck` landed as personality
   ops (POSIX.md 30–32, cross-backend). A raised signal (guest `kill`, or the embedder's
   `Posix::raise_signal` for a terminal `^C`) sets a pending bit; the guest polls `sigcheck` at a safe
   point and it returns the installed handler pointer of the lowest pending **caught** signal (ignored
   and default dispositions dropped) — exact for `trap`, no async interruption. **Remaining:** L1
   interruptible parks + L2 safepoint handlers (interrupt a running loop; default actions), parked S13.
6. **Job control + terminal** — process groups, `tcsetpgrp`, SIGTSTP/CONT, and readline/termios for
   interactive use. Deferrable behind `--noediting` (batch bash is still real bash); the terminal is
   its own large frontend effort.

Roughly ~80% of that is "solved-class" work (compile + libc + wiring, de-risked by Postgres/SQLite
already running); the ~20% that gates everything is **`fork` + R8**.

### The browser-only constraint — and how much of it is *already* there

Running bash *in the playground* is **not** just "native bash + compile for wasm": the playground is
**bytecode-only** (the tree-walker uses OS threads + a wall clock, and the wasm-JIT tier is a
hot-compute tier only). But the gap is **narrower than it first looks** — the bytecode engine's
cooperative, single-thread driver (`compile_and_run_with_host`, the exact entry the browser shell runs
on) **already lowers and drives the core §14 ops, wasm-safely:**

- **`instantiate` (op 0), `instantiate_module` (op 5), `join` (op 1) already work on bytecode,
  single-thread.** Op 0 was covered by `bytecode_parallel_instantiate.rs` (cooperative arm); op 5 —
  the "exec a command" primitive, a *separate-module* child — is pinned matching the tree-walk oracle
  by `bytecode_separate_module.rs`. No OS threads, no clock. So **sequential spawn/wait is already a
  browser capability**, not a missing one.

What the browser bytecode engine still **declines** (the real remaining gaps, in rough order):

1. **`instantiate_module_named` (op 13)** *(done — parity slice 1, `bytecode_instantiate_named.rs`)* —
   exec-with-inherited-stdout, what the shell's external-command path uses. Lowered in `compile_inst`
   to `Op::InstantiateModule` with a `grants: Option<(ptr, n)>` field (op 5 = `None`); the cooperative
   `drive` arm reads the by-name grant records from the parent window and builds the child powerbox via
   the shared `Host::spawn_named_child`. Unblocks *external commands* in the playground shell.
2. **The ring / `AddressSpace` / `SharedRegion` ops + concurrent stages** *(done — parity slice 2,
   `bytecode_concurrent_stages.rs`)* — the full concurrent ring pipeline (the tree-walk oracle's
   `concurrent_stages.rs`, same 410) now runs on the cooperative bytecode driver. The gap turned out
   **narrower than "lower the region ops"**: `SharedRegion.map`/`page_size` (ops 0/3) and
   `AddressSpace.create_region` (op 5) already ride the generic `cap.call` dispatch (they service from
   `(Host, Mem)` alone — only `SharedRegion.grant` op 4 needs a child-powerbox seam, and the ring path
   doesn't use it), and the production `drive` scheduler was **already** a cooperative multi-vCPU
   scheduler that parks a task on `memory.wait` and wakes it on `notify` (so item 3 below was already
   built). Two real changes closed it: (a) **`instantiate_named` (op 11)** — `instantiate` (op 0) plus a
   by-name grant list, the same-module twin of op 13 — lowered by extending `Op::Instantiate` with the
   same `grants` field, and `scan_seams` corrected to classify ops 11/13 as `has_instantiate` (not
   `has_coro`, which with `memory.wait`'s `has_thread` tripped the `has_coro && has_thread` veto → whole-
   module fallback); and (b) the drive's futex wait/notify now key on the **backing identity**
   (`Mem::futex_key`, `FutexKey::Region`) computed against the **waiting task's own confined window**
   (`extra_envs`), not the raw window offset against the root `mem` — so two children that mapped the
   same `SharedRegion` into separate windows rendezvous (S1c), instead of a child re-reading an
   unrelated root byte and spinning. Confinement (D38 masking) untouched — authority-TCB only (§2a).
3. **True concurrency** *(already built — folded into slice 2)* — the cooperative `drive` scheduler
   already interleaves live children on one thread with a logical clock (`TaskState::BlockedWait` parks,
   `notify`/child-completion wakes, a stuck set advances the clock, deadlock → `ThreadFault`); a
   confined `instantiate` child runs as a task slot exactly like a `thread.spawn` sibling. The earlier
   "runs a child to completion synchronously" note was stale for this driver. `poll` (WNOHANG) is the
   one remaining nuance and is already backend-portable (see the `poll` note above).
4. **`fork`** — durable clone over the `Instantiator` child model (§7). Whether the durable
   freeze/thaw path itself runs on the bytecode engine in wasm is the open question that gates *bash*
   specifically; slices 1–2 are the parity groundwork it sits on.

So this is a **sequence of scoped slices on the bytecode engine**, not an unbounded design problem:
op 13 → ring ops + op 11 → (scheduler, already built) → fork. The wasm-JIT tier can defer each §14 seam
to the bytecode engine (as it already does for its cross-tier helpers), so bringing bytecode to parity
brings *both* browser engines along. Slices 1 and 2 are done; **`fork` is the next frontier**.
