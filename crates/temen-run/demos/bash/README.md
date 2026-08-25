# bash — GNU bash on the LLVM on-ramp (#802)

The bring-up of the umbrella's (#794) target program: **the literal GNU bash source**, compiled
through the on-ramp as one whole-program module, hosted by the temen-posix personality this tree
built for it (fork #863, signals #796, job control #798, pipes #972, exec + `/bin` #801, the
controlling terminal #797, `longjmp`-from-a-handler #802-slice-1).

## Slice 2 (DONE) — translate + verify

`./build_bitcode.sh`: fetch bash 5.2.21 (fetched-not-vendored, GPLv3) → configure the bring-up
config → **native oracle build** (also generates y.tab.c and the `.def`-built builtins) → per-TU
bitcode with each Makefile's own flags (152 TUs: the link-line objects + libbuiltins/libglob/
libsh/libhistory(hist* only — the rest are readline standalone shims that duplicate bash's own)/
libtilde) → `llvm-link` + `bash_shim.c` + the reused waist → **translate (~0.8 s), verify:
clean**. Gate: `demo_bash_translates_and_verifies` (temen-llvm `translate.rs`, `#[ignore]`d
for wall-clock).

The bring-up config (`configure` flags, each with a reason in the script): `--without-bash-malloc`
(the waist malloc, not sbrk), `--disable-readline` (non-interactive first; interactive rides the
#797 terminal in slice 4), `--disable-nls`, `--disable-net-redirections` (no sockets), and
`ac_cv_type_long_double=no` (the printf builtin's `%Lf` would need x86_fp80 — denying the type
keeps `floatmax_t = double` in guest AND oracle). **Job control stays on.**

## Slice 3 (DONE) — first run: `bash -c` differential vs the oracle

The OS lane: the embedder grants the temen-posix personality as the **named "posix" capability**
(`temen_run::posix::posix_cap` + `run_with_caps` — the `posix_cap.rs` idiom); the shim's **band 0**
resolves it once (`__vm_cap_resolve("posix")`) and defines the real libc entry points
(open/read/write/stat/dirs/signals/termios/process ops) over `__vm_host_call` op dispatch,
marshaling C conventions (NUL strings, glibc struct layouts) to the op ABI. No translator lane was
added — this is the same named-cap route `posix_cap.rs`/`fs_cap.rs` already prove. bash runs the
**interpreter** (setjmp/fork are interp-only tiers). Gate: the run half of
`demo_bash_translates_and_verifies` — five `bash -c` scripts (echo/vars/arithmetic/for/functions),
stdout + exit byte-compared against the native oracle under identical argv/env.

## The gap-walk (the Tcl discipline: every gap gets a pinned unit test)

1. **`align 4294967296`** — clang stamps the max alignment (2^32, one past `u32`) on
   deliberately-trapping null stores (bash's `programming_error`). The `.ll` parser now saturates
   an alignment literal instead of refusing the module. Pin: `align_u32_max_saturates`.
2. **Old-C call-site drift** — bash's empty-parens prototypes (`extern void f ();`) let call
   sites invent their own function types: `add_unwind_protect(fn, 0)` is typed
   `(ptr, i32, ...)` at the site against a plain `(ptr, ptr)` definition. The native ABI hides
   the drift; the lowering now follows the **definition** for direct calls — arity split,
   va-area deposit only for a genuinely variadic callee, and integer args coerced to the
   definition's widths. Pin: `old_c_call_site_drift_follows_the_definition`.
3. **Old-C INDIRECT call-site drift** — the function-pointer twin: `typedef int Function ()`
   tables call cleanups through `(ptr, ...)` sites whose runtime target is a plain `void ()`
   (`add_unwind_protect(pop_stream, NULL)` → `(*cleanup)(arg)`). The strict typed `call_indirect`
   trapped `IndirectCallType` (a pinned security contract — never loosened). The translator now
   routes a **varargs indirect site** (the old-C unspecified-params marker; ANSI code never has
   one) through a synthesized **static dispatcher**: a funcref-equality chain direct-calls each
   address-taken candidate with its definition's own signature (args width-coerced, missing result
   padded 0), and everything else — exact-typed targets, real variadic targets, unknown funcrefs —
   falls to the strict `call_indirect` unchanged. CFI is never widened (every arm is a direct call
   to a statically-named function). Pin: `old_c_indirect_call_drift_dispatches_to_the_definition`.

Shim-side stub walk to first run: `qsort` (the tcl_shim heapsort — the on-ramp does not
synthesize it), `strerror` (glibc-matching strings — bash prints them in error messages the
differential compares), `strnlen`/`strdup`/`strncpy`/`strcat`/`strstr`/`strcasestr`/`strchrnul`,
`imaxdiv`, and the wide-char band (`mbsrtowcs`/`wcs*`/`isw*`/`towlower`/`wctype` — ASCII,
MB_CUR_MAX = 1).

## Slice 4 (fork/pipes rung DONE) — command substitution, subshells, builtin pipelines

bash **forks** on this lane: `` echo `echo nested` ``, `$(…)` (multiple, multi-line), `(subshell)`,
and builtin pipelines (`echo a | { read x; …; } | …` with trailing commands) all byte-match the
oracle. What it took (each pinned):

1. **The core-pipe builtins on the on-ramp** — `__vm_pipe`/`__vm_read`/`__vm_write`/`__vm_close`
   now lower exactly as the chibicc frontend's (`codegen_ir.c`), so band 0's #972 tag-protocol
   wrappers complete (interpreter tier — `CAP_SELF_PIPE` needs the `Real` scheduler). Pin:
   `core_pipe_builtins_roundtrip`.
2. **`posix_cap` grants everything `grant` grants** — the async-signal door (which carries the
   #799 caller-request door: without it `fork` was `-ENOSYS`), the #972 exec-remap hook, the #801
   op vtable (temen-run `posix.rs` + the new `temen_posix::cap_signal_source`/`cap_exec_remap_hook`/
   `cap_vtable`).
3. **A pristine guest-JIT grant forks** — the fixed powerbox prefix mints an (unused) `Jit`
   domain on every temen-run host; `fork_powerbox` refused it wholesale. A domain with no units, no
   installs, no native ctx now duplicates as an equally-empty grant (same quota, same index);
   live JIT state still fails closed.
4. **`fork_private` copies the `vm_map`-committed tail pages** — the waist malloc's heap lives in
   the reserved tail, so every malloc-using on-ramp program's fork was refused (`-EAGAIN`, which
   bash surfaces as `fork: retry`). Plain `Rw`/`Ro` tail pages now copy page-wise into the twin;
   `Backed` (§13) stays fail-closed. Pin: `fork_private_copies_vm_mapped_tail_pages`.
5. **The any-child blocking wait benches** — bash's no-job-control `waitchld` blocks in
   `waitpid(-1, …, 0)`; the #799 bench covered only specific pids, so the parent raced past its
   unfinished subshell/pipeline (the `end` before `in-5`'s newline diff). Op 28 now benches on
   the lowest live core-twin child for `-1` (correct for foreground waits — the caller loops
   reaping until all children are gone; the true any-child park key is a later rung).

End-to-end pin: `fork_over_the_named_posix_cap_copies_the_heap` + five fork-era scripts in the
capstone differential.

## Slice 4 (exec rung DONE) — external commands: the #801 `/bin` from bash

bash **execs**: `/bin/echo`, PATH lookup, exec'd pipelines (`seq 3 | sort | uniq | wc -l` — four
exec'd programs), redirections to memfs files, and command substitution over exec'd stages all
byte-match the oracle. The lane: `posix_libc/exec.c` (the #801 execve/execv/execvp, staged-pack
argv over the args region + `CAP_SELF_EXEC` image-replace) links as guest code — its `__px_*`
externs bridge in band 5, and `__vm_exec_module` joins the on-ramp's core-builtin lowerings
(self-namespace op 14, mirroring chibicc). `stage_bin.sh` compiles the `posix_utils` coreutils
(the chibicc world, unchanged) to `.temt` command modules; the harness grants each as a `Module`
and registers it as a filesystem executable inside the posix grant (`c_posix.rs`'s
`stage_executable` shape — `bash_probe` takes `BASH_PROBE_BIN=<dir>`). Gate: eight
external-command scripts in the capstone differential (18 scripts total).

## Slice 4 (signals rung DONE) — traps deliver

`trap … INT` + `kill -INT $$` runs the trap (in the parent AND in a fork-twin subshell), ignored
(`trap "" INT`) and repeated deliveries match, EXIT traps compose with subshells. The whole fix
was **one shim line**: async delivery (#796 L2) is gated on a registered handler stack (the
interp's safepoint redirect runs the C handler on a dedicated stack) and bash never calls
`sigaltstack` on this config — band 0 now registers a static 16 KiB stack in a ctor
(`llvm.global_ctors`, which the synthesized `_start` already runs). Gate: five trap scripts in
the capstone differential (23 scripts total).

Known nuance (deferred until a real script trips it): `(kill -INT $$); echo rc=$?` — `$?` after
the shell ITSELF is signaled from a subshell while waiting differs (temen 128, native 0: bash's
`wait_sigint` discard logic vs the personality's `128+sig` zombie status encoding).

## Interactive rung 1 (DONE) — `bash -i` on the #797 controlling terminal, foreground

The harness lane: `temen_run::posix::posix_cap_terminal` enables the #797 terminal at grant time;
the embedder types with `Posix::feed_terminal` from a feeder thread while the shell runs (the
`run_interp_terminal` witness shape). What works, session-proven and gated (the interactive block
in `demo_bash_translates_and_verifies`; `bash_probe` drives ad-hoc sessions via
`BASH_PROBE_TERM='line;^C;line;^D'`):

- **The prompt loop** — bash comes up `flags=himBHs` (interactive AND monitor/job-control mode —
  richer than native bash under a pipe, which loses `m`), prints PS1 to fd 2 between commands,
  reads canonical lines through the feed-time discipline (echo on the captured stdout).
- **`^C` at the prompt** — VINTR through the discipline → SIGINT to the foreground group → bash
  aborts the line, `$? = 130`, fresh prompt (native-exact).
- **`^D` on an empty line** — true EOF → bash prints its `exit` farewell and exits with the last
  command's status.
- **Commands at the prompt** — builtins, external commands, and pipelines (`seq 3 | cat`) run
  exactly as in `-c` mode.
- Shim fix en route: `getcwd(NULL, 0)` (the glibc allocate-extension — bash's shell-init cwd
  probe) now allocates instead of failing into the `shell-init: error retrieving current
  directory` warning.

## Interactive rung 2 (DONE) — job control: `^Z` / `jobs` / `fg`

The full foreground chain works, native-shaped (`[1]+  Stopped  cat`, `fg` resumes with the
terminal, the resumed job reads keystrokes). What the walk took (gated by the job-control block
in the capstone):

1. **The exec'd child's terminal-backed `read(0)`** — one world-level terminal handle can't
   serve every powerbox namespace: each `Proc` now carries its own `term_in` token (seeded at
   `enable_terminal`, cloned by fork, re-pointed by the exec-remap hook — the terminal end rides
   the same exec carry the adopted pipe ends do).
2. **The blocking `WUNTRACED` wait benches** — interactive bash's foreground wait is
   `waitpid(-1, …, WUNTRACED)` (no `WNOHANG`); the #799 bench treated any `WUNTRACED` as
   poll-only, so bash got `-ECHILD` and raced back to the prompt, competing with the job for the
   terminal. Op 28 now benches any non-`WNOHANG` wait, and a child ENTERING the stop park drains
   `ReapWait` benchers on its task (the `^Z` → stop-report wake).
3. **The exec image-replace re-wires the signal door** (`Scheduler::wire_signal_doors`, shared
   with the fork mint) — the fork-time closures captured the pre-exec host's domain id and flag
   cells, so `fg`'s SIGCONT woke a domain that no longer existed while the child parked under
   the new one.
4. **Dup'd-tty termios** — bash parks the terminal at fd 255; the `tc*` ops now gate on the
   duplicated-sentinel rule (`fd_is_terminal`), not a literal fd 0..=2.

## Interactive rung 3 (DONE) — background jobs: `cat &` / SIGTTIN / SIGCHLD

`cat & → jobs → fg → keystrokes-to-the-fg'd-job → ^D → exit 0`, native-shaped (gated by the
background-job block in the capstone). Three mechanisms, each a real gap the session walk hit:

1. **A background terminal read rings SIGTTIN** — the read op now runs `tty_background_check`
   (the write-side SIGTTOU doorbell's twin) before the terminal tag mints, so a background job
   reading the tty STOPS (default action) instead of parking on the input pipe and stealing
   keystrokes from the shell.
2. **The any-child park key** (`ParkEvent::TaskExitAny`) — bash's foreground
   `waitpid(-1, WUNTRACED)` used to bench on the LOWEST live child as an approximation, so a
   stopped background `cat` absorbed the bench while the actual foreground child exited and
   bash slept forever. The wildcard benches under a per-parent key
   (`REAP_ANY_BASE | parent-domain`) that every child-transition drain point (twin completion,
   the stop-park insert, `wake_stopped`) wakes; the park-vs-transition race closes with a
   level-triggered pending mark (`reap_any_pending`), not a `results` scan — a personality-lane
   twin's outcome lingers in `results` after the guest reaps it, so a scan would spin.
3. **SIGCHLD is generated** — a child transition (stop, continue, exit) now raises SIGCHLD in
   the parent through the delivery gate, so interactive bash's `waitchld` handler keeps the job
   table live: without it, `jobs` said Running after the SIGTTIN stop and `fg` skipped its
   SIGCONT (it only continues jobs it knows are stopped), then mis-read the stale stop report
   as the job re-stopping. The exit-hook site pends-and-arms only (it runs under the core's
   scheduler lock; benched waiters are woken by the core's own drains). Parents without a
   handler discard it at generation (default-ignore) — the C witnesses are unaffected.

Also en route: root exit now sweeps parked pipe readers/writers and `waitpid` benchers in
`teardown_run` (a pre-existing leak — a twin parked in a pipe read at root exit hung the run).

## Rung-3 tail (DONE) — `bg` without the read steal; here-docs pinned

- **The `-ERESTART` sentinel** — `bg`'s SIGCONT re-issued the stopped job's terminal read, which
  re-rang SIGTTIN, but the deferred stop fire lagged one dispatch and the reader CONSUMED the
  next typed line before parking (the second `jobs` after `bg` reached cat, not bash). POSIX
  stops *before* the I/O and the kernel transparently restarts it after continue — so the read
  op now returns `-ERESTART` (-85) instead of minting the input tag while a stop is pending, and
  both guest read wrappers (`bash_shim.c`, `posix_utils/util.c`) loop on it: the stop lands at
  the re-issued dispatch's safepoint poll, before the pipe is touched. Gated by a fourth
  interactive session block (`cat & → jobs → bg → jobs → kill -9 %1 → exit`, asserting the
  second `jobs` still lists the job).
- **Here-docs came free** — the slice-4 "remaining" note was stale: bash spools each here-doc
  into an unlinked temp file, which the #800/#801 fs surface already serves. All shapes work
  (expansion, quoted-delimiter, `<<-`, here-strings, builtin `read`/loops, exec'd commands);
  six differential scripts now pin them.

## Language differential (DONE) — 50 pure-in-shell constructs vs native, two fixes

A 50-construct sweep (`bash -c` vs the native 5.2.21 oracle: arrays + associative arrays, `case`
incl. `;;&`, the full parameter-expansion family, brace expansion, arithmetic + C-style `for`,
`[[ ]]`, `printf`, `read -a`/IFS, `local`/`declare -i`/`-r`/`-n`, `mapfile`, `extglob`, indirect
expansion) came back **48/50 byte-identical out of the box** — the tree already had the surface.
Two needed a fix:

- **`BASH_REMATCH`** — `[[ =~ ]]` matched correctly but filled no capture array. The guest
  `posix_libc/regex.c` defined its OWN `regex_t`/`regmatch_t`, but bash's TUs allocate them from
  the build host's glibc `<regex.h>` and read `re_nsub` + the `pmatch` offsets across the call.
  The layouts must match the glibc ABI byte-for-byte (`re_nsub@48` in a 64-byte `regex_t`;
  `regoff_t` is `int`, so `regmatch_t` is `{int,int}`). The old layout still *matched* (the
  guest's internal fields are self-consistent) but bash read the offsets from the wrong places →
  empty `BASH_REMATCH`. Now overlaid on glibc's struct.
- **Process substitution `<(…)`/`>(…)`** — bash builds with `HAVE_DEV_FD`, so it substitutes
  `/dev/fd/N` for the pipe end and the peer `open`s it. The personality's `open` now resolves
  `/dev/fd/N` and `/dev/std{in,out,err}` as a **dup** of that fd (sharing the description — an
  `Arc` clone of the CorePipe token, so the refcount/last-close EOF stay correct).

Six scripts pin these in the capstone (BASH_REMATCH captures, both process-substitution
directions, `/dev/stdin`).

## Pipeline `$?` (DONE — #1057)

The last stage of a pipeline that terminates via `exit()` (every forked bash pipeline stage /
group command reaching `exit_shell`) used to report `$? = 128` — the fork-twin **crash** status —
instead of its real exit code. Root cause: `reap_status` in temen-interp mapped **every**
`Err(Trap)` to `REAP_CRASH_STATUS`, but `Trap::Exit(code)` is a *clean* guest `exit(code)` ("not
an error — the domain asked to terminate"), exactly what the root path already turns into
`Outcome::Exited(code)`. A fork twin that exits via `exit()` rather than returning must reap with
that code. One-arm fix (`Err(Trap::Exit(code)) => code & 0xff`); only genuine traps
(unreachable, memory/cap faults) still reap 128. Surfaced by the language differential, masked in
every earlier pipe script by a trailing command. Pinned by three pipeline scripts in the
capstone (`true | { false; }` → `rc=1`, etc.).

## Explicit `exit` in `-c` mode (DONE — #1062)

`bash -c 'exit N'` (and `set -e`/`set -u` on error, which reach the same terminate path) used to
**busy-loop forever** — `run_unwind_frame` ↔ `jump_to_top_level` — never reaching the exit. Root
cause was in the setjmp/longjmp core (#795), not bash: the interp keyed each `setjmp` checkpoint by
the guest **jmp_buf address** and wrote nothing into the buffer, but bash's `parse_and_execute`
save/restores `top_level` with `COPY_PROCENV` (a plain `jmp_buf` **memcpy**). After the restore
memcpy the interp's address-keyed map still pointed at the *inner* checkpoint, so the `EXITPROG`
re-throw `longjmp(top_level)` resolved to `parse_and_execute`'s own handler again → infinite loop.
Interactive `exit`, implicit end-of-script exit, and subshell exit all worked (they don't hit that
copy-then-re-throw path). Fix: `setjmp` now mints a token, **writes it into the jmp_buf's opaque
first 8 bytes**, and keys the checkpoint by the token; `longjmp` reads the token back from the
(possibly-copied) buffer — so the identity rides the memcpy. Bounded by pruning checkpoints whose
frame has returned. Pinned by a `COPY_PROCENV`-shaped C witness (`c_longjmp_through_a_copied_jmp_buf`)
and three capstone scripts (`exit 7`, `set -e; false`, `set -u`).

## Differential round 3 (DONE) — deeper surface + `strftime`

A second differential pass over deeper constructs (trap ERR/RETURN, `set -e` in functions/subshells/
`pipefail`, getopts, arithmetic edge cases, associative-array counting, real-script control flow)
came back clean **except** `printf '%(fmt)T'`: the bash shim's `strftime`/`localtime` were stubs that
ignored the format and always printed `1970-01-01`. Ported the real UTC calendar math + format engine
from the postgres `time_shim.c` (gap #11e, glibc-exact) into `bash_shim.c` — `%Y %m %d %H %M %S %j %A
%a %B %b %p %y %C %I %e %u %w` all match native now, across epochs (incl. pre-1970 negatives). Also
taught the demo `/bin/head` the POSIX `-N` shorthand (`head -1`, as `declare -f f | head -1` uses).
Pinned by a dozen capstone scripts (the `%()T` formats, the trap/`set -e`/getopts/arith/assoc set, and
`seq 9 | head -3`).

## Differential round 4 (DONE) — whole real programs

Ran self-contained multi-line bash *programs* end-to-end vs native (the #802 "script suite" goal),
not one-liners: a recursive quicksort, an RPN calculator, a key=value state-machine parser, memoized
fibonacci, a retry loop with an EXIT trap, a getopts app, string tools, a recursive case-dispatcher,
and a word-frequency counter. **They all run correctly** — no new bash/interp bug surfaced, strong
viability evidence. The only gap was a *missing* coreutil: `tr` wasn't staged, so `tr`-based pipelines
(word frequency) produced nothing. Added a small chibicc-safe `tr` (`SET1→SET2` translate, `-d`
delete, `\n`/`\t` escapes, `a-z` ranges) to `posix_utils` + `stage_bin.sh`. (Also noted: deep
recursion through `$(...)` command-substitution — e.g. un-memoized fib(10) — is *correct* but slow,
since each call forks a subshell and svm forks are heavier than native; a perf characteristic, not a
bug.) Pinned by three whole-program capstone scripts (quicksort, the state-machine parse, the
`tr`+`sort` word-frequency counter).

## What remains (the slice ladder from the #802 sketch)

- The `^D`-EOF nuance (the one-shot EOF is writer-count state, so the shell's next read can
  consume an EOF meant for the job — native VEOF is a queued, one-READ event; the capstone
  sessions don't currently trip it).
- The `$?`-after-self-SIGINT edge above (slice 4's known nuance).
- Known band-0 papering (revisit when a differential trips over one): `fstat` synthesizes a
  chr-device for fds 0-2 and re-stats the recorded open path otherwise; `st_ino` is a path hash
  (same-file checks distinguish paths, not hardlinks); `sigsuspend` returns `EINTR` without
  suspending; readline/progcomp externs stay trap stubs (`--disable-readline`; the `complete`
  builtin would hit them).

| File | Role |
|---|---|
| `build_bitcode.sh` | the faithful fetch→configure→oracle→bitcode→link→translate pipeline |
| `bash_shim.c` | the bash-specific libc/OS surface (grows per slice; see its header) |
