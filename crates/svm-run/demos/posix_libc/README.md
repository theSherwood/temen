# posix_libc — guest-side libc modules for the POSIX personality (#800)

Real-libc functions bash calls that are **pure guest code**: they either compute
(no host state) or compose personality ops that already exist, so they live here
as freestanding C — no includes, no new ops — following the POSIX.md §1 split
(semantics guest-side or in swappable host bookkeeping, never in the VM core).

Each file is self-contained and concatenation-friendly: the `crates/svm/tests/c_posix.rs`
harness `include_str!`s them next to its `__px_` shim and compiles the lot as one
translation unit through the chibicc frontend. Files declare the `__px_` externs
they need themselves (identical duplicate declarations are legal C).

| File | Provides | Backed by |
|---|---|---|
| `fnmatch.c` | `fnmatch(3)` — `*` `?` brackets/ranges/negation/`[[:class:]]`, `FNM_PATHNAME`/`FNM_PERIOD`/`FNM_NOESCAPE`/`FNM_CASEFOLD` | pure compute; differential-tested against the host's `fnmatch(3)` |
| `posix_misc.c` | `putenv`, `wait3`, `wait4` | setenv/unsetenv/waitpid ops (12/35/28) |
| `regex.c` | `regcomp`/`regexec`/`regfree` — POSIX **ERE** with captures (`BASH_REMATCH`): `.` `^` `$`, brackets + `[[:class:]]`, groups, `\|`, `*` `+` `?` `{n,m}`; `REG_ICASE`/`REG_NOSUB`/`REG_NOTBOL`/`REG_NOTEOL`. Leftmost-**longest** (POSIX), via exhaustive exploration with a step budget — bash-sized patterns never hit it; `REG_NEWLINE` and BRE unimplemented | malloc/free ops (2/3); differential-tested — spans **and** captures — against the host's `regexec(3)` |

Still to come under #800: `glob`/`globfree` (over opendir/readdir ops 14–16
plus this `fnmatch`), and `getline`/`getdelim` (blocked on a minimal `FILE` layer
decision — see the issue).
