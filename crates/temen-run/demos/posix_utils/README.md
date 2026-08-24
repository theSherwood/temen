# posix_utils — the staged /bin coreutils (#801)

The userland half of #801's "filesystem of executables": each file here is one
coreutil as freestanding C, compiled **as its own command module**
(`--child-entry`) and registered under `/bin/<name>` via
`Posix::register_executable` — so `execvp` finds them on `PATH`, `stat` shows
their exec bits, and fork→exec pipelines run real multi-command workloads.

Unlike `../posix_libc` (library code concatenated *into* a caller's TU), these
are **programs**: each TU is `util.c` + the tool (+ `../posix_libc/regex.c`
before `grep.c`), staged by the `stage_coreutils` helper in
`crates/temen/tests/c_posix.rs`. `util.c` carries the shared runtime — the #972
tag-protocol fd wrappers (`read`/`write`/`close`/`pipe`), string/number
helpers, and a byte-at-a-time line reader — so every tool's stdin/stdout works
identically on personality fds, adopted core-pipe ends, and the terminal.

| Tool | Does | Notes |
|---|---|---|
| `true` / `false` | exit 0 / 1 | |
| `echo` | argv joined by spaces + `\n` | `-n` |
| `cat` | stdin or file args → stdout | `__px_open` per arg |
| `seq` | `seq LAST` / `seq FIRST LAST` | one integer per line |
| `head` | first N lines of stdin | `-n N`, default 10; byte-wise, never over-reads |
| `wc` | counts of stdin | `-l`/`-w`/`-c`, default `L W C` |
| `sort` | all stdin lines, byte order | fixed 64 KiB arena, overflow = exit 2 |
| `uniq` | collapse adjacent duplicates | `-c` prefixes the run count |
| `grep` | POSIX ERE over stdin/file | via `posix_libc/regex.c`; exit 0 hit / 1 miss / 2 error |
| `ls` | sorted entries of DIR (default `.`) | opendir/readdir |
| `pwd` | the cwd | getcwd (op 9) |

Deliberately small: no locales, no multi-flag parsing, fixed arenas over
malloc where a bound is honest. These exist to exercise the exec/PATH/pipe
machinery with real programs — when the LLVM on-ramp brings the GNU originals
(#795/#802), these stay as the fast smoke set.
