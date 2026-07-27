# The `svm-posix` shell — canonical C source

The Stage-0/Stage-1 shell (STAGE1.md): a real command interpreter compiled by the
in-tree **chibicc** onto the `svm-posix` personality (libc by name). These `.c`
files are the single source of truth, `include_str!`d by the differential test
`crates/svm/tests/c_shell.rs` and (as of the playground-shell epic) compiled into
the `shell.svmb` asset the browser playground runs.

| file | what | used by |
|------|------|---------|
| `shim.c` | guest libc shim — standard names (`read`/`write`/`open`/`opendir`/`getcwd`/…) adapting C's NUL-terminated `char*` to the personality's explicit `(ptr,len)` ABI, forwarding to the `__px_*` / `__spawn` / `__rg_*` imports discovered by `cap.self` reflection | shell |
| `ring.c` | the SPSC byte ring over a mapped `SharedRegion` (concurrent pipelines, STAGE1.md item 6) | shell + `__stage` runner |
| `shell_main.c` | the shell itself — `main()` read-eval loop, builtins, redirects, pipelines, `if`, variables, globbing, external-command dispatch | shell |
| `stage_runner_main.c` | the `__stage` filter runner: a `--child-entry` program the shell spawns per pipeline stage, mapping its granted rings and running one filter | external command |

The shell is assembled as `shim.c + ring.c + shell_main.c`; the `__stage` runner as
`ring.c + stage_runner_main.c` (it holds no personality capability — only rings +
reflection). Both must stay chibicc-compatible.
