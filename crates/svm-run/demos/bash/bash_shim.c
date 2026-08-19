/* bash libc shim — the bash-specific OS/libc surface the on-ramp neither synthesizes, resolves
 * to an svm-posix capability, nor covers via a reused shim (#802 slice 2). Ordinary guest C; a
 * guest definition shadows the on-ramp's would-be trap stub (the `tcl_shim.c` discipline).
 *
 * The reuse map (see README) — NOT redefined here:
 *   - printf/stdio family  → `../postgres/printf_shim.c`
 *   - strtod               → `../strtod/strtod.c`
 *   - mem/string/qsort/malloc/llvm.* → on-ramp-synthesized
 *   - open/read/write/close/lseek/stat/opendir/readdir/getcwd/chdir/getenv/setenv/unlink/exit
 *     and the process ops (fork/waitpid/kill/pipe/dup2/signals/pgids/termios)
 *                          → svm-posix capabilities (POSIX.md ops), resolved at load
 *
 * Slice 2's contract is translate+verify: whatever bash reaches that is NOT yet defined or
 * resolved rides `SVM_STUB_EXTERNS` trap stubs, and slice 3 (the first run) grows this file from
 * the stub report — the same gap-walk every capstone did. What is here now is only what a stub
 * would get WRONG at first touch.
 */
#include <stddef.h>

/* --- errno ------------------------------------------------------------------------------------ */
static int __bash_errno;
int *__errno_location(void) { return &__bash_errno; }

/* --- environ ----------------------------------------------------------------------------------
 * bash walks `environ` wholesale at startup (shell.c hands it to `initialize_shell_variables`).
 * Without a definition the extern lays out as zeroed BSS and the walk dereferences NULL — a trap
 * under the #964 guard (the exact Tcl #986 shape). Start it as a real empty vector; slice 3
 * seeds it from the personality's environ op instead. */
static char *shim_environ[1] = {0};
char **environ = shim_environ;
