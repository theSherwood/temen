/* Tcl interactive editor guest — the browser-playground REPL. Reads a Tcl script from **stdin**
 * (the Stream.read capability), evaluates the whole buffer in one interpreter, and prints the
 * completion result (or the error message). Anything the script `puts`-es goes straight to stdout
 * through Tcl's own stdout channel. The playground page pipes the Tcl editor's text in as stdin, so
 * a user writes and runs Tcl entirely client-side, in the sandbox.
 *
 * **Minimal embedding — no `Tcl_Init`, so no ambient OS surface.** `Tcl_CreateInterp` registers the
 * whole language and nearly all built-in commands (`set`/`expr`/`proc`/`if`/`while`/`for`/`foreach`/
 * `string`/`list`/`lindex`/`dict`/`regexp`/`switch`/…) in C, needing no script library on disk. We
 * deliberately skip `Tcl_Init` (which would `source init.tcl` and load encodings from a filesystem)
 * and `Tcl_FindExecutable` (which readlinks the executable path) — the language core needs neither.
 * That is the direct analog of QuickJS's `qjs_eval.c` "no quickjs-libc, no ambient OS surface." The
 * fuller-Tcl path (auto_load, `file`/`glob`, `clock`) that seeds the script library + encodings into
 * the svm-posix memfs is a documented follow-up (see README, "Full Tcl_Init").
 *
 * Stateless: each Run is a fresh interpreter, matching the "run the whole buffer" editor model (as in
 * `qjs_repl.c` / `sqlite_repl.c`). Built into a `.svmb` playground asset by
 * `browser/build-onramp-assets.mjs`.
 */
#include <string.h>
#include "tcl.h"

extern long read(int fd, void *buf, long n);

/* 1 MiB program buffer — the editor's text, read from stdin. */
static char src[1 << 20];

int main(void) {
    long len = 0;
    for (;;) {
        long r = read(0, src + len, (long)sizeof(src) - 1 - len);
        if (r <= 0)
            break;
        len += r;
        if (len >= (long)sizeof(src) - 1)
            break;
    }
    src[len] = 0;

    Tcl_Interp *interp = Tcl_CreateInterp();

    int code = Tcl_Eval(interp, src);

    /* Print the completion result REPL-style: the value of the last command on success, the error
     * message on failure. `puts` output has already gone to stdout through Tcl's stdout channel. */
    const char *res = Tcl_GetStringResult(interp);
    if (code != TCL_OK) {
        fputs("error: ", stdout);
    }
    if (res && res[0]) {
        fputs(res, stdout);
        fputc('\n', stdout);
    }

    Tcl_DeleteInterp(interp);
    return code == TCL_OK ? 0 : 1;
}
