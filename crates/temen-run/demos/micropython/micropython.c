/* MicroPython REPL guest — the browser-playground driver. Reads a Python program from **stdin** (the
 * Stream.read capability), runs it in one embedded interpreter, and lets the program's own `print(...)`
 * go to stdout (through the HAL override in `py_shim.c`, which writes straight to the Stream). The
 * playground page pipes the editor's text in as stdin, so a user writes and runs Python entirely
 * client-side, in the sandbox.
 *
 * **Minimal embedding — the `ports/embed` API, no ambient OS surface.** `mp_embed_init` sets up the GC
 * heap + interpreter; `mp_embed_exec_str` compiles and executes the source. This is the direct analog
 * of QuickJS's `qjs_eval.c` / Tcl's `tcl_repl.c`: the language core runs with no filesystem and no OS
 * personality. Frozen modules (a follow-up) give `import` with no `fs` capability.
 *
 * Stateless: each Run is a fresh interpreter, matching the "run the whole buffer" editor model. Built
 * into a `.temen` playground asset by `browser/build-onramp-assets.mjs` (slice C).
 */
#include "port/micropython_embed.h"

extern long read(int fd, void *buf, long n);

/* The MicroPython GC heap. 256 KiB is comfortable for REPL-scale programs; the window's heap can grow
 * via the Memory cap if a program needs more. */
static char heap[256 * 1024];

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

    /* &stack_top marks the top of the C stack for the GC's setjmp-based root scan (MICROPY_GCREGS_SETJMP). */
    int stack_top;
    mp_embed_init(&heap[0], sizeof(heap), &stack_top);
    mp_embed_exec_str(src);
    mp_embed_deinit();
    return 0;
}
