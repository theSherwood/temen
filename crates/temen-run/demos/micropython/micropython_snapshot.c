/* MicroPython warm-runtime-snapshot driver (the QuickJS/Tcl `*_snapshot.c` contract).
 *
 * `mp_embed_init` (GC heap + interpreter bring-up) is program-independent, so the playground pays it
 * once: the host runs `warmup()`, snapshots the window, then restores that image and runs `eval_run()`
 * per Run — evaluating only the user's Python on top. Fresh-per-Run isolation holds because each Run
 * restores the SAME post-warmup snapshot into a fresh window; MicroPython heap pointers are all
 * window-relative in the Temen model, so a byte-for-byte restore keeps them valid.
 *
 *   - `main`     — the one-shot cold path (read stdin → init → exec → deinit): the baseline, and what
 *                  the standalone `micropython.c` driver does. Kept self-contained.
 *   - `warmup`   — `mp_embed_init` into the static GC heap, then return. No stdin, no exec. The host
 *                  snapshots here.
 *   - `eval_run` — read stdin (the user's Python) and exec it over the warm (restored) interpreter.
 */
#include "port/micropython_embed.h"

extern long read(int fd, void *buf, long n);

static char heap[256 * 1024];
static char src[1 << 20];

static long read_stdin(void) {
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
    return len;
}

/* BASELINE (cold): read, init, exec, teardown. */
int main(void) {
    read_stdin();
    int stack_top;
    mp_embed_init(&heap[0], sizeof(heap), &stack_top);
    mp_embed_exec_str(src);
    mp_embed_deinit();
    return 0;
}

/* WARM PHASE 1: bring up the interpreter into the statics. The host snapshots the window after this
 * returns — the state is program-independent (no stdin, no exec). */
int warmup(void) {
    int stack_top;
    mp_embed_init(&heap[0], sizeof(heap), &stack_top);
    return 0;
}

/* WARM PHASE 2: read stdin (the user's Python) and exec it over the warm, restored interpreter. */
int eval_run(void) {
    read_stdin();
    mp_embed_exec_str(src);
    return 0;
}
