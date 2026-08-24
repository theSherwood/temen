/* QuickJS warm-runtime-snapshot prototype driver (WASM_AOT.md follow-on).
 *
 * The playground's `qjs_repl.c` rebuilds the whole QuickJS runtime (JS_NewRuntime + JS_NewContext +
 * every intrinsic) on EVERY Run — ~380 ms warm in the browser, dwarfing a do-nothing program. This
 * driver splits that into a **warm-once, restore-per-Run** shape so the host can snapshot the window
 * after the (program-independent) init and restore it per Run, only evaluating the user's code on top:
 *
 *   - `main`      — the original one-shot cold path (read stdin → init → eval → print). The BASELINE
 *                   the prototype measures against; uses local rt/ctx so it's self-contained.
 *   - `warmup`    — init runtime + context + bindings into STATIC globals, then return. No stdin, no
 *                   eval → the produced window is program-independent. The host snapshots here.
 *   - `eval_run`  — read stdin (the user's JS), eval over the warm (restored) g_ctx, print. Runs per
 *                   Run over a window restored from the warmup snapshot.
 *
 * Fresh-per-Run isolation is preserved because each Run restores the SAME post-warmup snapshot into a
 * fresh window (a var defined in one Run can't leak into the next). QuickJS heap pointers are all
 * window-relative in the Temen model, so a byte-for-byte window restore keeps them valid.
 *
 * Built like `qjs_repl.c` (engine + openlibm + shims, llvm-linked, translated); driven by the native
 * `qjs_snapshot` example.
 */
#include <stdio.h>
#include <string.h>
#include "quickjs.h"

extern long read(int fd, void *buf, long n);

/* 1 MiB program buffer — the editor's text, read from stdin. */
static char src[1 << 20];

/* Warm runtime, left resident by `warmup` for `eval_run` (captured/restored by the host snapshot). */
static JSRuntime *g_rt;
static JSContext *g_ctx;

/* `print(...)` / `console.log(...)` — space-separated args, newline-terminated, to stdout. */
static JSValue js_print(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    (void)this_val;
    for (int i = 0; i < argc; i++) {
        if (i)
            putchar(' ');
        const char *s = JS_ToCString(ctx, argv[i]);
        if (s) {
            fputs(s, stdout);
            JS_FreeCString(ctx, s);
        }
    }
    putchar('\n');
    return JS_UNDEFINED;
}

/* Bind `print` + `console.log` onto a context's global object. */
static void bind_console(JSContext *ctx) {
    JSValue global = JS_GetGlobalObject(ctx);
    JS_SetPropertyStr(ctx, global, "print", JS_NewCFunction(ctx, js_print, "print", 1));
    JSValue console = JS_NewObject(ctx);
    JS_SetPropertyStr(ctx, console, "log", JS_NewCFunction(ctx, js_print, "log", 1));
    JS_SetPropertyStr(ctx, global, "console", console);
    JS_FreeValue(ctx, global);
}

/* Read stdin into `src` (NUL-terminated), returning the length. */
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

/* Eval `src` over `ctx`, printing the completion value (or the uncaught exception), REPL-style. */
static void eval_and_print(JSContext *ctx) {
    JSValue val = JS_Eval(ctx, src, strlen(src), "<repl>", JS_EVAL_TYPE_GLOBAL);
    if (JS_IsException(val)) {
        JSValue exc = JS_GetException(ctx);
        const char *e = JS_ToCString(ctx, exc);
        printf("Uncaught %s\n", e ? e : "exception");
        JS_FreeCString(ctx, e);
        JS_FreeValue(ctx, exc);
    } else {
        const char *s = JS_ToCString(ctx, val);
        printf("%s\n", s ? s : "undefined");
        JS_FreeCString(ctx, s);
    }
    JS_FreeValue(ctx, val);
}

/* BASELINE (cold): the original one-shot flow — read, init, eval, print, teardown. */
int main(void) {
    read_stdin();
    JSRuntime *rt = JS_NewRuntime();
    JSContext *ctx = JS_NewContext(rt);
    bind_console(ctx);
    eval_and_print(ctx);
    JS_FreeContext(ctx);
    JS_FreeRuntime(rt);
    return 0;
}

/* WARM PHASE 1: init the runtime + context + bindings into the statics. The host snapshots the window
 * after this returns — the state is program-independent (no stdin read, no eval). */
int warmup(void) {
    g_rt = JS_NewRuntime();
    g_ctx = JS_NewContext(g_rt);
    bind_console(g_ctx);
    return 0;
}

/* WARM PHASE 2: read stdin (the user's JS) and eval it over the warm, restored context. Runs per Run
 * over a window the host restored from the warmup snapshot; no runtime rebuild. */
int eval_run(void) {
    read_stdin();
    eval_and_print(g_ctx);
    return 0;
}
