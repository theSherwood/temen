/* Lua warm-runtime-snapshot driver (WASM_AOT.md warm+JIT · issue #805). The two-phase twin of
 * `lua_eval_harness.c`: it splits the single-shot `main` (read stdin -> newstate + openlibs -> eval)
 * into a program-independent `warmup` (create the lua_State + open the editor libs into a static, which
 * the host snapshots) and a per-Run `eval_run` (read the editor's chunk from stdin, run it over the
 * warm, restored state, print any error). Fresh-per-Run isolation holds: each Run restores the SAME
 * post-warmup snapshot, so a global set in one Run can't leak into the next.
 *
 * Unlike `lua_eval_harness.c`'s fixed 48 MiB static arena, this uses the on-ramp heap
 * (`luaL_newstate`'s default `realloc`/`free` allocator -> the translator's `__svm_malloc` bump heap),
 * so the snapshot image is the small live prefix `[0, brk)` (like QuickJS's ~4 MiB) and the declared
 * window stays under the warm session's mapped window (svm_warm_open requires declared < mapped). `free`
 * is a no-op bump-allocator free, exactly as for the QuickJS card — fine for playground-scale snippets.
 *
 * Built like `lua_eval.ll` (same Lua core + libs + guest layers, swapping this harness in for
 * `lua_eval_harness.c`); see fixtures/lua/README.md "Regenerating (eval)". Driven by the browser warm
 * session (`svm_warm_open` snapshots after `warmup`; `svm_warm_eval` restores + runs `eval_run`).
 */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

extern long read(int fd, void *buf, long n);
extern long write(int fd, const void *buf, long n);
static unsigned long slen(const char *s) { unsigned long n = 0; while (s[n]) n++; return n; }

static char src[1 << 20]; /* up to 1 MiB of editor text */

/* The editor lib set — base/string/table/math/coroutine/io/os (matches lua_eval_harness.c). io.open
 * degrades to nil (no fs cap granted); print/io.write reach stdout via the Stream capability. */
static void open_editor_libs(lua_State *L) {
  static const luaL_Reg libs[] = {
    {LUA_GNAME, luaopen_base},          {LUA_STRLIBNAME, luaopen_string},
    {LUA_TABLIBNAME, luaopen_table},    {LUA_MATHLIBNAME, luaopen_math},
    {LUA_COLIBNAME, luaopen_coroutine}, {LUA_IOLIBNAME, luaopen_io},
    {LUA_OSLIBNAME, luaopen_os},        {(void *)0, (void *)0},
  };
  for (const luaL_Reg *lib = libs; lib->func; lib++) { luaL_requiref(L, lib->name, lib->func, 1); lua_pop(L, 1); }
}

/* Read the editor text from stdin into `src` (NUL-terminated); returns the byte length. */
static long read_stdin(void) {
  long len = 0;
  for (;;) {
    long r = read(0, src + len, (long)sizeof(src) - 1 - len);
    if (r <= 0) break;
    len += r;
    if (len >= (long)sizeof(src) - 1) break;
  }
  src[len] = 0;
  return len;
}

/* Load + run `src` (len bytes) over `L`; print any load/runtime error to stdout. Returns 0 / 2. */
static int run_chunk(lua_State *L, long len) {
  if (luaL_loadbuffer(L, src, (size_t)len, "=input") != LUA_OK || lua_pcall(L, 0, 0, 0) != LUA_OK) {
    const char *msg = lua_tostring(L, -1);
    if (msg) { write(1, "error: ", 7); write(1, msg, (long)slen(msg)); write(1, "\n", 1); }
    return 2;
  }
  return 0;
}

/* Warm runtime, left resident by `warmup` for `eval_run` (captured/restored by the host snapshot). */
static lua_State *g_L;

/* BASELINE (cold): the original one-shot flow — read, newstate, open libs, run, close. Self-contained,
 * so the native prototype / differential can measure the cold path against the warm one. */
int main(void) {
  long len = read_stdin();
  lua_State *L = luaL_newstate();
  if (!L) return 1;
  open_editor_libs(L);
  int rc = run_chunk(L, len);
  lua_close(L);
  return rc;
}

/* WARM PHASE 1: create the lua_State and open the editor libs into the static g_L. Program-independent
 * (no stdin, no eval), so the window the host snapshots after this returns is reusable by every Run. */
int warmup(void) {
  g_L = luaL_newstate();
  if (!g_L) return 1;
  open_editor_libs(g_L);
  return 0;
}

/* WARM PHASE 2: read the editor chunk from stdin and run it over the warm, restored state. Runs per Run
 * over a window the host restored from the warmup snapshot; no runtime rebuild. */
int eval_run(void) {
  long len = read_stdin();
  return run_chunk(g_L, len);
}
