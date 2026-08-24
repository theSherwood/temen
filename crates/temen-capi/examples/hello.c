/*
 * A complete C embedding of the Temen runtime via temen.h: parse hand-written IR, bind a built-in
 * `write` (stdout) capability and a custom host-defined `meaning` capability by name, instantiate,
 * run on the JIT, and read back the captured stdout and the entry's return value.
 *
 * Build (from the repo root, after `cargo build -p temen-capi`):
 *   cc crates/temen-capi/examples/hello.c \
 *      -I crates/temen-capi/include -L target/debug -ltemen_capi \
 *      -lpthread -ldl -lm -o /tmp/temen_hello && /tmp/temen_hello
 *
 * See examples/README.md.
 */
#include "temen.h"
#include <stdio.h>
#include <string.h>

/* A host-defined capability: meaning(x) = x + 40. Reached by the guest as call.sym "meaning".
 * This one only computes on its scalar args, so it ignores the guest-window handle (mem). */
static int32_t meaning(void *ctx, uint32_t op, const int64_t *args, size_t n_args,
                       int64_t *results, size_t cap, TemenGuestMem *mem) {
  (void)ctx;
  (void)op;
  (void)mem;
  if (n_args < 1 || cap < 1) return -1; /* trap the cap call, fail-closed */
  results[0] = args[0] + 40;
  return 1; /* one result written */
}

static const char *IR =
    "memory 15\n"
    "data ro 16384 \"Hello from C!\\n\"\n"
    "export 0 func \"_start\" 0\n"
    "func () -> (i64) {\n"
    "block 0 () {\n"
    "  v0 = i32.const 0\n"            /* dummy handle operand: the slot binding dispatches */
    "  v1 = i64.const 16384\n"
    "  v2 = i64.const 14\n"
    "  v3 = call.sym \"write\" (i64, i64) -> (i64) v0 (v1, v2)\n"
    "  v4 = i32.const 0\n"
    "  v5 = i64.const 2\n"
    "  v6 = call.sym \"meaning\" (i64) -> (i64) v4 (v5)\n"
    "  return v6\n"
    "  }\n"
    "}\n";

int main(void) {
  TemenModule *m = temen_module_parse_text(IR);
  if (!m) {
    fprintf(stderr, "parse: %s\n", temen_last_error());
    return 1;
  }
  TemenImports *imports = temen_imports_new();
  temen_imports_provide_stdout(imports, "write");
  temen_imports_provide_host_fn(imports, "meaning", 0, meaning, NULL);

  /* Consumes m and imports. */
  TemenInstance *inst = temen_instantiate_with_imports(m, imports);
  if (!inst) {
    fprintf(stderr, "instantiate: %s\n", temen_last_error());
    return 1;
  }

  TemenRun *run = temen_instance_run(inst, TEMEN_BACKEND_JIT, NULL);
  if (!run) {
    fprintf(stderr, "run: %s\n", temen_last_error());
    return 1;
  }

  size_t len = 0;
  const uint8_t *out = temen_run_stdout(run, &len);
  printf("guest stdout: %.*s", (int)len, out);
  printf("entry returned: %lld\n", (long long)temen_run_result(run, 0));

  int ok = (len == strlen("Hello from C!\n") &&
            memcmp(out, "Hello from C!\n", len) == 0 && temen_run_result(run, 0) == 42);

  temen_run_free(run);
  temen_instance_free(inst);

  printf(ok ? "OK\n" : "MISMATCH\n");
  return ok ? 0 : 1;
}
