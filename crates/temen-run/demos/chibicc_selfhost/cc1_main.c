// Guest entry for chibicc-the-guest (SELFHOST_C.md §5 A): the `-cc1 --emit-ir` slice as a
// standalone `main`, replacing frontend/chibicc/main.c entirely so the driver (subprocess spawning,
// glob, temp files — Appendix A.3: load-fatal externs on the personality) is never compiled in.
// Upstream chibicc sources are untouched; this file defines the globals main.c owns and drives
// tokenize → preprocess → parse → codegen_ir directly (main.c's cc1() without the -include/-M/-E
// driver options, and without its open_memstream detour — codegen_ir writes sequentially, so the
// output FILE* is passed straight through).
#include "chibicc.h"
#include <sys/stat.h>

// Globals owned by main.c in the native build (chibicc.h externs + codegen_ir.c's two).
StringArray include_paths;
bool opt_fcommon = true;
bool opt_fpic;
bool opt_emit_ir = true;
bool opt_emit_object; // --emit-object: emit a linkable Temen unit (cc -c) instead of a whole program
bool opt_g; // debug info off by default (`-g` turns it on); the `debug.*` waist is ~a third of the
            // emitted IR, so the playground compiles clean+fast unless the user opts into debugging.
bool opt_child_entry;
int opt_data_page = 16384; // --data-page: RO/writable D40 isolation granularity; the browser (64 KiB
                           // wasm host page) passes `--data-page 65536`. Matches main.c's default.
char *base_file;

// main.c's file_exists (preprocess.c calls it for __has_include); OP_STAT serves it (App. A.1).
bool file_exists(char *path) {
  struct stat st;
  return !stat(path, &st);
}

// codegen.c owns this in the native build, but parse.c/codegen_ir.c call it 15× (struct
// layout, stack slots) — and codegen.c (the x86 backend) is excluded from the guest TU set,
// so without this definition it becomes a trap-if-called stub and every struct layout traps.
// Caught by the step-2 stub report (build_chibicc_temen.sh step 3a).
int align_to(int n, int align) {
  return (n + align - 1) / align * align;
}

// usage: chibicc [-Idir] [--data-page N] <in.c> [out]   ("-"/absent out → stdout). The include dir
// defaults to /include — where the memfs seed mounts frontend/chibicc/include for the guest (§5 C).
// The optional leading -Idir lets the native reference (self-host differential, run_selfhost_diff.sh)
// point at the real header tree instead, since /include doesn't exist on a host fs; it never
// affects the emitted IR (headers aren't named in the output), so both sides stay comparable.
// --data-page N sets the D40 RO/writable isolation granularity (the browser passes 65536 for its
// 64 KiB wasm host page, so a compiled program's RO globals never share a host page with writable data).
// Debug info is **off by default** (the `debug.*` waist is ~a third of the emitted IR); `-g` turns it
// on so the W1 debugger can consume the source-line/variable waist, `-g0` is the explicit off (a no-op
// against the default). The playground passes `-g` only when the user opts into source-level debugging.
int main(int argc, char **argv) {
  int n_inc = 0;
  StringArray opt_include = {0}; // `-include FILE`: force-included before the main file (see below)
  int ai = 1;
  while (ai < argc && argv[ai][0] == '-' && argv[ai][1]) {
    if (argv[ai][1] == 'I') {
      // Multiple `-I` dirs, searched in the order given (first `-I` searched first). One dir was
      // enough for a self-contained program, but compiling chibicc's *own* TUs pulls chibicc.h's
      // system-header closure, which spans chibicc's bundled `include/` **and** the system roots
      // (`/usr/include[/x86_64-linux-gnu]`) — the memfs seed mounts each at its real path and the
      // search order mirrors native chibicc's `add_default_include_paths` (bundled first).
      strarray_push(&include_paths, argv[ai] + 2);
      n_inc++;
      ai++;
    } else if (!strcmp(argv[ai], "-include") && ai + 1 < argc) {
      // `-include FILE` — tokenized and prepended to the main file (below), like `main.c`'s cc1().
      // Needed to compile chibicc's own TUs: they call `strtoul`/`atoi`/… whose modern-glibc
      // `<stdlib.h>` declarations chibicc's parser can't ingest, so the self-host prelude
      // (`selfhost_prelude.h`) supplies clean ones — exactly as `emit_object_real` builds the guest.
      strarray_push(&opt_include, argv[ai + 1]);
      ai += 2;
    } else if (!strcmp(argv[ai], "--data-page") && ai + 1 < argc) {
      opt_data_page = atoi(argv[ai + 1]);
      ai += 2;
    } else if (!strcmp(argv[ai], "--emit-object")) {
      opt_emit_object = true; // emit a linkable unit (cc -c) — see codegen_ir.c
      ai++;
    } else if (!strcmp(argv[ai], "-g")) {
      opt_g = true;
      ai++;
    } else if (!strcmp(argv[ai], "-g0")) {
      opt_g = false;
      ai++;
    } else {
      break;
    }
  }
  if (argc <= ai) {
    fprintf(stderr, "usage: chibicc [-Idir] <in.c> [out]\n");
    return 2;
  }
  base_file = argv[ai];
  if (n_inc == 0)
    strarray_push(&include_paths, "/include"); // default mount when no -I is given (existing tests)

  // Define the predefined macros (`__FILE__`, `__LINE__`, `__STDC__`, `__SIZEOF_*`, `__linux__`, …)
  // so real programs and glibc-shape headers (e.g. `<assert.h>`'s `__FILE__`/`__LINE__`) work. Safe
  // in the sandbox: init_macros' only host dependencies — `time`/`localtime`/`ctime_r`/`stat` for
  // `__DATE__`/`__TIME__`/`__TIMESTAMP__` — are stubbed in chibicc_extra.c (fixed 1970 epoch).
  init_macros();

  // Force-include each `-include FILE` ahead of the main file, then tokenize the main file and append
  // it — one token stream preprocessed as a whole, matching `main.c`'s cc1(). The prelude is
  // declarations only, so it emits no IR; the main file still starts at its own line 1, keeping
  // `__FILE__`/`__LINE__` (and thus the emitted IR) identical to a native `cc1_main` build.
  Token *tok = NULL;
  for (int i = 0; i < opt_include.len; i++) {
    char *incl = opt_include.data[i];
    char *path = file_exists(incl) ? incl : search_include_paths(incl);
    if (!path) {
      fprintf(stderr, "-include: %s: cannot open file\n", incl);
      return 1;
    }
    Token *ti = tokenize_file(path);
    if (!ti) {
      fprintf(stderr, "%s: tokenize failed\n", path);
      return 1;
    }
    if (!tok || tok->kind == TK_EOF) {
      tok = ti;
    } else {
      Token *e = tok;
      while (e->next->kind != TK_EOF)
        e = e->next;
      e->next = ti;
    }
  }
  Token *tmain = tokenize_file(base_file);
  if (!tmain) {
    fprintf(stderr, "%s: tokenize failed\n", base_file);
    return 1;
  }
  if (!tok || tok->kind == TK_EOF) {
    tok = tmain;
  } else {
    Token *e = tok;
    while (e->next->kind != TK_EOF)
      e = e->next;
    e->next = tmain;
  }
  tok = preprocess(tok);
  Obj *prog = parse(tok);

  FILE *out = stdout;
  if (argc > ai + 1 && strcmp(argv[ai + 1], "-") != 0) {
    out = fopen(argv[ai + 1], "w");
    if (!out) {
      fprintf(stderr, "%s: cannot open for writing\n", argv[ai + 1]);
      return 1;
    }
  }
  codegen_ir(prog, out);
  if (out != stdout)
    fclose(out);
  return 0;
}
