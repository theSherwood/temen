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
bool opt_g = true; // -g always on: the W1 debugger consumes the waist (SELFHOST_C.md §5 A)
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
// Caught by the step-2 stub report (build_chibicc_svmb.sh step 3a).
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
// -g0 turns off debug info (`-g` is on by default for the W1 debugger); the playground passes it since
// it never debugs the compiled program, and the `debug.*` sections are ~a third of the emitted IR.
int main(int argc, char **argv) {
  char *inc = "/include";
  int ai = 1;
  while (ai < argc && argv[ai][0] == '-' && argv[ai][1]) {
    if (argv[ai][1] == 'I') {
      inc = argv[ai] + 2;
      ai++;
    } else if (!strcmp(argv[ai], "--data-page") && ai + 1 < argc) {
      opt_data_page = atoi(argv[ai + 1]);
      ai += 2;
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
  strarray_push(&include_paths, inc);

  Token *tok = tokenize_file(base_file);
  if (!tok) {
    fprintf(stderr, "%s: tokenize failed\n", base_file);
    return 1;
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
