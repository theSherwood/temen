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

// usage: chibicc <in.c> [out]   ("-"/absent → stdout). Include path fixed to /include — the
// memfs seed mounts frontend/chibicc/include there (§5 C).
int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: chibicc <in.c> [out]\n");
    return 2;
  }
  base_file = argv[1];
  strarray_push(&include_paths, "/include");

  Token *tok = tokenize_file(base_file);
  if (!tok) {
    fprintf(stderr, "%s: tokenize failed\n", base_file);
    return 1;
  }
  tok = preprocess(tok);
  Obj *prog = parse(tok);

  FILE *out = stdout;
  if (argc >= 3 && strcmp(argv[2], "-") != 0) {
    out = fopen(argv[2], "w");
    if (!out) {
      fprintf(stderr, "%s: cannot open for writing\n", argv[2]);
      return 1;
    }
  }
  codegen_ir(prog, out);
  if (out != stdout)
    fclose(out);
  return 0;
}
