/* Golden opcode corpus check (a native host tool): run every program in `corpus/opcodes.corpus` on the CPU and
 * compare the end state — both stacks and an FNV-1a over memory + the device page — with the recorded
 * expectation. The expectations come from uxn5's spec-compliant core (uxn.wasm), so this pins uxn.c to
 * the reference semantics with no external dependency at test time: 300 random straight-line programs
 * (every non-control-flow opcode in every mode, including stack/memory wrap-around) plus hand-written
 * control-flow programs (every jump form, lambdas, subroutines) and a primes program. Line format:
 *   <program hex> <wst bytes hex | -> <rst bytes hex | -> <fnv32 hex>
 * Driven by crates/temen-llvm/tests/uxn_diff.rs (`cpu_matches_golden_corpus`). Exit 0 iff all match. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "uxn.c"

static Uint8 ram[UXN_BANKS * 0x10000];
Uint8 uxn_dei(Uxn *u, Uint8 port) { return u->dev[port]; }
void uxn_deo(Uxn *u, Uint8 port) { (void)u; (void)port; }

static int unhex(const char *s, Uint8 *out, int cap) {
  int n = 0;
  if (!strcmp(s, "-")) return 0;
  for (; s[0] && s[1] && n < cap; s += 2) {
    unsigned v;
    if (sscanf(s, "%2x", &v) != 1) return -1;
    out[n++] = (Uint8)v;
  }
  return n;
}

static void stack_hex(const Stack *s, char *out) {
  if (!s->ptr) { strcpy(out, "-"); return; }
  for (int i = 0; i < s->ptr; i++) sprintf(out + 2 * i, "%02x", s->dat[i]);
}

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: uxn_corpus corpus/opcodes.corpus\n"); return 2; }
  FILE *f = fopen(argv[1], "r");
  if (!f) { perror(argv[1]); return 2; }
  static char line[8192], prog[4096], wst[1024], rst[1024], fnv[16];
  static Uint8 code[2048];
  int lineno = 0, failures = 0;
  while (fgets(line, sizeof line, f)) {
    lineno++;
    if (sscanf(line, "%4095s %1023s %1023s %15s", prog, wst, rst, fnv) != 4) { fprintf(stderr, "line %d: malformed\n", lineno); return 2; }
    int n = unhex(prog, code, sizeof code);
    if (n <= 0) { fprintf(stderr, "line %d: bad program\n", lineno); return 2; }
    Uxn u;
    memset(&u, 0, sizeof u);
    memset(ram, 0, sizeof ram);
    memcpy(ram + 0x100, code, (size_t)n);
    u.ram = ram;
    uxn_eval(&u, 0x100);
    unsigned h = 2166136261u;
    for (int i = 0; i < 0x10000; i++) h = (h ^ ram[i]) * 16777619u;
    for (int i = 0; i < 256; i++) h = (h ^ u.dev[i]) * 16777619u;
    char got_wst[1024], got_rst[1024], got_fnv[16];
    stack_hex(&u.wst, got_wst);
    stack_hex(&u.rst, got_rst);
    sprintf(got_fnv, "%08x", h);
    if (strcmp(got_wst, wst) || strcmp(got_rst, rst) || strcmp(got_fnv, fnv)) {
      failures++;
      fprintf(stderr, "line %d: MISMATCH\n  program %s\n  wst want %s got %s\n  rst want %s got %s\n  fnv want %s got %s\n",
              lineno, prog, wst, got_wst, rst, got_rst, fnv, got_fnv);
    }
  }
  fclose(f);
  printf("uxn_corpus: %d programs, %d mismatches\n", lineno, failures);
  return failures ? 1 : 0;
}
