// Consumer that exercises the **emit-object libc's printf + stdio** (task #20 increment 2b): the
// `printf_emit.c` varargs engine (integer/string/hex/width/precision/flags) driven through both the
// direct `printf`/`snprintf` path and the `FILE*` path (`fprintf`, `open_memstream`). Linked against
// `emit_libc.c` and run through `_start` on interp==JIT under the powerbox; `main`'s return value and
// the stdout bytes are fixed oracles (see the test). Deliberately no floats and no file I/O — those
// are later increments (the float path needs a correctly-rounded dtoa; fd≥3 needs the 2c fs cap).
//
// Declares only what it calls (no system headers) so it compiles as a standalone emit-object unit; the
// definitions all come from the linked libc.
typedef unsigned long size_t;

int printf(const char *, ...);
int snprintf(char *, size_t, const char *, ...);
int fprintf(void *, const char *, ...); // FILE* as an opaque pointer
int fputc(int, void *);
void *open_memstream(char **, size_t *);
int fclose(void *);
size_t strlen(const char *);

extern void *stdout;

int main(void) {
  // 1) Integer/hex/octal radix + unsigned + long length modifiers.
  printf("%d %u %x %X %o\n", -42, 4000000000u, 255, 255, 64);
  printf("%ld %lu\n", -1234567890123L, 9876543210UL);

  // 2) Width, precision, and flags — the fiddly padding logic.
  printf("[%5d][%-5d][%05d][%+d][% d]\n", 42, 42, 42, 42, 42);
  printf("[%8.3d][%-8.3d][%.0d]\n", 7, 7, 0);
  printf("[%*d][%.*s]\n", 6, 99, 3, "truncated");

  // 3) Chars, strings, literal percent, and the null-string guard.
  printf("%c%c%c %s %% %s\n", 'a', 'b', 'c', "str", (char *)0);

  // 4) snprintf returns the would-be length and NUL-terminates within the bound.
  char buf[8];
  int n = snprintf(buf, sizeof buf, "%d-%d-%d", 111, 222, 333);
  printf("snlen=%d buf=%s\n", n, buf);

  // 5) The FILE* path: a memory stream built with fprintf/fputc, then read back and printed.
  char *mem = 0;
  size_t mlen = 0;
  void *ms = open_memstream(&mem, &mlen);
  fprintf(ms, "mem<%d,%s>", 7, "x");
  fputc('!', ms);
  fclose(ms);
  printf("stream=%s len=%d\n", mem, (int)mlen);

  // 6) And the plain fd path (fprintf to stdout).
  fprintf(stdout, "fd=%s\n", "stdout");

  return (int)strlen("hello"); // 5 — a small deterministic return
}
