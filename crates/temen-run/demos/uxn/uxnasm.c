/* uxnasm — the native build-time front-end over uxnasm_core.c (the guest uses the core directly):
 *   cc -O2 -o uxnasm uxnasm.c && ./uxnasm demo.tal demo.rom */
#include <stdio.h>
#include <stdlib.h>
#include "uxnasm_core.c"

static Uint8 rom[0x10000];

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: uxnasm in.tal out.rom\n"); return 2; }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 1; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *src = malloc((size_t)n + 1);
  if (fread(src, 1, (size_t)n, f) != (size_t)n) { perror("read"); return 1; }
  src[n] = 0;
  fclose(f);
  int rom_len;
  if (!uxnasm_assemble(src, (int)n, rom, &rom_len)) {
    fprintf(stderr, "uxnasm: line %d: %s\n", uxnasm_error_line, uxnasm_error);
    return 1;
  }
  FILE *o = fopen(argv[2], "wb");
  if (!o) { perror(argv[2]); return 1; }
  fwrite(rom + 0x100, 1, (size_t)rom_len, o);
  fclose(o);
  fprintf(stderr, "uxnasm: %s → %s (%d bytes, %d labels)\n", argv[1], argv[2], rom_len, A.nlabels);
  return 0;
}
