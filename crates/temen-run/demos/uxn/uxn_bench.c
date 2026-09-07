/* Headless benchmark driver: read a ROM from stdin, run UXN_BENCH_FRAMES frames (no input), print one
 * line — the frame count and an FNV-1a hash of the LAST composed frame, so a run is checkable for
 * determinism across engines (the same hash on native, bytecode, Cranelift JIT, wasm) while the frame
 * loop itself is unmeasured by anything but the wall clock outside. Built as a Temen guest and natively
 * from this one file (only `read`/`write`/`malloc`), like uxn_diff.c. Driven by
 * crates/temen-llvm/examples/uxn_bench.rs. */
#include "uxn.c"
#include "varvara.c"

extern long read(int fd, void *buf, unsigned long n);
extern long write(int fd, const void *buf, unsigned long n);

#ifndef UXN_BENCH_FRAMES
#define UXN_BENCH_FRAMES 600
#endif

static Uxn u;

void varvara_console_write(const Uint8 *buf, int len) { write(1, buf, (unsigned long)len); }

static void print_hex(Uint32 v) {
  static const char hex[] = "0123456789abcdef";
  char s[8];
  for (int i = 7; i >= 0; i--, v >>= 4) s[i] = hex[v & 0xf];
  write(1, s, 8);
}
static void print_dec(int v) {
  char s[12];
  int n = 12;
  do { s[--n] = (char)('0' + v % 10); v /= 10; } while (v);
  write(1, s + n, (unsigned long)(12 - n));
}

int main(void) {
  u.ram = malloc(UXN_BANKS * 0x10000);
  for (long i = 0; i < UXN_BANKS * 0x10000; i++) u.ram[i] = 0;
  varvara_init(&u);
  long got = 0, n;
  while (got < 0xff00 && (n = read(0, u.ram + 0x100 + got, (unsigned long)(0xff00 - got))) > 0) got += n;
  uxn_eval(&u, 0x100);
  int w = 0, h = 0, frames = 0;
  const Uint8 *rgba = 0;
  for (; frames < UXN_BENCH_FRAMES && !varvara_halted(&u); frames++) {
    varvara_screen_vector(&u);
    rgba = varvara_frame(1, &w, &h);
  }
  Uint32 hash = 2166136261u;
  for (int i = 0; rgba && i < w * h * 4; i++) hash = (hash ^ rgba[i]) * 16777619u;
  write(1, "frames ", 7);
  print_dec(frames);
  write(1, " last ", 6);
  print_hex(hash);
  write(1, "\n", 1);
  return 0;
}
