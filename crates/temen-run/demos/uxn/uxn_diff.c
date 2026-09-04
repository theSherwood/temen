/* Headless frame-hash differential (the §18 oracle, the doom_diff.c shape): read a ROM from stdin,
 * run its reset vector, then UXN_DIFF_FRAMES frames with a fixed key script, printing an FNV-1a hash of
 * every composed frame. Built BOTH as a Temen guest (clang → on-ramp) and as a native `cc` binary from
 * this one file — only `read`/`write`/`malloc`, which the on-ramp provides — so the two hash streams
 * must be identical. Driven by the `uxn_diff` test (crates/temen-llvm/tests/uxn_diff.rs). */
#include "uxn.c"
#include "varvara.c"

extern long read(int fd, void *buf, unsigned long n);
extern long write(int fd, const void *buf, unsigned long n);

#ifndef UXN_DIFF_FRAMES
#define UXN_DIFF_FRAMES 120
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

/* The key script: (frame, button mask, key byte) — arrows held for a stretch, a couple of key taps. */
static const struct { int frame; Uint8 button, key; } script[] = {
  {5, 0x80, 0}, {25, 0xa0, 0}, {40, 0x20, 0}, {55, 0x00, 0}, {60, 0x00, 'a'}, {70, 0x40, 0},
  {90, 0x50, 0}, {100, 0x00, 0}, {105, 0x00, ' '},
};

int main(void) {
  u.ram = malloc(UXN_BANKS * 0x10000);
  for (long i = 0; i < UXN_BANKS * 0x10000; i++) u.ram[i] = 0;
  varvara_init(&u);
  long got = 0, n;
  while (got < 0xff00 && (n = read(0, u.ram + 0x100 + got, (unsigned long)(0xff00 - got))) > 0) got += n;
  uxn_eval(&u, 0x100);
  int si = 0, w, h;
  for (int f = 0; f < UXN_DIFF_FRAMES && !varvara_halted(&u); f++) {
    while (si < (int)(sizeof script / sizeof script[0]) && script[si].frame == f) {
      varvara_controller(&u, script[si].button, script[si].key);
      si++;
    }
    varvara_screen_vector(&u);
    const Uint8 *rgba = varvara_frame(1, &w, &h);
    Uint32 hash = 2166136261u;
    for (int i = 0; i < w * h * 4; i++) hash = (hash ^ rgba[i]) * 16777619u;
    write(1, "frame ", 6);
    print_dec(f);
    write(1, " ", 1);
    print_hex(hash);
    write(1, "\n", 1);
  }
  return 0;
}
