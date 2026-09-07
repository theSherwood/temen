/* Headless frame-hash differential (the §18 oracle, the doom_diff.c shape): read a ROM — or Uxntal
 * source, assembled by the in-guest uxnasm_core.c (text: an ASCII first byte and no NUL among the first
 * 256 bytes — a ROM's reset vector has a `00` operand or padding within that) — from stdin, run its
 * reset vector, then UXN_DIFF_FRAMES frames with a fixed key script, printing an FNV-1a hash of
 * every composed frame. Built BOTH as a Temen guest (clang → on-ramp) and as a native `cc` binary from
 * this one file — only `read`/`write`/`malloc`, which the on-ramp provides — so the two hash streams
 * must be identical. Driven by the `uxn_diff` test (crates/temen-llvm/tests/uxn_diff.rs). */
#include "uxn.c"
#include "varvara.c"
#include "uxnasm_core.c"

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

/* The input script: (frame, kind, a, b, c) — kind 0 = controller (button mask, key byte), 1 = mouse
 * (x, y, state), 2 = wheel (dx, dy). Arrows held for a stretch, key taps, clicks, a wheel notch. */
static const struct { int frame, kind, a, b, c; } script[] = {
  {5, 0, 0x80, 0, 0},   {25, 0, 0xa0, 0, 0},  {40, 0, 0x20, 0, 0},  {55, 0, 0x00, 0, 0},
  {60, 0, 0x00, 'a', 0}, {70, 0, 0x40, 0, 0},  {75, 1, 200, 100, 0}, {76, 1, 200, 100, 1},
  {77, 1, 200, 100, 0},  {90, 0, 0x50, 0, 0},  {95, 2, 0, 1, 0},     {100, 0, 0x00, 0, 0},
  {105, 0, 0x00, ' ', 0}, {110, 1, 30, 150, 1}, {111, 1, 30, 150, 0},
};

int main(void) {
  u.ram = malloc(UXN_BANKS * 0x10000);
  for (long i = 0; i < UXN_BANKS * 0x10000; i++) u.ram[i] = 0;
  varvara_init(&u);
  long got = 0, n;
  char *in = malloc(1 << 20);
  while (got < (1 << 20) && (n = read(0, in + got, (unsigned long)((1 << 20) - got))) > 0) got += n;
  int text = got > 0 && (Uint8)in[0] < 0x80;
  for (long i = 0; i < got && i < 256; i++)
    if (in[i] == 0) text = 0;
  if (text) {
    int rom_len;
    if (!uxnasm_assemble(in, (int)got, u.ram, &rom_len)) {
      write(1, "uxnasm: ", 8);
      write(1, uxnasm_error, (unsigned long)s_len(uxnasm_error));
      write(1, "\n", 1);
      return 1;
    }
  } else {
    for (long i = 0; i < got && i < 0xff00; i++) u.ram[0x100 + i] = (Uint8)in[i];
  }
  free(in);
  uxn_eval(&u, 0x100);
  int si = 0, w, h;
  for (int f = 0; f < UXN_DIFF_FRAMES && !varvara_halted(&u); f++) {
    while (si < (int)(sizeof script / sizeof script[0]) && script[si].frame == f) {
      if (script[si].kind == 0) varvara_controller(&u, (Uint8)script[si].a, (Uint8)script[si].b);
      else if (script[si].kind == 1) varvara_mouse(&u, script[si].a, script[si].b, (Uint8)script[si].c);
      else varvara_wheel(&u, script[si].a, script[si].b);
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
